"""动作解码与离线评测。
IDM：(z_{t-ih..t}, z_{t+1..t+H}) → 这 H 帧之间的 2H 个 20 Hz 动作（ih 帧历史提供当前速度：OSC 控制下同一动作的位移依赖速度）；BC 基线：(z_{t-3..t}, 语言) → 同一个动作块，不用世界模型。
对齐：帧 k = 环境第 2k 步执行动作之前的画面（render stride 2），帧 k→k+1 之间执行 a_{2k}, a_{2k+1}。
meta/actions.npy 只存了 a_{2k}，完整的 20 Hz 动作从原始 hdf5 读取，加载时和 meta 核对一遍。
骨干（两者相同）：逐 patch 把各帧特征按通道拼接 → 线性层 + 位置编码 → 几层自注意力 → 注意力池化（同 probe.py）→ MLP。
输出：每个动作 6 维连续量（按训练集逐维标准化，MSE）+ 夹爪 logit（BCE，推理时取符号得到 ±1）。
动作不是 7 维时（RoboTwin：14 维关节 qpos 路点，两个夹爪也是连续开度）全部维都按连续量处理：逐维标准化 + MSE，不取符号、闭环不截断（cont / clip）。
--wrist n：另外输入腕部相机 t-n+1..t 的真实帧（data/feats/<encoder>_wrist，自己的 mu/sd 存在 ckpt 的 wnorm 里），
作为另一组 T 个 token（自己的输入层和位置编码）；WM 不预测腕部画面，所以只用已观测到的帧。IDM 仍然不看语言。
--proprio k1,k2：本体感觉（data/meta/states/<k>.npy，如 robot0_eef_pos,robot0_gripper_qpos），取与图像历史相同的帧
（IDM t-ih..t），特征 = 当前值 + 相邻帧差分（显式速度），按训练帧逐维标准化（mu/sd 是网络 buffer，随 state_dict 存进 ckpt）→ 一个额外 token。
闭环 worker 从 env.step 返回的 obs 取同名 key：帧 k 的 obs = 执行 a_{2k-2}, a_{2k-1} 之后、a_{2k} 之前，与渲染时 regenerate_obs_from_state(states[2k]) 一致。
--lang：IDM 也输入指令（与 BC 相同的语言 token），检验"语言只进 WM"的设计；--txt_aug / --txt_noise σ（BC 与 --lang IDM）：
训练时随机取 --wm 的改写增广表里的一条、文本嵌入加相对 RMS 的高斯噪声（同 train.py tx()），val 用原指令。
--head flow：π0 式动作专家，用线性 flow matching 生成整个动作块（2H 个，6 维标准化连续量 + 夹爪 ±1 一起作为 7 维流输出，推理取符号）；
带噪动作 token（AdaLN(τ)）经交叉注意力读骨干输出（KV 每次决策只算一次），Euler --fsteps 步。--H 8 时条件为 WM 的整个 8 帧计划。
--wpf N（降频档，默认关）：IDM 的一帧 = N 个路点（数据根不变，按 delta = N/2 隔帧取），动作块 = N·H 个路点；配 train --delta N/2 的 WM，闭环每次决策执行 N 个路点。
--rtc_dmax D（训练时 RTC，默认关）：flow 头另学"已发出动作前缀"条件——训练时每个样本随机取前缀长度 d ∈ {0..D}，动作块前 d 个路点用真值、
层级 1（不加噪）、加零初始化的前缀掩码嵌入 pemb，只在其余位置算 flow 损失；推理时 Net.rtc_pre（closed_loop Policy.rtc_prefix 设）给前缀，
输出的前 d 个路点就是前缀、其余接着生成。不给前缀时与原网络逐位相同。

  python -m probwm.idm train --kind idm --ih 3 --out runs/goal/idm/idm.pt
  python -m probwm.idm train --kind bc --p_drop 0.1 --noise 0.2 --out runs/goal/idm/bc.pt
  python -m probwm.idm train --kind idm --ih 3 --d 512 --depth 4 --steps 4000 --wrist 1 --wm runs/goal/chunk8/ckpt.pt --out runs/goal/idm/idm_wrist_s4000.pt
  python -m probwm.idm train --kind idm --ih 3 --d 512 --depth 4 --steps 4000 --proprio robot0_eef_pos,robot0_gripper_qpos --wm runs/goal/chunk8/ckpt.pt --out runs/goal/idm/idmP_s4000.pt
  python -m probwm.idm train --kind idm --ih 3 --d 512 --depth 4 --steps 4000 --H 8 --head flow --wm runs/goal/chunk8/ckpt.pt --out runs/goal/idm/idmF8_s4000.pt
  python -m probwm.idm eval --idm runs/goal/idm/idm.pt --bc runs/goal/idm/bc.pt   # 结果写到 idm ckpt 同目录 offline.json
"""
import argparse
import copy
import json
import math
import os
import time
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fm
from .model import Block, CBlock, modulate, rope_table, tsin
from .train import build, load_data

STRIDE = 2  # 渲染步长：1 帧 = 2 个环境步
SAMPLERS = {"scratch4": dict(steps=4), "scratch10": dict(steps=10), "scratch2": dict(steps=2), "scratch3": dict(steps=3),
            "scratch1": dict(steps=1),  # (ctrl) 从纯噪声一步 Euler 到 τ=1：x = ε + u(ε, 0) ≈ E[x1 | h, c]（flow 的一步均值，确定性推理对照）
            "warm": dict(tau0=0.2, n_lan=5, snr=0.4, steps=4),
            "pc": dict(tau0=0.2, n_lan=5, snr=0.4, steps=4, pc=True),          # 对照：Langevin 但不用 z_t
            "rtc3": dict(tau0=0.3, n_lan=5, snr=0.4, steps=4),                  # chunk 模型：上一次计划平移后热启动
            "rtc5": dict(tau0=0.5, n_lan=5, snr=0.4, steps=4),
            "warm0": dict(tau0=0.3, n_lan=0, steps=4),
            "warm1": dict(tau0=0.1, n_lan=5, snr=0.4, steps=4),     # 你的方案：轻度加噪（90% 噪声）+ Langevin
            "warm1k20": dict(tau0=0.1, n_lan=20, snr=0.4, steps=4)}  # 训练时模拟热启动的模型：不做 Langevin  # WM 采样器（fm.sample 的参数）


def pfeat(zp):  # 本体感觉历史 (B,nf,dp) 原始值 → 当前值 + 相邻帧差分 (B,nf·dp)
    return torch.cat([zp[:, -1], torch.diff(zp, dim=1).flatten(1)], 1)


class Net(nn.Module):
    def __init__(s, D, T, n_in, n_act, d_txt=0, d=256, depth=2, heads=4, nq=4, p_drop=0.0, n_w=0, n_p=0, flow=False, T_w=0, rope=False, ad=7, kfd=False, aux_pd=0, ntau=0, nh=0, dhid=0, rtc=False):
        super().__init__()
        if rtc:  # 训练时 RTC（idm train --rtc_dmax）：动作块前缀位置（已发出的路点）的掩码嵌入，零初始化（不给前缀时与原模型同构）；只接 flow 头
            assert flow, "--rtc_dmax 只接 flow 头 IDM"
            s.pemb = nn.Parameter(torch.zeros(d))
        if nh:  # (vpp) 读冻结 WM 的近槽隐状态（--vpp）：nh 个槽 × 每 patch dhid 维（WM 宽度），逐槽无仿射 LayerNorm 后按通道拼接 → 线性层，逐 patch 加到骨干输入
            s.nh, s.hln, s.hin = nh, nn.LayerNorm(dhid, elementwise_affine=False), nn.Linear(nh * dhid, d)
        if ntau:  # (xden) 噪声感知 IDM（--tau_in）：计划各槽的流匹配层级 τ（τ=1 干净）→ 逐 patch 加到骨干输入；最后一层零初始化
            s.ntau, s.tauemb = ntau, nn.Sequential(nn.Linear(256 * ntau, d), nn.SiLU(), nn.Linear(d, d))
            nn.init.zeros_(s.tauemb[-1].weight), nn.init.zeros_(s.tauemb[-1].bias)
        if kfd:  # 关键帧目标 IDM（--kf_delta）：到关键帧还有几帧（0..64，10 Hz）→ 一个 token
            s.dk = nn.Embedding(65, d)
        if aux_pd:  # 辅助稠密监督（--aux_dense）：agentview 的 patch token → 未来 8 帧特征的低维投影（固定随机投影 D → aux_pd）
            s.auxh = nn.Linear(d, 8 * aux_pd)
            s.register_buffer("aproj", torch.randn(D, aux_pd, generator=torch.Generator().manual_seed(0)) / D ** 0.5)
            s.lam_aux = 1.0
        s.n_act, s.flow, s.steps, s.p_pdrop, s.ad = n_act, flow, 10, 0.0, ad  # ad：动作维数（LIBERO / RoboCasa 7；RoboTwin 14 维 qpos）
        s.inp, s.pos = nn.Linear(n_in * D, d), nn.Parameter(torch.randn(1, T, d) * 0.02)
        s.rope, s.dh = rope, d // heads
        if rope:  # 2D RoPE（idm --rope）：去掉 patch 的绝对位置嵌入，编码层换成带 RoPE 的自注意力（模型同 WM 的 CBlock，无 dropout）
            del s.pos
            s.register_buffer("pos", torch.zeros(1, T, d), persistent=False)
        if n_w:  # 腕部相机 n_w 帧：另一组 token（T_w 个，可少于 agentview，如 2×2 池化后的 64；0 = 同 T）
            s.win, s.wpos = nn.Linear(n_w * D, d), nn.Parameter(torch.randn(1, T_w or T, d) * 0.02)
        if n_p:  # 本体感觉：n_p 维特征（pfeat）→ 一个 token；逐维 mu/sd 作 buffer（训练时用训练帧填好）
            s.pin, s.ppos = nn.Linear(n_p, d), nn.Parameter(torch.randn(1, 1, d) * 0.02)
            s.register_buffer("pmu", torch.zeros(n_p)), s.register_buffer("psd", torch.ones(n_p))
        s.txt = nn.Linear(d_txt, d) if d_txt else None
        s.blocks = nn.ModuleList(CBlock(d, heads) for _ in range(depth)) if rope else nn.ModuleList(nn.TransformerEncoderLayer(d, heads, 4 * d, p_drop, "gelu", batch_first=True,
                                                            norm_first=True) for _ in range(depth))
        if flow:  # 动作专家：与 WM 同款 Block（AdaLN(τ) 自注意力 + 交叉注意力 + MLP），层数同骨干
            s.hn0, s.ain, s.apos = nn.LayerNorm(d), nn.Linear(ad, d), nn.Parameter(torch.randn(1, n_act, d) * 0.02)
            s.temb = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
            s.fblocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
            s.hn, s.hada, s.head = nn.LayerNorm(d, elementwise_affine=False), nn.Linear(d, 2 * d), nn.Linear(d, ad)
            for m in [*(b.ada for b in s.fblocks), s.hada, s.head]:
                nn.init.zeros_(m.weight), nn.init.zeros_(m.bias)
            return
        s.q, s.att = nn.Parameter(torch.randn(1, nq, d) * 0.02), nn.MultiheadAttention(d, heads, batch_first=True)
        s.mlp = nn.Sequential(nn.LayerNorm(nq * d), nn.Dropout(p_drop), nn.Linear(nq * d, 2 * d), nn.GELU(),
                              nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, n_act * ad))

    def forward(s, zs, txt=None, tmask=None, zw=None, zp=None, y=None, dk=None, yaux=None, tau=None, zh=None, pd=None):
        """zs (B,n_in,T,D) → (B,n_act,7)：前 6 维为标准化的连续动作，第 7 维为夹爪 logit（flow 头：夹爪值，取符号）。语言作为一个额外 token。
        zw (B,n_w,T,D)：腕部相机帧；zp (B,nf,dp)：本体感觉原始值（各自只在对应网络上使用）。
        flow 头：给 y（标准化动作块，夹爪 ±1）时返回 flow matching 损失，否则从噪声 Euler s.steps 步采样。
        tau (B,ntau)：(xden) 噪声感知 IDM 的计划各槽层级（不给 = 全 1 = 干净计划）；没有 tauemb 的网络忽略。
        zh (B,nh,T,dhid)：(vpp) WM 的近槽隐状态（只有 --vpp 的网络读；全 0 = 去掉隐状态的对照）。"""
        h = s.inp(zs.transpose(1, 2).flatten(2).to(s.inp.weight.dtype)) + s.pos
        if hasattr(s, "hin"):
            h = h + s.hin(s.hln(zh).transpose(1, 2).flatten(2).to(h.dtype)).to(h.dtype)  # 调用方都在 bf16 autocast 里（隐状态是 bf16）
        if hasattr(s, "tauemb"):
            tt = torch.ones(len(h), s.ntau, device=h.device) if tau is None else tau.float()
            h = h + s.tauemb(tsin(tt.flatten()).view(len(h), -1).to(s.tauemb[0].weight.dtype))[:, None].to(h.dtype)
        if hasattr(s, "win"):
            h = torch.cat([h, s.win(zw.transpose(1, 2).flatten(2).to(s.win.weight.dtype)) + s.wpos], 1)
        if hasattr(s, "pin"):
            p = (pfeat(zp.float()) - s.pmu) / s.psd
            if s.training and s.p_pdrop > 0:  # 整段丢弃（置为均值），逼 IDM 以未来帧为主
                p = torch.where(torch.rand(len(p), 1, device=p.device) < s.p_pdrop, torch.zeros_like(p), p)
            h = torch.cat([h, (s.pin(p.to(s.pin.weight.dtype))[:, None] + s.ppos).to(h.dtype)], 1)
        if dk is not None and hasattr(s, "dk"):
            h = torch.cat([h, s.dk(dk.clamp(0, 64).long())[:, None].to(h.dtype)], 1)
        if s.txt is not None:
            m = tmask.float()[..., None]
            h = torch.cat([h, s.txt((txt.float() * m).sum(1) / m.sum(1).clamp_min(1))[:, None].to(h.dtype)], 1)
        if s.rope:
            ns = (1 + hasattr(s, "win")) * s.pos.shape[1]  # 网格 token：agentview（+ 腕部，各用自己的网格），其余（本体感觉 / 语言）不带位置
            pe = rope_table(s.pos.shape[1], s.dh, ns // s.pos.shape[1], h.shape[1] - ns, h.device)
            for b in s.blocks:
                h = b(h, pe)
        else:
            for b in s.blocks:
                h = b(h)
        if s.flow:
            kv = [b.ca.kvs(s.hn0(h)) for b in s.fblocks]
            if y is not None:
                l = fm.loss(s.vel, kv, y, "abs") if pd is None else s.rtc_loss(kv, y, pd)  # pd (B,)：各样本的前缀长度（--rtc_dmax 训练）
                if yaux is not None and hasattr(s, "auxh"):  # yaux (B,8,T,aux_pd)：真实未来 8 帧特征 @ aproj
                    T = s.pos.shape[1]
                    pa = s.auxh(h[:, :T]).float().view(len(h), T, 8, -1).transpose(1, 2)
                    l = l + s.lam_aux * F.mse_loss(pa, yaux.float())
                return l
            # 推理时可选（调用方在 net 上设属性，默认都没有 → 原行为）：fixed_eps (B,n_act,ad) 给定的初始噪声；
            # warm_init = (z (B,n_act,ad) 标准化动作块, ok (B,) bool, τ0)：动作滚动——ok 的样本从 z 加噪到 τ0 起积分，其余从纯噪声。last_out 存本次输出。
            # rtc_pre = (P (B,d,ad) 标准化的已发出路点, dv (B,) 各样本前缀长度 ≤ d)：动作块前 dv 个位置固定为 P（rtc_sample）；None（默认）= 原行为。
            # Policy.rtc_prefix 只在至少一个样本 dv > 0 时才设（这里不读 dv，避免 GPU 同步）
            z0, x0, wi = h.new_zeros(len(h), s.n_act, s.ad, dtype=torch.float), getattr(s, "fixed_eps", None), getattr(s, "warm_init", None)
            rp = getattr(s, "rtc_pre", None)
            if rp is not None:
                assert wi is None, "rtc_pre 不与 warm_init（--act_roll）同用"
                out = s.rtc_sample(kv, rp[0], rp[1], x0)
            elif wi is None or not bool(wi[1].any()):
                out = fm.sample(s.vel, kv, z0, "abs", steps=s.steps, x0=x0)[0]
            else:
                ok = wi[1]
                sub = lambda m: [tuple(t[m] for t in kv_) for kv_ in kv]
                out = z0.clone()
                out[ok] = fm.sample(s.vel, sub(ok), wi[0][ok].float(), "abs", steps=s.steps, tau0=wi[2], x0=None if x0 is None else x0[ok])[0]
                if bool((~ok).any()):
                    out[~ok] = fm.sample(s.vel, sub(~ok), z0[~ok], "abs", steps=s.steps, x0=None if x0 is None else x0[~ok])[0]
            s.last_out = out
            return out
        y = s.att(s.q.expand(len(h), -1, -1).to(h.dtype), h, h, need_weights=False)[0]
        return s.mlp(y.flatten(1)).view(len(h), s.n_act, s.ad)

    def ctx(s, zs, txt=None, tmask=None, zw=None, zp=None, dk=None, tau=None, zh=None):
        """(rteasy 11-18) 推理用：flow 头 IDM 的骨干 → 各 fblock 的交叉注意力 kv（closed_loop --idm_graph 把它按输入形状捕获成 CUDA graph）。
        骨干逐行抄自 forward（训练路径不走这里）；只读输入、不抽随机数（eval 下 dropout / 本体丢弃都不生效）。"""
        assert s.flow and not s.training
        h = s.inp(zs.transpose(1, 2).flatten(2).to(s.inp.weight.dtype)) + s.pos
        if hasattr(s, "hin"):
            h = h + s.hin(s.hln(zh).transpose(1, 2).flatten(2).to(h.dtype)).to(h.dtype)  # 调用方都在 bf16 autocast 里（隐状态是 bf16）
        if hasattr(s, "tauemb"):
            tt = torch.ones(len(h), s.ntau, device=h.device) if tau is None else tau.float()
            h = h + s.tauemb(tsin(tt.flatten()).view(len(h), -1).to(s.tauemb[0].weight.dtype))[:, None].to(h.dtype)
        if hasattr(s, "win"):
            h = torch.cat([h, s.win(zw.transpose(1, 2).flatten(2).to(s.win.weight.dtype)) + s.wpos], 1)
        if hasattr(s, "pin"):
            p = (pfeat(zp.float()) - s.pmu) / s.psd
            if s.training and s.p_pdrop > 0:  # 整段丢弃（置为均值），逼 IDM 以未来帧为主
                p = torch.where(torch.rand(len(p), 1, device=p.device) < s.p_pdrop, torch.zeros_like(p), p)
            h = torch.cat([h, (s.pin(p.to(s.pin.weight.dtype))[:, None] + s.ppos).to(h.dtype)], 1)
        if dk is not None and hasattr(s, "dk"):
            h = torch.cat([h, s.dk(dk.clamp(0, 64).long())[:, None].to(h.dtype)], 1)
        if s.txt is not None:
            m = tmask.float()[..., None]
            h = torch.cat([h, s.txt((txt.float() * m).sum(1) / m.sum(1).clamp_min(1))[:, None].to(h.dtype)], 1)
        if s.rope:
            ns = (1 + hasattr(s, "win")) * s.pos.shape[1]  # 网格 token：agentview（+ 腕部，各用自己的网格），其余（本体感觉 / 语言）不带位置
            pe = rope_table(s.pos.shape[1], s.dh, ns // s.pos.shape[1], h.shape[1] - ns, h.device)
            for b in s.blocks:
                h = b(h, pe)
        else:
            for b in s.blocks:
                h = b(h)
        return [b.ca.kvs(s.hn0(h)) for b in s.fblocks]

    def infer(s, zs, txt=None, tmask=None, zw=None, zp=None, dk=None, tau=None, zh=None):
        """(rteasy 11-18) closed_loop --idm_graph 的推理入口（net.forward 换成它）：kv 由 s.ctx 出（可被 CUDA graph 捕获），采样同 forward 的推理分支（噪声在外面照常抽）。"""
        kv = s.ctx(zs, txt, tmask, zw, zp, dk, tau, zh)
        z0, x0, wi = zs.new_zeros(len(zs), s.n_act, s.ad, dtype=torch.float), getattr(s, "fixed_eps", None), getattr(s, "warm_init", None)
        rp = getattr(s, "rtc_pre", None)  # 同 forward 的推理分支
        if rp is not None:
            assert wi is None, "rtc_pre 不与 warm_init（--act_roll）同用"
            out = s.rtc_sample(kv, rp[0], rp[1], x0)
        elif wi is None or not bool(wi[1].any()):
            out = fm.sample(s.vel, kv, z0, "abs", steps=s.steps, x0=x0)[0]
        else:
            ok = wi[1]
            sub = lambda m: [tuple(t[m] for t in kv_) for kv_ in kv]
            out = z0.clone()
            out[ok] = fm.sample(s.vel, sub(ok), wi[0][ok].float(), "abs", steps=s.steps, tau0=wi[2], x0=None if x0 is None else x0[ok])[0]
            if bool((~ok).any()):
                out[~ok] = fm.sample(s.vel, sub(~ok), z0[~ok], "abs", steps=s.steps, x0=None if x0 is None else x0[~ok])[0]
        s.last_out = out
        return out

    def vel(s, x, t, kv, *_, pm=None):  # flow 头的速度场：x (B,n_act,7) 带噪动作块，t (B,) → (B,n_act,7)
        # t 也可以是 (B,n_act) 逐位置层级（AdaLN 逐 token，model.modulate / Block 支持）；pm (B,n_act) 前缀掩码（1 = 已发出的路点）→ 加 pemb。默认原行为
        g = s.ain(x.to(s.ain.weight.dtype)) + s.apos
        if pm is not None:
            g = g + pm[..., None].to(g.dtype) * s.pemb.to(g.dtype)
        c = s.temb(tsin(t.flatten()).to(g.dtype)).view(*t.shape, -1) if t.dim() == 2 else s.temb(tsin(t).to(g.dtype))
        for b, k in zip(s.fblocks, kv):
            g = b(g, c, k)
        sh, sc = s.hada(c).chunk(2, -1)
        return s.head(modulate(s.hn(g), sh, sc))

    def rtc_loss(s, kv, y, pd):
        """训练时 RTC 的 flow 损失：y (B,n_act,ad) 标准化动作块，pd (B,) 各样本的前缀长度 d（已发出的路点数）。
        层级抽法同 fm.loss（abs、α 3、30% 均匀）；前 d 个位置层级置 1（x_τ = y，不加噪）并加 pemb，只在其余位置算损失（逐 token 平均）。d = 0 的样本与原损失同分布。"""
        B, N = y.shape[:2]
        u = torch.rand(B, device=y.device)
        t = torch.where(torch.rand(B, device=y.device) < 0.3, u, fm.shift(u, 3.0))
        pm = (torch.arange(N, device=y.device)[None] < pd[:, None]).float()
        tt = torch.where(pm > 0, 1.0, t[:, None])
        e = torch.randn_like(y)
        xt, v = (1 - tt[..., None]) * e + tt[..., None] * y, y - e
        w = 1 - pm
        den = getattr(s, "rtc_den", None)  # idm train --accum：整批的非前缀位置总数（各微批按整批归一，累加后等于整批损失）；默认 None = 本批自己的
        return (((s.vel(xt, tt, kv, pm=pm).float() - v.float()) ** 2).mean(-1) * w).sum() / (w.sum().clamp_min(1) if den is None else den)

    def rtc_sample(s, kv, P, dv, x0=None):
        """带前缀的采样：P (B,d,ad) 标准化的已发出路点、dv (B,) 各样本有效前缀长度（0 = 无前缀，与原采样同分布）。
        Euler 网格同 fm.sample（τ0 = 0、α 3、s.steps 步，初始噪声照常抽或用 x0）；前缀位置每步都写回 P、层级 1 + pemb，输出的前 dv 个位置就是 P。"""
        B, N, dev = len(dv), s.n_act, dv.device
        pm = torch.arange(N, device=dev)[None] < dv[:, None]
        Pf = torch.zeros(B, N, s.ad, device=dev)
        k = min(N, P.shape[1])
        Pf[:, :k] = P[:, :k].float()
        m3, pmf = pm[..., None], pm.float()
        x = torch.randn(B, N, s.ad, device=dev) if x0 is None else x0.float()
        x = torch.where(m3, Pf, x)
        ts = fm.shift(torch.linspace(0, 1, s.steps + 1), 3.0).tolist()
        for t0, t1 in zip(ts[:-1], ts[1:]):
            v = s.vel(x, torch.full((B, N), t0, device=dev).masked_fill(pm, 1.0), kv, pm=pmf)
            x = torch.where(m3, Pf, x + (t1 - t0) * v)
        return x


def next_contact(data, root):
    """每帧之后第一次夹爪指令翻转的那一帧（没有就取 demo 最后一帧）的帧号（data 的帧序）；同 train.py --keyframe 的目标。"""
    gr = np.load(Path(root) / "meta/actions.npy", mmap_mode="r")[data.src][:, 6] > 0
    nxt = np.empty(len(gr), np.int64)
    for st, ln in zip(data.start.tolist(), data.length.tolist()):
        g = gr[st:st + ln]
        ev = np.nonzero(g[1:] != g[:-1])[0] + 1
        j = np.searchsorted(ev, np.arange(ln), side="right")
        nxt[st:st + ln] = st + (np.where(j < len(ev), ev[np.minimum(j, len(ev) - 1)], ln - 1) if len(ev) else ln - 1)
    return torch.from_numpy(nxt).to(data.device)


def fars(a):  # --far "8" / "4,8" → [8] / [4, 8]；0 或空 = 普通 IDM
    return [int(x) for x in str(getattr(a, "far", 0) or 0).split(",") if int(x) > 0]


def fextra(a):  # --far_extra "16,24,32"：稠密近处 t+1..t+H 之后再读这几个远时距帧（MS-far WM 的远槽；超出 demo 末尾取最后一帧）
    return [int(x) for x in str(getattr(a, "far_extra", "") or "").split(",") if x]


def offsets(a, K):  # 输入帧相对 t 的偏移（单位：WM 步）；--far k[,k2]：未来输入只有这几帧（目标条件 IDM，配远点 / 关键帧 WM）
    if a.kind == "idm" and getattr(a, "vpp", "") == "hid":  # (vpp) 只读隐状态：帧输入只有历史（未来的信息全从 WM 隐状态来）
        return torch.arange(-a.ih, 1)
    if a.kind == "idm" and fars(a):
        return torch.cat([torch.arange(-a.ih, 1), torch.tensor(fars(a))])
    if a.kind == "idm" and fextra(a):
        return torch.cat([torch.arange(-a.ih, a.H + 1), torch.tensor(fextra(a))])
    return torch.arange(-a.ih, a.H + 1) if a.kind == "idm" else torch.arange(-(K - 1), 1)


def poffs(a, K):  # 本体感觉帧相对 t 的偏移：默认与图像历史相同；--ph n 只取 t-n+1..t（n=1：只有当前值，没有速度）；--pfut k 再加真实未来 t+1..t+k（离线探针）
    o = torch.arange(1 - a.ph, 1) if a.ph else torch.arange(-a.ih, 1) if a.kind == "idm" else torch.arange(-(K - 1), 1)
    return torch.cat([o, torch.arange(1, 1 + getattr(a, "pfut", 0))])


def tau_draw(B, H, spec, alpha, n, dev):
    """(xden) 噪声感知 IDM（--tau_in）训练时计划各槽的流匹配层级 τ (B,H)，τ=1 干净。spec = "p_clean,p_roll,p_sort"，其余概率 = 每个样本所有槽同一个层级。
    roll：滚动剖面 u_j = 1 − (j + φ)/n，φ ∈ (0,1]（φ=1 即闭环 --idm_tau pre 时 MS-far 窗口前 H 槽的起始层级，n = WM 的槽数）；
    sort：逐槽独立 U(0,1) 降序排（近槽轻、远槽重的任意单调剖面）；same：U(0,1)。τ = shift(u, α)（α 同 WM）。"""
    pc, pr, ps = (float(x) for x in spec.split(","))
    r, j = torch.rand(B, 1, device=dev), torch.arange(H, device=dev)
    u = torch.where(r < pc, torch.ones(B, H, device=dev),
                    torch.where(r < pc + pr, 1 - (j + 1 - torch.rand(B, 1, device=dev)) / n,
                                torch.where(r < pc + pr + ps, torch.rand(B, H, device=dev).sort(1, descending=True).values, torch.rand(B, 1, device=dev).expand(B, H))))
    return fm.shift(u.clamp(0, 1), alpha)


def tau_noise(z, tau, ih):  # (xden) z (B,n_in,T,D) 里的计划帧（第 ih+1 起 H 帧）换成 x_τ = (1 − τ)ε + τ·z（标准化空间，同 WM 的 abs 目标）
    H = tau.shape[1]
    zf, t = z[:, ih + 1:ih + 1 + H].float(), tau[:, :, None, None]
    return torch.cat([z[:, :ih + 1], ((1 - t) * torch.randn_like(zf) + t * zf).to(z.dtype), z[:, ih + 1 + H:]], 1)


def vpp_fp(wm):  # (vpp) WM 权重的指纹：idmVPP 绑定训练它的那个 WM，闭环核对（换 WM 要重训）
    return float(sum(p.detach().float().sum().item() for p in (wm.head[-1].weight, wm.blocks[-1].ada.weight, wm.blocks[0].sa.q.weight)))


def vpp_roll_tau(wm, wa, dev):  # (vpp) 闭环 --roll 1,0,8 的 MS-far 稳态：第 j 槽（时距由近到远）去噪前层级 τ_j = shift(1 − (j+1)/n, α)，msroll 这一步的 WM 前向就在这组 τ 上
    return fm.shift(1 - (torch.arange(wm.n_fut, device=dev, dtype=torch.float) + 1) / wm.n_fut, wa.alpha)


@torch.no_grad()
def vpp_hidden(wm, wa, data, f, H, g=None, trunk=None):
    """(vpp) 训练用的近槽隐状态：同闭环 msroll 那一步——各目标槽（wa.tgt_offs，超出 demo 末尾取最后一帧，同 train --far_clamp）的真实特征
    按滚动剖面加噪 x_j = τ_j z + (1 − τ_j) ε，以 K 帧真实历史（开头不足重复第一帧）+ 语言（增广表随机一条）为条件，WM 的条件路前向一次，
    取输出头之前的逐 token 特征（model.WM.trunk，最后一层 + 最终 AdaLN 调制）的前 H 个槽（t+1..t+H）。→ (B, H, T, d) bf16。g：固定噪声（val）；trunk：编译过的 wm.trunk。"""
    dev, T = f.device, wm.pos.shape[1]
    vo = torch.tensor([int(x) for x in wa.tgt_offs.split(",")], device=dev)
    c = zin(data, f, vo).float()  # zin 两端都截：历史开头重复第一帧、未来超出末尾取最后一帧
    vt = vpp_roll_tau(wm, wa, dev)
    t = vt[None, :, None, None]
    x = (t * c + (1 - t) * torch.randn(c.shape, device=dev, generator=g)).flatten(1, 2)
    tx_, tm_ = data.text(data.ftask[f], rand=g is None)
    with torch.autocast("cuda", torch.bfloat16):
        hh = (trunk or wm.trunk)(x, vt[None].expand(len(f), -1), wm.context(zin(data, f, torch.arange(-(wa.K - 1), 1)), tx_, tm_))
    return hh[:, :H * T].unflatten(1, (H, T)).to(torch.bfloat16)


def new_net(a, dims, K, n_act):
    D, T, Dt = dims
    net = Net(D, T, len(offsets(a, K)) + int(getattr(a, "kf_extra", False)), n_act, Dt if a.kind == "bc" or a.lang else 0, a.d, a.depth, p_drop=a.p_drop, n_w=len(woffs(a)) if a.wrist else 0, T_w=getattr(a, "wT", 0),
              n_p=a.pdim * len(poffs(a, K)), flow=a.head == "flow", rope=getattr(a, "rope", False), ad=getattr(a, "adim", 7),
              kfd=getattr(a, "kf_delta", False), aux_pd=getattr(a, "aux_pd", 64) if getattr(a, "aux_dense", 0.0) > 0 else 0,
              ntau=a.H if getattr(a, "tau_in", "") else 0, nh=a.H if getattr(a, "vpp", "") else 0, dhid=getattr(a, "vpp_d", 0),
              rtc=(getattr(a, "rtc_dmax", 0) or 0) > 0)
    net.steps, net.p_pdrop = a.fsteps, a.p_pdrop
    if getattr(a, "aux_dense", 0.0) > 0:
        net.lam_aux = a.aux_dense
    return net


def make(ck, dev):  # 旧 ckpt 没有 wrist / proprio / head → 关闭
    a = Namespace(**{"ih": 0, "p_drop": 0.0, "wrist": 0, "proprio": "", "pdim": 0, "p_pdrop": 0.0, "ph": 0, "head": "det", "fsteps": 10, "lang": False, "far": 0,
                     **ck["args"]})
    net = new_net(a, ck["dims"], ck["K"], ck["n_act"])
    net.load_state_dict(ck["model"])
    return net.to(dev).eval(), a, {k: v.to(dev) for k, v in ck["astats"].items()}


@torch.no_grad()
def pseudo(path, data, A, f, bs=2048):  # 用 IDM（真实未来帧）给 f 这些帧标动作 → 原始单位、夹爪 ±1，与闭环一样截到 [-1, 1]
    net, ia, st = make(torch.load(path, map_location="cpu"), data.device)
    offs = offsets(ia, data.K)
    for j in range(0, len(f), bs):
        with torch.autocast("cuda", torch.bfloat16):
            out = net(zin(data, f[j:j + bs], offs)).float()
        A[f[j:j + bs]] = clip(decode(out, st), st)
    return A


def cont(st):  # 全连续动作（RoboTwin：14 维关节 qpos，夹爪也是连续开度）：astats 的 mu 覆盖全部维；LIBERO / RoboCasa 的 mu 只有前 6 维
    return st["mu"].shape[-1] != 6


def decode(out, st):  # → 原始动作单位；夹爪 ±1（全连续时逐维反标准化）
    if cont(st):
        return out.float() * st["sd"] + st["mu"]
    return torch.cat([out[..., :6].float() * st["sd"] + st["mu"], torch.where(out[..., 6:] > 0, 1.0, -1.0)], -1)


def clip(y, st):  # 闭环 / 伪标签：OSC 动作截到 [-1, 1]；关节角（全连续）不截
    return y if cont(st) else y.clamp(-1, 1)


def load_actions(data, wa, h5root, n, zpad=False):
    """→ (N,n,7)：第 f 帧（demo 内第 k 帧）对应 a_{2k..2k+n-1}，越界处重复最后一个动作（只在不参与训练的帧上出现）。
    zpad（(mika) idm train --end_min）：越界处平移 / 旋转 6 维置 0（demo 结束后静止），夹爪维仍重复最后一个。"""
    root = Path(wa.data)
    tasks = json.loads((root / "meta/tasks.json").read_text())
    ix = np.load(root / "meta/index.npz")
    suites, out, fs = wa.suites.split(",") if wa.suites else None, [], {}
    for t, d, l in zip(ix["task_id"], ix["demo"], ix["demo_len"]):
        s = tasks[t]["suite"]
        if suites and s not in suites:
            continue
        p = Path(h5root) / s / f"{tasks[t]['task']}_demo.hdf5"
        if p not in fs:
            fs[p] = h5py.File(p, "r")
        a = fs[p][f"data/demo_{d}/actions"][()]
        assert l == math.ceil(len(a) / STRIDE), (str(p), d, l, len(a))
        k_ = STRIDE * np.arange(l)[:, None] + np.arange(n)
        a_ = a[np.minimum(k_, len(a) - 1)]
        if zpad:
            a_[k_ >= len(a), :6] = 0
        out.append(a_)
    A = np.concatenate(out).astype(np.float32)
    err = np.abs(A[:, 0] - np.load(root / "meta/actions.npy")[data.src]).max()
    assert err < 1e-5, f"hdf5 动作与 meta 不对齐（{err}）"
    return torch.from_numpy(A).to(data.device)


def frames(data, split, H, hist):  # 有 H 帧未来的帧；hist=True 时还要求有完整的 K 帧历史（与 WM 推演同一批 anchor）
    f = torch.nonzero(data.fsplit == split).squeeze(1)
    f = f[f + H * data.delta < data.fend[f]]
    return f[f - (data.K - 1) * data.delta >= data.fstart[f]] if hist else f


def labeled(data, n, seed=0):
    """低动作标注实验：每个任务只保留 n 条带动作的训练 demo（n≤0 为全部）→ 逐帧 bool。
    每个任务的训练 demo 用固定种子打乱后取前 n 条，所以 n=5 ⊂ 10 ⊂ 20；val 不受影响。"""
    if n <= 0:
        return torch.ones(len(data.f), dtype=torch.bool, device=data.device)
    g, keep = torch.Generator().manual_seed(seed), torch.zeros(len(data.length), dtype=torch.bool)
    for t in data.task_d.unique():
        d = torch.nonzero((data.task_d == t) & (data.split_d == 0)).squeeze(1)
        keep[d[torch.randperm(len(d), generator=g)[:n]]] = True
    return keep.repeat_interleave(data.length).to(data.device)


def excl_keep(data, wa, xs, k, seed=0):
    """(vidonly) --excl_keep：被 --excl_suites 去掉的 suite 里，每个任务仍留 k 条训练 demo 的动作 → 逐帧 bool（True = 留）。
    每个任务的干净训练 demo 按固定种子打乱后取前 k 条（k=1 ⊂ 5 ⊂ 10），同编号的 DR 复本（同一轨迹换外观）一起留；布局生成 demo（编号 ≥ 1000）不留。"""
    root = Path(data.root)
    ix, tasks = np.load(root / "meta/index.npz"), json.loads((root / "meta/tasks.json").read_text())
    sel = np.isin([tasks[t]["suite"] for t in ix["task_id"]], wa.suites.split(",")) if wa.suites else np.ones(len(ix["demo"]), bool)  # 同 Feats.load
    dm = ix["demo"][sel]
    aug = ix["aug"][sel] > 0 if "aug" in ix.files else np.zeros(len(dm), bool)
    td, sp, g, ok = data.task_d.cpu().numpy(), data.split_d.cpu().numpy(), np.random.default_rng(seed), np.zeros(len(dm), bool)
    for t in np.unique(td):
        if tasks[t]["suite"] in xs:
            c = np.flatnonzero((td == t) & (sp == 0) & ~aug)
            ok |= (td == t) & (dm < 1000) & np.isin(dm, dm[c[g.permutation(len(c))[:k]]])
    return torch.from_numpy(ok).to(data.device).repeat_interleave(data.length.to(data.device))


def zin(data, f, offs, F=None):  # 开头不足的历史帧重复第一帧（与闭环一致）；未来帧由 frames() 保证在 demo 内。F：另一组同帧序特征（腕部）
    ix = torch.maximum(f[:, None] + offs.to(f.device) * data.delta, data.fstart[f][:, None])
    ix = torch.minimum(ix, data.fend[f][:, None] - 1)  # 超出 demo 末尾的帧（只有 --far_extra 的远时距会）取最后一帧，同 train --far_clamp
    if F is not None:  # 腕部 / 本体：腕部特征可在 CPU 内存（PROBWM_FEATS_CPU=1），或按需从本地分片读（data.CatViews，PROBWM_FEATS_MMAP=1）
        if hasattr(F, "to_dev"):
            return F.to_dev(ix.cpu(), ix.device)
        return F[ix] if F.device == ix.device else F[ix.cpu()].to(ix.device, non_blocking=True)
    return data.get(ix) if hasattr(data, "get") else data.f[ix]  # data.get：特征可放 CPU（PROBWM_FEATS_CPU=1）


def woffs(a):  # 腕部相机帧相对 t 的偏移：t-n+1..t；--wfut k 再加真实未来 t+1..t+k（离线探针：未来腕部画面对动作的信息量）
    return torch.arange(1 - a.wrist, 1 + getattr(a, "wfut", 0))


def load_wrist(data, wa, norm=None, keys="wrist", sdev=None):
    """额外相机流：feats/<encoder>_<key>（与 agentview 同一全局帧序）；keys 逗号分隔时（如 right,wrist）各视角按 token 拼接，
    各自的逐通道 mu / sd（norm 为空时与 Feats 一样：固定种子抽 20000 个训练帧），多视角时 mu / sd 为 (T, D)。
    常驻 GPU；PROBWM_FEATS_CPU=1 放 CPU 内存；PROBWM_FEATS_MMAP=1 且都是本地分片库时按需读（data.CatViews，不占内存）。→ (W, wnorm)"""
    from .data import CatViews, MmapParts, load_parts
    dev, ks = data.device, keys.split(",")
    fds = [Path(wa.data) / f"feats/{wa.encoder}_{k}" for k in ks]
    infs = [json.loads((fd / "info.json").read_text()) if (fd / "info.json").exists() else {} for fd in fds]
    ident = len(data.src) == len(data.f) and (np.asarray(data.src) == np.arange(len(data.src))).all()
    if norm is None and all((fd / "wnorm.pt").exists() for fd in fds):  # (rteasy 11-16) 特征库自带的腕部 norm（如只在 clean 帧上算的统计量；DR 帧填 mu = 标准化后为 0 的"腕部缺省"）
        ws = [torch.load(fd / "wnorm.pt", map_location="cpu") for fd in fds]
        norm = {k_: ws[0][k_] for k_ in ("mu", "sd")} if len(ks) == 1 else {k_: torch.cat([w[k_].expand(int(w["T"]), -1) for w in ws]) for k_ in ("mu", "sd")}  # wnorm.pt = {mu (D,), sd (D,), T}
    tr = torch.nonzero(data.fsplit == 0).squeeze(1)
    pick = tr[torch.randperm(len(tr), device=dev, generator=torch.Generator(device=dev).manual_seed(0))[:20000]]
    stat = lambda x: (x.mean(0), x.std(0).clamp_min(1e-6))  # x: (n, T, D) → 逐通道
    if os.environ.get("PROBWM_FEATS_MMAP") and ident and all("parts" in i for i in infs):
        W = CatViews([MmapParts(fd, i, len(data.src)) for fd, i in zip(fds, infs)])
        if norm is None:
            x = W.raw(pick.cpu(), sdev or dev).float()  # sdev="cpu"（idm train --wstat_cpu）：统计量在 CPU 上算，省掉启动时约 30 GB 显存
            T1 = x.shape[1] // len(ks)
            ms = [stat(x[:, v * T1:(v + 1) * T1].flatten(0, 1)) for v in range(len(ks))]
            norm = {"mu": ms[0][0].cpu(), "sd": ms[0][1].cpu()} if len(ks) == 1 else \
                {"mu": torch.stack([m[0] for m in ms]).repeat_interleave(T1, 0).cpu(), "sd": torch.stack([m[1] for m in ms]).repeat_interleave(T1, 0).cpu()}
            del x
        W.mu, W.sd = norm["mu"].to(dev), norm["sd"].to(dev)
        return W, norm
    sdev = "cpu" if os.environ.get("PROBWM_FEATS_CPU") else dev
    Ws, mus, sds = [], [], []
    for v, (fd, inf) in enumerate(zip(fds, infs)):
        W = None
        if "parts" in inf:  # 分片 / 放在别的机器上的特征库：整份读进内存（全局帧序）
            f = load_parts(fd, inf, inf["parts"][-1][3])
            if sdev == "cpu" and ident:
                W = f  # data 帧序即全局帧序：直接用，省一份内存
            else:
                f = f.numpy()
        else:
            f = np.load(fd / "feats.npy", mmap_mode="r")
        if W is None:
            W = torch.empty((len(data.src), *f.shape[1:]), dtype=torch.float16, device=sdev)
            for a, l in zip(data.start.tolist(), data.length.tolist()):  # 逐 demo 拷贝（data.src 在 demo 内连续）
                W[a:a + l] = torch.from_numpy(np.array(f[data.src[a]:data.src[a] + l])).to(sdev)
        if norm is None:
            m = stat(W[pick.to(W.device)].to(dev).float().flatten(0, 1))
        else:  # 已有 norm：单视角为 (D,)，多视角为 (T, D)，取这个视角的那一段
            T1 = W.shape[1]
            m = (norm["mu"], norm["sd"]) if len(ks) == 1 else (norm["mu"][v * T1:(v + 1) * T1], norm["sd"][v * T1:(v + 1) * T1])
            m = (m[0].to(dev), m[1].to(dev))
        for i in range(0, len(W), 4096):
            W[i:i + 4096] = ((W[i:i + 4096].to(dev).float() - m[0]) / m[1]).to(W.dtype).to(W.device)
        Ws.append(W), mus.append(m[0].expand(W.shape[1], -1) if len(ks) > 1 and m[0].dim() == 1 else m[0]), sds.append(m[1].expand(W.shape[1], -1) if len(ks) > 1 and m[1].dim() == 1 else m[1])
    if norm is None:
        norm = {"mu": mus[0].cpu(), "sd": sds[0].cpu()} if len(ks) == 1 else {"mu": torch.cat(mus).cpu(), "sd": torch.cat(sds).cpu()}
    torch.cuda.empty_cache()
    return (Ws[0] if len(Ws) == 1 else torch.cat(Ws, 1)), norm


def load_prop(data, keys):  # 本体感觉 key（逗号分隔）→ (N,dp) 原始单位，data 帧序（data.state 已按 data.src 取好），常驻 GPU
    return torch.cat([data.state(k) for k in keys.split(",")], 1)


def ytgt(act, st):  # flow 头的目标：6 维标准化连续量 + 夹爪 ±1（全连续：逐维标准化）
    if cont(st):
        return (act - st["mu"]) / st["sd"]
    return torch.cat([(act[..., :6] - st["mu"]) / st["sd"], torch.where(act[..., 6:] > 0, 1.0, -1.0)], -1)


def loss(out, act, st, wg):
    if cont(st):
        return F.mse_loss(out.float(), (act - st["mu"]) / st["sd"])
    lc = F.mse_loss(out[..., :6].float(), (act[..., :6] - st["mu"]) / st["sd"])
    return lc + wg * F.binary_cross_entropy_with_logits(out[..., 6].float(), (act[..., 6] > 0).float())


@torch.no_grad()
def metrics(out, act, st):
    p = decode(out, st)
    if cont(st):  # 全连续：没有夹爪二值项（RoboTwin 的夹爪维是第 6、13 维的连续开度，含在 nmse_dim 里）
        e = (p - act) ** 2
        return {"mse": e.mean().item(), "nmse": (e / st["sd"] ** 2).mean().item(), "mse7": e.mean().item(), "grip": float("nan"),
                "nmse_dim": (e / st["sd"] ** 2).flatten(0, -2).mean(0).tolist()}
    e, g = (p[..., :6] - act[..., :6]) ** 2, (p[..., 6:] - torch.where(act[..., 6:] > 0, 1.0, -1.0)) ** 2
    return {"mse": e.mean().item(), "nmse": (e / st["sd"] ** 2).mean().item(), "mse7": torch.cat([e, g], -1).mean().item(),
            "grip": ((p[..., 6] > 0) == (act[..., 6] > 0)).float().mean().item(),
            "nmse_dim": (e / st["sd"] ** 2).flatten(0, -2).mean(0).tolist()}


def setup(wm_path, h5, H, dev, suites=None, root="", zpad=False, wpf=0):  # suites 非空：只载入这些 suite（norm 仍用 WM ckpt 里的）；root：换数据根（norm 仍用这个 WM 的）；zpad：见 load_actions
    ck = torch.load(wm_path, map_location="cpu")
    wa = Namespace(**ck["args"])
    wa.suites = wa.suites if suites is None else suites
    wa.data = root or wa.data
    if wpf:  # --wpf（降频档）：IDM 的一帧 = wpf 个路点。数据根不变（帧 k ↔ a_{STRIDE·k}），按 delta = wpf / STRIDE 隔帧取（同 train --delta）
        assert wpf % STRIDE == 0 and getattr(wa, "delta", 1) in (1, wpf // STRIDE), f"--wpf {wpf} 须是 {STRIDE} 的倍数，--wm 的 delta 须为 1 或 wpf / {STRIDE}"
        wa.delta = wpf // STRIDE
    data = load_data(wa, dev, ck["norm"])
    return ck, wa, data, load_actions(data, wa, h5, STRIDE * wa.delta * H, zpad)


def train(a):
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    em = getattr(a, "end_min", 0)  # (mika) --end_min h：训练 / val 帧只要求 h 帧未来在 demo 内（超出的计划帧取最后一帧、动作 6 维置 0）；0 = 原行为（要求 H 帧）
    assert (getattr(a, "wpf", 0) or 2) == 2 or not (getattr(a, "rtc_dmax", 0) or 0), "降频档（--wpf ≠ 2）不与 --rtc_dmax 同用（没有验证过）"
    ck, wa, data, A = setup(a.wm, a.h5, a.H, dev, a.suites or None, getattr(a, "data", ""), zpad=em > 0, wpf=getattr(a, "wpf", 0) or 0)  # --wpf：帧—路点比
    Hf = em if em > 0 else max([a.H] + fars(a))  # --far：要求 z_{t+far} 在 demo 内
    tr, va, lab = frames(data, 0, Hf, False), frames(data, 1, Hf, True), labeled(data, a.n_demo)
    if a.trainval:  # 每任务 50 条 demo 全部训练（val 只作监控）
        tr, lab = torch.cat([tr, frames(data, 1, Hf, False)]), lab | (data.fsplit == 1)
    fsu = lambda ss: torch.tensor([s in ss.split(",") for s in data.task_suite], device=dev)[data.task_d.to(dev).repeat_interleave(data.length.to(dev))]  # (vidonly) 逐帧：是否属于这些 suite
    if getattr(a, "excl_suites", ""):  # (vidonly) 目标 suite 只给视频：训练帧去掉这些 suite（动作 / 本体统计、--bal 抽样都只用剩下的 suite）；载入、norm、val 不变
        n0, dx = len(tr), fsu(a.excl_suites)
        if getattr(a, "excl_keep", 0) > 0:  # (vidonly) 每个任务仍留 k 条 demo 的动作（少量目标动作）
            dx = dx & ~excl_keep(data, wa, a.excl_suites.split(","), a.excl_keep)
        tr = tr[~dx[tr]]
        print(f"--excl_suites {a.excl_suites}（每任务留 {getattr(a, 'excl_keep', 0)} 条 demo）：训练帧 {n0} → {len(tr)}", flush=True)
    fvx = fsu(a.val_suites) if getattr(a, "val_suites", "") else None  # (vidonly) val 另报这些 suite
    if a.pseudo:  # VPT 式：标注 demo 用真实动作，其余训练 demo 用 IDM（只在标注 demo 上训练）预测的伪动作，全部训练帧都用上
        A = pseudo(a.pseudo, data, A, tr[~lab[tr]])
        print(f"伪标签 {int((~lab[tr]).sum())} 帧，真实标注 {int(lab[tr].sum())} 帧（IDM {a.pseudo}）", flush=True)
    else:
        tr = tr[lab[tr]]
    rfl, trs = None, tr  # (flywheel) --roll_frac：逐帧是否 rollout；动作 / 本体统计量只用原 demo 帧
    if getattr(a, "roll_frac", 0.0) > 0:
        import numpy as np  # train() 后面有局部 import numpy，这里也要局部绑定
        assert not wa.suites, "--roll_frac 只接不筛 suite 的数据根"
        rfl = torch.from_numpy(np.load(Path(wa.data) / "meta/index.npz")["roll"] > 0).to(dev).repeat_interleave(data.length.to(dev))
        trs = tr[~rfl[tr]]
    a.adim = A.shape[-1]  # 7（6 维 OSC + 夹爪 ±1）或全连续（RoboTwin 14 维 qpos：mu / sd 覆盖全部维，见 cont）
    nc = 6 if a.adim == 7 else a.adim
    st = {"mu": A[trs, :, :nc].reshape(-1, nc).mean(0), "sd": A[trs, :, :nc].reshape(-1, nc).std(0).clamp_min(1e-4)}
    n_act, offs = A.shape[1], offsets(a, wa.K)
    zi = lambda f: zin(data, f, offs)
    if getattr(a, "far_kf", False):  # 目标条件 IDM，目标 = 下一个接触关键帧（配 train --keyframe 的 WM）：最后一个输入帧换成那一帧
        assert fars(a) == [1], "--far_kf 用 --far 1 占位"
        kfi = next_contact(data, wa.data)
        zi = lambda f: torch.cat([zin(data, f, offs)[:, :-1], getattr(data, "get", data.f.__getitem__)(kfi[f])[:, None]], 1)
    if getattr(a, "kf_extra", False):  # J1：普通 chunk 计划之外再读下一个接触关键帧（目标帧），chunk 跟得慢时可以朝关键帧走
        kfi = next_contact(data, wa.data)
        zi = lambda f: torch.cat([zin(data, f, offs), getattr(data, "get", data.f.__getitem__)(kfi[f])[:, None]], 1)
    assert a.wrist <= (a.ih + 1 if a.kind == "idm" else wa.K), "腕部帧数不能超过 agentview 的历史 + 当前帧"
    W, wn = load_wrist(data, wa, keys=getattr(a, "wkeys", "wrist") or "wrist", **({"sdev": "cpu"} if getattr(a, "wstat_cpu", False) else {})) if a.wrist else (None, None)
    a.wT = W.shape[1] if a.wrist else 0  # 腕部 token 数（feats/<encoder>_wrist 可以是池化过的，如 RoboCasa 的 8×8）
    zw = (lambda f: zin(data, f, woffs(a), W)) if a.wrist else (lambda f: None)
    fu = fu_ema = None
    T0 = ck["dims"][1]
    fz = lambda z: z  # (pro FU) val / 评估：EMA 的 F 把每帧 [头部; 腕部] 融成 s（不开 --fuse 时原样）
    if getattr(a, "fuse", ""):  # (pro 11-18 FU) 腕部融合模块 F_φ 与 IDM 联合训（probwm/fuse.py）：IDM 的帧输入全换成 s = F(h, w)；历史 / 当前帧用在线 F，未来帧用 EMA 副本
        from . import fuse as fuse_
        assert a.kind == "idm" and a.wrist and a.head == "flow" and not (fars(a) or fextra(a) or getattr(a, "vpp", "") or getattr(a, "tau_in", "") or a.wm_frac > 0
                                                                     or getattr(a, "kf_extra", False) or getattr(a, "far_kf", False) or getattr(a, "save_every", 0)
                                                                     or getattr(a, "roll_frac", 0.0) > 0 or getattr(a, "val_suites", "")), \
            "--fuse 只接普通 flow 头 IDM + --wrist（不与 --far / --far_extra / --vpp / --tau_in / --wm_frac / 关键帧 / --save_every / --roll_frac / --val_suites 同用）"
        fd, fdep, fpool = (int(x) for x in a.fuse.split(","))
        fu = fuse_.Fuse(ck["dims"][0], T0, W.shape[1], fd, fdep, max(1, fd // 64), fpool).to(dev)
        fu_ema = copy.deepcopy(fu).requires_grad_(False)
        zi0 = zi
        zi = lambda f: torch.cat([zi0(f), zin(data, f, offs, W).to(data.f.dtype if hasattr(data, "f") else torch.float16)], 2)  # 每帧 [头部 T0; 腕部 Tw] token
        fz = lambda z: fuse_.frames(fu_ema, z, T0)
        print(f"--fuse {a.fuse}：F {sum(p.numel() for p in fu.parameters()) / 1e6:.2f}M 参数（d{fd}×{fdep}，腕部池化 {fpool}），λ_stat {a.fuse_stat}、λ_res {a.fuse_res}、EMA {a.fuse_ema}", flush=True)
    P = load_prop(data, a.proprio) if a.proprio else None
    a.pdim = P.shape[1] if a.proprio else 0
    zp = (lambda f: zin(data, f, poffs(a, wa.K), P)) if a.proprio else (lambda f: None)
    vh = vring = None
    if getattr(a, "vpp", ""):  # (vpp) 冻结的 --wm（MS-far 滚动 WM）在线前向出近槽隐状态，见 vpp_hidden
        assert a.kind == "idm" and a.head == "flow" and not (fars(a) or fextra(a) or a.wrist or a.lang or getattr(a, "tau_in", "") or a.pseudo or getattr(a, "prefetch", False)
                                                             or getattr(a, "p_fdrop", 0.0) > 0 or getattr(a, "kf_extra", False) or getattr(a, "far_kf", False)
                                                             or a.wm_frac > 0 or getattr(a, "mix", "") or getattr(a, "bal", -1.0) >= 0 or getattr(a, "roll_frac", 0.0) > 0), \
            "--vpp 只接普通 flow 头 IDM、均匀抽训练帧（不与 --far / --far_extra / --wrist / --lang / --tau_in / --pseudo / --prefetch / --p_fdrop / 关键帧 / --wm_frac / --mix / --bal / --roll_frac 同用）"
        vwm = build(wa, ck["dims"]).to(dev).eval().requires_grad_(False)
        vwm.load_state_dict(ck["model"])
        a.vpp_d, a.vpp_fp = wa.d, vpp_fp(vwm)
        vtr = torch.compile(vwm.trunk) if a.vpp_compile else None  # 编译后 WM 前向约快一倍（B20Z：6.5 → 3.2 ms / 样本）
        vg = lambda f, g=None: torch.cat([vpp_hidden(vwm, wa, data, f[j:j + a.vpp_chunk], a.H, g, vtr) for j in range(0, len(f), a.vpp_chunk)])
        vh, vnew = vg, a.vpp_new or a.bs
        vb, vptr = [None, None], [0]

        def vring():  # 环形缓冲：每步只给 vnew 个新帧算隐状态（换掉最旧的），batch = 缓冲里全部 bs 帧（--vpp_new < bs 时每个隐状态连用 bs / vnew 步）
            if vb[0] is None:
                vb[0] = tr[torch.randint(len(tr), (a.bs,), device=dev)]
                vb[1] = vg(vb[0])
            elif vnew >= a.bs:
                vb[0] = tr[torch.randint(len(tr), (a.bs,), device=dev)]
                vb[1] = vg(vb[0])
            else:
                j = (vptr[0] + torch.arange(vnew, device=dev)) % a.bs
                vptr[0] = (vptr[0] + vnew) % a.bs
                vb[0][j] = tr[torch.randint(len(tr), (vnew,), device=dev)]
                vb[1][j] = vg(vb[0][j])
            return vb[0], zi(vb[0]), vb[1]
        print(f"vpp {a.vpp}：WM {a.wm}（d{wa.d}，槽 {wa.tgt_offs}，指纹 {a.vpp_fp:.6g}），每步新算 {min(vnew, a.bs)} / {a.bs} 个隐状态", flush=True)
    net = new_net(a, ck["dims"], wa.K, n_act).to(dev)
    rtd = getattr(a, "rtc_dmax", 0) or 0  # 训练时 RTC：每个样本的前缀长度 d，以 --rtc_p0 的概率为 0，否则在 1..rtc_dmax 均匀（rtc_draw）
    if rtd:
        assert a.head == "flow" and a.kind == "idm" and not getattr(a, "fuse", ""), "--rtc_dmax 只接普通 flow 头 IDM（不与 --fuse 同用）"
        assert rtd < n_act, "--rtc_dmax 须小于动作块长度"
        rtc_draw = lambda B: torch.where(torch.rand(B, device=dev) < a.rtc_p0, 0, torch.randint(1, rtd + 1, (B,), device=dev))
        print(f"--rtc_dmax {rtd}：前缀长度 d = 0 的概率 {a.rtc_p0:g}，其余在 1..{rtd} 均匀；前缀位置层级 1、不算损失（动作块 {n_act} 个路点）", flush=True)
    if getattr(a, "init", ""):  # 从已有头继续（结构须相同；本体 mu / sd 下面照样按本数据重算）
        sd_ = torch.load(a.init, map_location="cpu")["model"]
        assert rtd or "pemb" not in sd_, f"--init {a.init} 是训练时 RTC 的 IDM（带 pemb），须同时给 --rtc_dmax"
        if rtd:  # 从普通 IDM 热启动时只缺 pemb（零初始化）；从 RTC 的 IDM 热启动时全部键对上
            r_ = net.load_state_dict(sd_, strict=False)
            assert set(r_.missing_keys) <= {"pemb"} and not r_.unexpected_keys, r_
            print("init from", a.init, "（--rtc_dmax：" + ("缺 pemb，零初始化）" if r_.missing_keys else "全部键对上）"), flush=True)
        else:
            print("init from", a.init, net.load_state_dict(sd_), flush=True)
    assert not getattr(a, "kf_delta", False) or getattr(a, "far_kf", False), "--kf_delta 要配 --far_kf"
    dkf = (lambda f: kfi[f] - f) if getattr(a, "kf_delta", False) else (lambda f: None)  # 到关键帧还有几帧（训练用真实值）
    if a.proprio:  # 逐维标准化统计量（训练帧，开头不足的历史同样重复第一帧）
        x = pfeat(zp(trs))
        net.pmu.copy_(x.mean(0)), net.psd.copy_(x.std(0).clamp_min(1e-4))
        print(f"proprio {a.proprio}：每帧 {a.pdim} 维 × {len(poffs(a, wa.K))} 帧，sd {net.psd.cpu().numpy().round(4)}", flush=True)
    fl = a.head == "flow"
    pars = [*net.parameters(), *(fu.parameters() if fu is not None else [])]
    opt = torch.optim.AdamW(pars, lr=a.lr, weight_decay=a.wd, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(i / a.steps, 1.0))))
    amp = lambda: torch.autocast("cuda", torch.bfloat16)
    txt = (lambda f: data.text(data.ftask[f])) if a.kind == "bc" or a.lang else (lambda f: ())
    fl_last = {}  # (pro FU) 最近一步的门 / 残差统计（log 用）

    def txt_tr(f):  # 训练 batch：可选随机改写 + 文本噪声（同 train.py tx()）
        if a.kind != "bc" and not a.lang:
            return ()
        t, m = data.text(data.ftask[f], rand=a.txt_aug)
        if a.txt_noise > 0:
            t = t.float()
            t = t + a.txt_noise * t.pow(2).mean(-1, keepdim=True).sqrt() * torch.randn_like(t) * m[..., None]
        return t, m
    assert not a.txt_aug or data.temb.dim() == 4, f"--txt_aug 需要 --wm 的文本是改写增广表（{wa.text}）"
    nw, co = int(a.bs * a.wm_frac) if a.kind == "idm" else 0, torch.arange(-(wa.K - 1), 1)
    wcm = None
    if nw and getattr(a, "wm_cache", ""):  # --wm_cache：WM 计划预先采好存在盘上（scripts/rc_wmcache.py：<前缀>_idx.npy 帧下标、<前缀>_plan.npy (N,H,T,D)）
        import numpy as np  # train() 后面有局部 import numpy，这里也要局部绑定
        trh = torch.from_numpy(np.load(a.wm_cache + "_idx.npy")).to(dev)
        wcm = np.load(a.wm_cache + "_plan.npy", mmap_mode="r")
        assert wcm.shape[1:] == (a.H, *data.f.shape[1:]), (wcm.shape, a.H, data.f.shape)
        print(f"wm_cache {a.wm_cache}：{len(trh)} 个 anchor 的 WM 计划，每 batch {nw} 个", flush=True)
    elif nw:  # 一部分样本的未来帧换成 WM 采样：让 IDM 适应 WM 输出（运动幅度偏小、细节模糊）
        wm, trh = build(wa, ck["dims"]).to(dev).eval().requires_grad_(False), frames(data, 0, a.H, True)
        trh = trh[lab[trh]]
        wm.load_state_dict(ck["model"])
        wf = lambda g: fm.rollout(wm, zin(data, g, co), *data.text(data.ftask[g]), a.H, wa.mode, alpha=wa.alpha,
                                  sigma=wa.sigma, rs=data.rs, cfg=a.cfg, **SAMPLERS[a.wm_sampler])[0]
        if a.wm_S:  # 预先给每个 anchor 采 wm_S 个 WM 未来（与每步在线采样同分布，但省掉每步的采样开销）
            wc = torch.empty(a.wm_S, len(trh), a.H, *data.f.shape[1:], dtype=data.f.dtype, device=dev)
            with torch.no_grad(), amp():
                for s_ in range(a.wm_S):
                    for j in range(0, len(trh), 64):
                        wc[s_, j:j + 64] = wf(trh[j:j + 64]).to(wc.dtype)
            torch.cuda.empty_cache()
    print(f"{a.kind}: params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M  train {len(tr)}  val {len(va)}  "
          f"n_act {n_act}", flush=True)
    pacc = (getattr(a, "accum", 1) or 1) > 1  # --accum k：全局 --bs 不变，分 k 个微批前向 / 反传（只为显存，等价整批的平均损失）；默认 1 = 原路径
    assert not pacc or (a.kind == "idm" and fl and fu is None and not nw and vring is None and not a.lang and not getattr(a, "tau_in", "")
                        and not getattr(a, "aux_dense", 0.0) > 0 and not getattr(a, "kf_delta", False) and a.bs % a.accum == 0), \
        "--accum 只接普通 flow 头 IDM（可配 --rtc_dmax；不与 --fuse / --wm_frac / --vpp / --lang / --tau_in / --aux_dense / --kf_delta 同用），--bs 须被 --accum 整除"
    trg = None
    if getattr(a, "mix", ""):  # 固定比例抽训练帧：干净 / 其余增广（DR 复本）/ 生成 demo（编号 ≥ 1000），同 train --p_main --p_lg
        import numpy as np
        root = Path(wa.data)
        ix, tasks = np.load(root / "meta/index.npz"), json.loads((root / "meta/tasks.json").read_text())
        keep = np.isin([tasks[t]["suite"] for t in ix["task_id"]], wa.suites.split(",")) if wa.suites else np.ones(len(ix["demo"]), bool)
        aug, lg = ix["aug"][keep] > 0, ix["demo"][keep] >= 1000
        gf = torch.from_numpy(np.where(~aug, 0, np.where(lg, 2, 1))).repeat_interleave(data.length).to(dev)
        trg = [tr[gf[tr] == k] for k in range(3)]
        mix = [float(x) for x in a.mix.split(",")]
        nmix = [int(round(a.bs * x)) for x in mix[:2]]
        nmix.append(a.bs - sum(nmix))
        print(f"IDM 固定比例 {mix}：各组训练帧 {[len(x) for x in trg]}", flush=True)
    bal = None
    if getattr(a, "bal", -1.0) >= 0:  # 四 suite 联合训练（fullbench）：suite / 任务均衡抽训练帧
        from .data import bal_sampler
        assert trg is None and not nw, "--bal 不与 --mix / --wm_frac 同用"
        bal = bal_sampler(data, tr, a.bal, wa.suites.split(",") if wa.suites else None, p_x=getattr(a, "bal_x", 0.0))  # (cam) --bal_x：index 的 xtask 列 > 0 的 demo 成第 4 组
    vr = None
    if rfl is not None:  # (flywheel) 固定比例：原 demo 帧 / rollout 帧；另报留出 rollout（split 2）的 val（固定抽 16384 帧）
        assert trg is None and bal is None and not nw, "--roll_frac 不与 --mix / --bal / --wm_frac 同用"
        trg = [tr[~rfl[tr]], tr[rfl[tr]]]
        nmix = [a.bs - int(round(a.bs * a.roll_frac)), int(round(a.bs * a.roll_frac))]
        vr = frames(data, 2, Hf, True)
        vr = vr[torch.randperm(len(vr), device=dev, generator=torch.Generator(device=dev).manual_seed(0))[:16384]]
        print(f"--roll_frac {a.roll_frac}：每 batch 原 demo / rollout {nmix}，训练帧 {[len(x) for x in trg]}，留出 rollout val {len(vr)} 帧", flush=True)

    twc = None
    if getattr(a, "task_w", ""):  # (rteasy) 按任务难度重加权：训练帧按"逐帧均匀 × 所属任务权重"抽（权重表见 scripts/rteasy_tw.py；缺的任务 1）
        assert trg is None and bal is None and not nw and wcm is None, "--task_w 不与 --mix / --bal / --roll_frac / --wm_frac / --wm_cache 同用"
        twd = json.loads(Path(a.task_w).read_text())
        twd = twd.get("w", twd)
        tw = torch.tensor([float(twd.get(n, 1.0)) for n in data.task_names], device=dev)
        ft = data.task_d.to(dev).repeat_interleave(data.length.to(dev))[tr]
        twc = tw[ft].double().cumsum(0)  # float64：几百万帧的累积和在 float32 下会丢增量
        s0, s1 = torch.bincount(ft, minlength=len(tw)).float(), torch.bincount(ft, weights=tw[ft], minlength=len(tw))
        s0, s1 = s0 / s0.sum(), s1 / s1.sum()
        print(f"--task_w {a.task_w}：任务权重 {tw.min():.2f}–{tw.max():.2f}；帧份额变化最大的任务 "
              + "，".join(f"{data.task_names[i]} {100 * s0[i]:.1f}→{100 * s1[i]:.1f}%" for i in (s1 / s0.clamp_min(1e-9)).argsort(descending=True)[:5].tolist()), flush=True)

    def draw():  # 训练帧（+ 不需要改帧时直接取好特征）
        f = bal(a.bs) if bal is not None else \
            (tr[torch.searchsorted(twc, torch.rand(a.bs, device=dev, dtype=torch.float64) * twc[-1]).clamp_max(len(tr) - 1)] if twc is not None else  # (rteasy) --task_w
             tr[torch.randint(len(tr), (a.bs,), device=dev)]) if trg is None else \
            torch.cat([g[torch.randint(len(g), (k,), device=dev)] for g, k in zip(trg, nmix)])
        if wcm is not None:  # --wm_cache：前 nw 个换成缓存了计划的 anchor，未来帧用缓存的 WM 计划
            k = torch.randint(len(trh), (nw,), device=dev)
            f[:nw] = trh[k]
            z = zi(f)
            z[:nw, a.ih + 1:a.ih + 1 + a.H] = torch.from_numpy(np.stack([wcm[j] for j in k.cpu().numpy()])).to(dev, z.dtype)
            return f, z
        return f, None if nw else zi(f)
    nb = draw
    if getattr(a, "prefetch", False):  # 后台线程先取好下一个 batch（同一 CUDA 流；读盘释放 GIL）
        from concurrent.futures import ThreadPoolExecutor
        assert not nw or wcm is not None, "--prefetch 不与在线采样的 --wm_frac 同用（--wm_cache 可以）"
        pf = ThreadPoolExecutor(1)
        fut = [pf.submit(draw)]

        def nb():
            b = fut[0].result()
            fut[0] = pf.submit(draw)
            return b
    t0, i0, rp = time.time(), 0, Path(str(a.out) + ".resume")
    if getattr(a, "save_every", 0) and rp.exists():  # --save_every：被杀后同命令重跑，从最近的续训点接着训
        r = torch.load(rp, map_location="cpu")
        net.load_state_dict(r["model"]), opt.load_state_dict(r["opt"]), sched.load_state_dict(r["sched"])
        i0 = r["step"] + 1
        print(f"续训：从第 {r['step']} 步的 {rp} 接着训", flush=True)
    for i in range(i0, a.steps + 1):
        zh = None
        if vring is None:
            f, z = nb()
        else:  # (vpp)
            f, z, zh = vring()
        if nw and wcm is None:
            k = torch.randint(len(trh), (nw,), device=dev)
            f[:nw] = trh[k]
            z = zi(f)
        if getattr(a, "kf_noise", 0.0) > 0:  # J1：关键帧输入加噪 τ ~ U(0, kf_noise)（训练用真实关键帧、闭环用预测的，补这个分布差）
            tk = (torch.rand(len(z), device=dev) * a.kf_noise)[:, None, None]
            z[:, -1] = ((1 - tk) * z[:, -1].float() + tk * torch.randn(z[:, -1].shape, device=dev)).to(z.dtype)
        if nw and wcm is None:
            with torch.no_grad(), amp():
                z[:nw, a.ih + 1:] = (wc[torch.randint(a.wm_S, (nw,), device=dev), k] if a.wm_S else wf(f[:nw])).to(z.dtype)
        if getattr(a, "p_fdrop", 0.0) > 0:  # 计划 dropout：整段未来换成当前帧的复制（= 没有计划；配 --lang 时退化为同输入的 BC）
            z = torch.where((torch.rand(len(z), device=dev) < a.p_fdrop)[:, None, None, None] & (torch.arange(z.shape[1], device=dev) > a.ih)[None, :, None, None],
                            z[:, a.ih:a.ih + 1], z)
        tau = None
        if getattr(a, "tau_in", ""):  # (xden) 噪声感知 IDM：计划帧加随机层级的流匹配噪声，各槽 τ 作为条件
            tau = tau_draw(len(z), a.H, a.tau_in, a.tau_alpha, a.tau_n, dev)
            z = tau_noise(z, tau, a.ih)
        lf = 0.0
        if fu is not None:  # (pro FU) 前 ih+1 帧（历史 + 当前）在线 F（带梯度），其后 H 帧未来用 EMA 的 F（不回传，防止 F 与 IDM 合谋走捷径）
            nh = a.ih + 1
            zh, zx = z[:, :, :T0], z[:, :, T0:]
            s_h, r = fu(zh[:, :nh].flatten(0, 1), zx[:, :nh].flatten(0, 1), delta=True)
            with torch.no_grad():
                s_f = fuse_.frames(fu_ema, z[:, nh:], T0)
            z = torch.cat([s_h.view(len(z), nh, T0, -1), s_f.to(s_h.dtype)], 1)
            hf = zh[:, :nh].flatten(0, 2).float()
            sf_ = hf + r.flatten(0, 1)
            lstat = (sf_.mean(0) - hf.mean(0)).pow(2).mean() + (sf_.std(0) - hf.std(0)).pow(2).mean()  # 逐通道均值 / 方差对齐 h（分布正则）
            lres = r.pow(2).mean()  # ‖s − h‖² 小权重
            lf = a.fuse_stat * lstat + a.fuse_res * lres
            fl_last.update(stat=float(lstat), rr=float(lres.sqrt() / hf.pow(2).mean().sqrt()))
        w = zw(f)
        if w is not None and a.p_wdrop > 0:  # 腕部 dropout：整段腕部输入置零（标准化空间的均值），逼 IDM 以 WM 的未来为主
            w = torch.where(torch.rand(len(w), 1, 1, 1, device=dev) < a.p_wdrop, torch.zeros_like(w), w)
        ya = None
        if getattr(a, "aux_dense", 0.0) > 0:  # 辅助：真实未来 8 帧（demo 内截断）的低维投影
            ix8 = torch.minimum(f[:, None] + torch.arange(1, 9, device=dev) * data.delta, data.fend[f][:, None] - 1)
            ya = getattr(data, "get", data.f.__getitem__)(ix8).float() @ net.aproj
        if pacc:  # --accum：整批的输入噪声 / RTC 前缀长度按原顺序一次抽好，各微批前向、反传（损失 / accum）
            mb_ = a.bs // a.accum
            zn_, wn_, p_, y_ = z + a.noise * torch.randn_like(z), None if w is None else w + a.noise * torch.randn_like(w), zp(f), ytgt(A[f], st)
            pd_ = rtc_draw(len(f)) if rtd else None
            net.rtc_den = (n_act - pd_).sum().clamp_min(1).float() if rtd else None  # RTC 损失按整批的非前缀位置数归一（各微批不再 / accum）
            opt.zero_grad(set_to_none=True)
            l = 0.0
            try:
                for j in range(0, a.bs, mb_):
                    with amp():
                        l_ = net(zn_[j:j + mb_], zw=None if wn_ is None else wn_[j:j + mb_], zp=None if p_ is None else p_[j:j + mb_], y=y_[j:j + mb_],
                                 **({"pd": pd_[j:j + mb_]} if rtd else {})) / (1 if rtd else a.accum)
                    l_.backward()
                    l = l + l_.detach()
            finally:
                net.rtc_den = None
        else:
            with amp():
                o = net(z + a.noise * torch.randn_like(z), *txt_tr(f), zw=w if w is None else w + a.noise * torch.randn_like(w), zp=zp(f),
                        y=ytgt(A[f], st) if fl else None, dk=dkf(f), yaux=ya, **({"tau": tau} if tau is not None else {}),
                        **({"zh": zh + a.noise * torch.randn_like(zh)} if zh is not None else {}), **({"pd": rtc_draw(len(f))} if rtd else {}))
                l = o if fl else loss(o, A[f], st, a.wg)
            opt.zero_grad(set_to_none=True)
            (l + lf).backward()
        torch.nn.utils.clip_grad_norm_(pars, 1.0)
        opt.step(), sched.step()
        if fu is not None:
            with torch.no_grad():
                for pe, p_ in zip(fu_ema.parameters(), fu.parameters()):
                    pe.lerp_(p_, 1 - a.fuse_ema)
        if getattr(a, "save_every", 0) and i > 0 and i % a.save_every == 0:
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": i}, rp)
        if i % a.log == 0 or i == a.steps:
            net.eval()
            if getattr(a, "tau_in", "") and i == i0 and len(va) > 16384:  # (xden) --tau_in 每次要多算一遍 val_roll：两遍 val 都只用固定的 16384 帧（特征在 CPU 时整个 val 一遍要 2 分钟）
                va = va[torch.randperm(len(va), device=dev, generator=torch.Generator(device=dev).manual_seed(0))[:16384]]
            if vh is not None and i == i0 and len(va) > a.vpp_val:  # (vpp) val 每帧要一次 WM 前向：只用固定的 --vpp_val 帧
                va = va[torch.randperm(len(va), device=dev, generator=torch.Generator(device=dev).manual_seed(0))[:a.vpp_val]]
            vk = (lambda g: {}) if vh is None else (lambda g, _g=torch.Generator(device=dev).manual_seed(0): {"zh": vh(g, _g)})  # (vpp) val 的隐状态：固定噪声（各次 log 可比）
            vc = getattr(a, "val_chunk", 0) or 1024  # --val_chunk：val 分块（只省显存）
            with torch.no_grad(), amp():
                out = torch.cat([net(fz(zi(va[j:j + vc])), *txt(va[j:j + vc]), zw=zw(va[j:j + vc]),
                                     zp=zp(va[j:j + vc]), dk=dkf(va[j:j + vc]), **vk(va[j:j + vc])) for j in range(0, len(va), vc)])
            m, m2 = metrics(out, A[va], st), metrics(out[:, :2], A[va, :2], st)
            xr = ""
            if rtd:  # val：第 2–3 个路点（延迟 2 个路点时实际执行的那两个），不给前缀 / 给真值前缀 d = 2
                with torch.no_grad(), amp():
                    op = []
                    for g_ in va.split(vc):
                        net.rtc_pre = (ytgt(A[g_], st)[:, :2], torch.full((len(g_),), 2, device=dev))
                        op.append(net(fz(zi(g_)), *txt(g_), zw=zw(g_), zp=zp(g_), dk=dkf(g_)))
                    net.rtc_pre = None
                    op = torch.cat(op)
                xr += f"  val[2:4] 无前缀 {metrics(out[:, 2:4], A[va, 2:4], st)['nmse']:.4f} / 真值前缀 d=2 {metrics(op[:, 2:4], A[va, 2:4], st)['nmse']:.4f}"
            if vh is not None:  # (vpp) 隐状态全 0（= 去掉隐状态，只剩历史帧 [+ 计划] + 本体）的对照：看 IDM 用了多少 WM 隐状态的信息
                with torch.no_grad(), amp():
                    o0 = torch.cat([net(zi(g), zp=zp(g), zh=torch.zeros(len(g), a.H, ck["dims"][1], a.vpp_d, device=dev, dtype=torch.bfloat16)) for g in va.split(1024)])
                xr = f"  val_noh nmse {metrics(o0, A[va], st)['nmse']:.4f}（前 2 个 {metrics(o0[:, :2], A[va, :2], st)['nmse']:.4f}）"
            if getattr(a, "tau_in", ""):  # (xden) 闭环 --idm_tau pre 的层级剖面（MS-far 窗口前 H 槽的起始层级，以真实未来为底）下的 val，前 2 个动作
                tr_ = fm.shift(1 - (torch.arange(a.H, device=dev) + 1) / a.tau_n, a.tau_alpha)[None]
                with torch.no_grad(), amp():
                    outr = torch.cat([net(tau_noise(zi(g), tr_.expand(len(g), -1), a.ih), *txt(g), zw=zw(g), zp=zp(g), dk=dkf(g), tau=tr_.expand(len(g), -1))
                                      for g in va.split(1024)])
                xr = f"  val_roll 前 2 个 {metrics(outr[:, :2], A[va, :2], st)['nmse']:.4f}"
            if fvx is not None:  # (vidonly)
                k = fvx[va]
                mx, mx2 = metrics(out[k], A[va][k], st), metrics(out[k, :2], A[va][k, :2], st)
                xr += f"  val[{a.val_suites}] nmse {mx['nmse']:.4f}（前 2 个 {mx2['nmse']:.4f}）grip {mx['grip']:.3f}"
            if vr is not None and len(vr):  # (flywheel)
                with torch.no_grad(), amp():
                    o2 = torch.cat([net(zi(g), *txt(g), zw=zw(g), zp=zp(g), dk=dkf(g)) for g in vr.split(1024)])
                xr += f"  val[rollout 留出] nmse {metrics(o2, A[vr], st)['nmse']:.4f}（前 2 个 {metrics(o2[:, :2], A[vr, :2], st)['nmse']:.4f}）"
            net.train()
            if fu is not None:
                gm, gx = fu_ema.stats()
                xr += f"  fuse g {gm:.4f} / max {gx:.4f}  ‖s−h‖/‖h‖ {fl_last.get('rr', 0):.4f}  stat {fl_last.get('stat', 0):.5f}"
            print(f"{i:>6} loss {l.item():.4f}  val nmse {m['nmse']:.4f}（前 2 个 {m2['nmse']:.4f}）mse {m['mse']:.5f} grip {m['grip']:.3f}{xr}  "
                  f"{(time.time() - t0) / max(i - i0, 1) * 1e3:.0f}ms/it", flush=True)
    if rp.exists():  # 训完了，续训点不再需要
        rp.unlink()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "args": vars(a), "dims": ck["dims"], "K": wa.K, "n_act": n_act,
                "astats": {k: v.cpu() for k, v in st.items()}, "wm": a.wm, "wnorm": wn, **({"fuse": fuse_.save(fu_ema, wn)} if fu is not None else {})}, a.out)


@torch.no_grad()
def evaluate(a):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    ick = torch.load(a.idm, map_location="cpu")
    idm, ia, st = make(ick, dev)
    idm.steps = a.fsteps or idm.steps
    Hv = max([ia.H, a.Hv] + fars(ia))  # anchor 集要求有 Hv 帧未来：不同 H 的 IDM 用 --Hv 8 在同一批 anchor 上比较
    ck, wa, data, A = setup(a.wm or ick["wm"], a.h5, Hv, dev, wpf=getattr(ia, "wpf", 0) or 0)  # 按 IDM 训练时的帧—路点比取帧
    A = A[:, :ick["n_act"]]
    wm = build(wa, ck["dims"]).to(dev).eval()
    wm.load_state_dict(ck["model"])
    bck = torch.load(a.bc, map_location="cpu") if a.bc else {}
    wn = ick.get("wnorm") or bck.get("wnorm")  # 带腕部相机的 IDM / BC：两者的腕部 norm 算法相同（同一批训练帧、同一种子）
    W = load_wrist(data, wa, wn, keys=getattr(ia, "wkeys", "wrist") or "wrist")[0] if wn else None
    zw = lambda f, n: zin(data, f, woffs(ia) if n == ia.wrist else torch.arange(1 - n, 1), W) if n else None
    P = load_prop(data, ia.proprio) if ia.proprio else None
    zp = lambda f: zin(data, f, poffs(ia, wa.K), P) if ia.proprio else None
    va, H, amp = frames(data, 1, Hv, True), len(fars(ia)) or ia.H, lambda: torch.autocast("cuda", torch.bfloat16)  # H = 未来输入帧数
    if a.n:
        va = va[torch.randperm(len(va), device=dev)[:a.n]]
    print(f"val anchors {len(va)}", flush=True)
    io, ctx_o = offsets(ia, wa.K), torch.arange(-(wa.K - 1), 1)
    past, gt = lambda f: zin(data, f, io[:ia.ih + 1]), lambda f: zin(data, f, io)
    cases = {"idm_gt": gt, "idm_copy": lambda f: torch.cat([past(f), data.f[f][:, None].expand(-1, H, -1, -1)], 1)}
    for name in a.samplers.split(","):
        def wmf(f, kw=SAMPLERS[name]):
            p, _ = fm.rollout(wm, zin(data, f, ctx_o), *data.text(data.ftask[f]), H, wa.mode, alpha=wa.alpha,
                              sigma=wa.sigma, rs=data.rs, cfg=a.cfg, **kw)
            return torch.cat([past(f), p.to(data.f.dtype)], 1)
        cases[f"idm_wm_{name}"] = wmf
    res, ns = {}, a.n_s if idm.flow else 1
    for name, fn in cases.items():
        outs, zerr = [], []
        for j in range(0, len(va), a.chunk):
            f = va[j:j + a.chunk]
            torch.manual_seed(j)  # WM 未来与 IDM 无关：不同 IDM（含 H=1 / 8、flow 采样）拿到同一批 WM 样本
            with amp():
                z = fn(f)
                tx = data.text(data.ftask[f]) if ia.lang else ()
                outs.append(torch.stack([idm(z, *tx, zw=zw(f, ia.wrist), zp=zp(f)).float() for _ in range(ns)]))
            zerr.append(((z[:, ia.ih + 1:].float() - gt(f)[:, ia.ih + 1:].float()) ** 2).mean((1, 2, 3)))
        o = torch.cat(outs, 1)
        for k, x in ([(name, o[0])] + ([(f"{name}_mean{ns}", o.mean(0))] if ns > 1 else [])):  # flow：单个样本 / ns 个样本的平均
            res[k] = {**metrics(x, A[va], st), "nmse2": metrics(x[:, :2], A[va, :2], st)["nmse"],
                      "grip2": metrics(x[:, :2], A[va, :2], st)["grip"], "z_nmse": torch.cat(zerr).mean().item()}
    if a.bc:
        bc, ba, bst = make(bck, dev)
        assert ba.H == H and not ba.proprio, "IDM 与 BC 的动作块长度不一致（BC 暂不支持 proprio 离线评测）"
        with amp():
            out = torch.cat([bc(zin(data, va[j:j + a.chunk], ctx_o), *data.text(data.ftask[va[j:j + a.chunk]]),
                                zw=zw(va[j:j + a.chunk], ba.wrist)).float() for j in range(0, len(va), a.chunk)])
        res["bc"] = metrics(out, A[va], bst)
    print(f"{'':>24} {'nmse':>7} {'mse':>8} {'mse7':>7} {'grip':>6} {'nmse2':>7} {'grip2':>6} {'z_nmse':>7}   （2 = 前 2 个动作，即闭环执行的部分）")
    for k, r in res.items():
        print(f"{k:>24} {r['nmse']:7.4f} {r['mse']:8.5f} {r['mse7']:7.4f} {r['grip']:6.3f} {r.get('nmse2', float('nan')):7.4f} "
              f"{r.get('grip2', float('nan')):6.3f} {r.get('z_nmse', float('nan')):7.4f}")
    out = Path(a.out or Path(a.idm).parent / "offline.json")
    out.write_text(json.dumps({"args": vars(a), "results": res}, indent=1))


def main():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a("cmd", choices=["train", "eval"])
    a("--wm", default="", help="世界模型 ckpt：复用它的数据设置和 norm（train 默认 abs_a13；eval 默认取 idm ckpt 里记的）")
    a("--h5", default="libero_data", help="<h5>/<suite>/<task>_demo.hdf5")
    a("--kind", default="idm", choices=["idm", "bc"]); a("--H", type=int, default=1, help="未来帧数；动作块 = 2H 个")
    a("--ih", type=int, default=3, help="IDM 用的历史帧数（不含 z_t），≤ K-1"); a("--p_drop", type=float, default=0.0)
    a("--wm_frac", type=float, default=0.0, help="IDM 训练时未来帧换成 WM 采样的比例"); a("--wm_sampler", default="scratch4")
    a("--wm_S", type=int, default=0, help=">0：wm_frac 的 WM 未来改为预先给每个 anchor 采这么多个（训练时 WM 用 --cfg）")
    a("--d", type=int, default=256); a("--depth", type=int, default=2)
    a("--steps", type=int, default=2000); a("--bs", type=int, default=256); a("--lr", type=float, default=5e-4)
    a("--wd", type=float, default=0.05); a("--wg", type=float, default=1.0, help="夹爪 BCE 的权重")
    a("--noise", type=float, default=0.1, help="输入特征加高斯噪声（标准化空间）"); a("--log", type=int, default=500)
    a("--seed", type=int, default=0); a("--out", default="")
    a("--wkeys", default="wrist", help="--wrist 的额外相机流：feats/<encoder>_<key>，逗号分隔时各视角按 token 拼接（如 right,wrist = RoboCasa 的右视角 + 腕部）")
    a("--wrist", type=int, default=0, help=">0：另外输入腕部相机 t-n+1..t 的 n 帧真实画面（≤ ih+1；BC ≤ K）"); a("--p_wdrop", type=float, default=0.0, help="训练时整段丢弃腕部输入的概率")
    a("--proprio", default="", help="本体感觉 key，逗号分隔，如 robot0_eef_pos,robot0_gripper_qpos（data/meta/states）")
    a("--p_pdrop", type=float, default=0.0, help="训练时整段丢弃本体感觉输入的概率")
    a("--lang", action="store_true", help="IDM 也输入指令（语言 token，同 BC）")
    a("--txt_aug", action="store_true", help="BC / --lang：训练时随机取改写增广表里的一条（--wm 的文本须是增广表）")
    a("--txt_noise", type=float, default=0.0, help="BC / --lang：训练时文本嵌入加的高斯噪声（相对每个 token 的 RMS）")
    a("--ph", type=int, default=0, help=">0：本体感觉只取最近这么多帧（1 = 只用当前值，不给速度）；0 = 与图像历史相同")
    a("--rope", action="store_true", help="2D RoPE 代替 patch 的绝对位置嵌入（同 train --rope）")
    a("--head", default="det", choices=["det", "flow"], help="flow：flow matching 动作专家，生成整个动作块")
    a("--fsteps", type=int, default=0, help="flow 头推理的 Euler 步数（train 时 0 → 10；eval 时 0 = 用 ckpt 里的）")
    a("--mix", default="", help="训练帧固定比例：干净,DR,生成（如 0.5,0.25,0.25；生成 = demo 编号 ≥ 1000）")
    a("--kf_noise", type=float, default=0.0, help="--kf_extra：训练时关键帧输入加噪的最大层级")
    a("--kf_delta", action="store_true", help="--far_kf：另输入'到关键帧还有几帧'（训练用真实值；闭环要外部给，plandiag 用专家的 oracle 值）")
    a("--aux_dense", type=float, default=0.0, help=">0：辅助损失权重——IDM 的 patch token 同时预测真实未来 8 帧特征的低维投影（先想近处再出动作）")
    a("--aux_pd", type=int, default=64, help="--aux_dense 的投影维数")
    a("--kf_extra", action="store_true", help="IDM 在 chunk 计划之后再多读一帧：下一个接触关键帧（两级 WM 用；闭环时为关键帧 WM 的预测）")
    a("--data", default="", help="IDM 训练换数据根（norm、编码器、语言表仍按 --wm 的 WM；如 300-MG 的 kf WM 配 50 条人类 demo 训的 IDM）")
    a("--far_kf", action="store_true", help="目标条件 IDM 的目标 = 下一个接触关键帧（可变时距，配 train --keyframe 的 WM；需 --far 1 占位）")
    a("--far", default="0", help="目标条件 IDM：未来输入只有 z_{t+k}（如 8，或关键帧 4,8），不是 z_{t+1..t+H}；输出 t 起的 2H 个动作；配 train --tgt_off 的远点 / 关键帧 WM")
    a("--Hv", type=int, default=0, help="eval：val anchor 要求至少有这么多帧未来（不同 H 的 IDM 在同一批 anchor 上比较）")
    a("--n_s", type=int, default=8, help="eval：flow 头另外报告这么多个样本的平均")
    a("--suites", default="", help="train：只用这些 suite（逗号分隔；默认 = --wm 的 suite；norm 仍用 WM 的）")
    a("--trainval", action="store_true", help="train：val demo 也参与训练（标准 LIBERO：每任务 50 条全用）")
    a("--n_demo", type=int, default=0, help=">0：每个任务只用这么多条带动作的训练 demo（嵌套子集）")
    a("--pseudo", default="", help="IDM ckpt：其余训练 demo 的动作换成它预测的伪标签（VPT 式半监督基线）")
    a("--idm", default=""); a("--bc", default=""); a("--samplers", default="scratch4,warm")
    a("--cfg", type=float, default=1.0); a("--n", type=int, default=0, help=">0：只评测这么多个 val anchor")
    a("--chunk", type=int, default=512)
    a("--bal_x", type=float, default=0.0, help="(cam) 配 --bal：index 的 xtask 列 > 0 的 demo（如宽视角复本）单独成第 4 组，在有这组帧的任务里占这个比例（data.bal_sampler 的 p_x）；默认 0 = 原行为")
    a("--bal", type=float, default=-1.0, help="train ≥0：四 suite 联合训练的均衡抽样（data.bal_sampler，同 train --bal）：各 suite 等量、任务均匀、任务内干净帧占这个比例")
    a("--fuse", default="", help="(pro 11-18 FU) 'd,depth,pool'：腕部融合模块 F_φ 与 IDM 联合训（需 --wrist；如 384,2,1）；IDM 的帧输入换成 s = h + tanh(g)·Δ")
    a("--fuse_stat", type=float, default=1.0, help="(pro FU) s 的逐通道均值 / 方差对齐 h 的权重"); a("--fuse_res", type=float, default=0.01, help="(pro FU) ‖s − h‖² 的权重")
    a("--fuse_ema", type=float, default=0.999, help="(pro FU) 未来帧用的 F 的 EMA 衰减（存下来的 F 也是 EMA 副本）")
    a("--save_every", type=int, default=0, help="train：每这么多步存一个续训点（<out>.resume，含优化器）；同命令重跑自动接着训")
    a("--wm_cache", default="", help="配 --wm_frac：WM 计划从这个前缀的缓存读（scripts/rc_wmcache.py 生成），不在线采样，可配 --prefetch")
    a("--init", default="", help="train：从这个 IDM / BC ckpt 的权重开始（同结构）")
    a("--p_fdrop", type=float, default=0.0, help="train：以这个概率把整段未来输入换成当前帧的复制（计划 dropout；配 --lang = 'BC 通路 + 计划'，计划置零时退化为 BC）")
    a("--end_min", type=int, default=0, help="(mika) 短 episode（MIKASA 重建版 5 任务只有 8–21 个 WM 帧）：train 帧只要求这么多帧未来在 demo 内，超出部分计划帧取最后一帧、动作平移 / 旋转置 0；默认 0 = 要求 H 帧")
    a("--far_extra", default="", help="IDM 在稠密近处 t+1..t+H 之后再读这几个远时距帧（如 16,24,32 = MS-far 的远槽；闭环取 WM 的远槽预测）")
    a("--wfut", type=int, default=0, help="离线探针：腕部流另加真实未来 t+1..t+k 帧（需 --wrist；闭环没有未来腕部，不能直接用）")
    a("--pfut", type=int, default=0, help="离线探针：本体感觉另加真实未来 t+1..t+k 帧（闭环不能直接用）")
    a("--prefetch", action="store_true", help="train：后台线程预取下一个 batch 的特征（PROBWM_FEATS_MMAP / CPU 特征时读盘与 GPU 计算重叠）")
    a("--tau_in", default="", help="(xden) 噪声感知 IDM：训练时计划帧换成 x_τ = (1−τ)ε + τ·z（逐槽 τ，见 tau_draw），τ 作为条件输入；"
                                   "值 = p_clean,p_roll,p_sort（如 0.2,0.5,0.3）；闭环 closed_loop --idm_tau pre 读 WM 未去噪完的窗口；默认关")
    a("--tau_alpha", type=float, default=13.0, help="--tau_in：层级的 shift α（同 WM 的 --alpha）"); a("--tau_n", type=int, default=11, help="--tau_in：滚动剖面的槽数（MS-far WM 的 n_fut）")
    a("--excl_suites", default="", help="(vidonly) train：训练帧去掉这些 suite（逗号分隔；目标 suite 只给视频、不给动作）；载入 / norm / val 不变；默认关")
    a("--val_suites", default="", help="(vidonly) train：val 另报这些 suite 的指标（逗号分隔）；默认关")
    a("--task_w", default="", help="(rteasy) train：任务权重表 JSON（{任务名: 权重} 或 {\"w\": {…}}，scripts/rteasy_tw.py）：训练帧按逐帧均匀 × 所属任务权重抽；默认关")
    a("--roll_frac", type=float, default=0.0, help="(flywheel) train >0：每个 batch 里这一比例从 rollout 帧（index.npz 的 roll > 0，scripts/flywheel_build.py 的合并根，配 --data）抽，"
                                                "其余从原 demo 帧；动作 / 本体标准化统计量只用原 demo 帧；另报留出 rollout（split 2）的 val；默认关")
    a("--excl_keep", type=int, default=0, help="(vidonly) 配 --excl_suites：每个被去掉的任务仍留 k 条干净训练 demo（连同其 DR 复本）的动作；默认 0")
    a("--vpp", default="", choices=["", "hid", "cat"], help="(vpp) IDM 读冻结 --wm（MS-far 滚动 WM）在闭环同款条件下的近槽隐状态（vpp_hidden：各槽按 --roll 1,0,8 的稳态剖面加噪、"
                                                         "条件路前向一次、取输出头之前最后一层的 t+1..t+H 槽 token，线性投影到 IDM 宽度）：hid = 只读隐状态 + 历史帧 + 本体；"
                                                         "cat = 隐状态 + 原来的计划帧（训练时真实未来）一起读；闭环由 closed_loop 从 msroll 那一步的前向里直接取，不多跑 WM；默认关")
    a("--vpp_new", type=int, default=0, help="(vpp) 每步新算隐状态的帧数（< bs 时用环形缓冲，每个隐状态连用 bs / vpp_new 步，省 WM 前向）；0 = bs（不复用）")
    a("--vpp_val", type=int, default=2048, help="(vpp) val 只用固定的这么多帧（每帧一次 WM 前向）"); a("--vpp_chunk", type=int, default=64, help="(vpp) WM 前向的分块大小（显存）")
    a("--vpp_compile", type=int, default=1, help="(vpp) 1 = WM 的 trunk 用 torch.compile（训练时出隐状态）；0 = eager")
    a("--rtc_dmax", type=int, default=0, help="train >0：训练时 RTC——每个样本随机取前缀长度 d（见 --rtc_p0），动作块前 d 个路点用真值、层级 1（不加噪）、加零初始化的前缀掩码嵌入，"
                                               "只在其余位置算 flow 损失；推理由 closed_loop Policy.rtc_prefix（仿真：robotwin_eval --sim_delay / --rtc）给已发出的路点作前缀；只接 flow 头；默认 0 = 关")
    a("--rtc_p0", type=float, default=1 / 3, help="--rtc_dmax 时 d = 0 的概率（其余在 1..rtc_dmax 均匀）")
    a("--accum", type=int, default=1, help="train：梯度累积——全局 --bs 不变，每步分这么多个微批前向 / 反传（显存放不下整批时用，等价整批的平均损失）；只接普通 flow 头 IDM（可配 --rtc_dmax）；默认 1")
    a("--val_chunk", type=int, default=0, help="train：val 每块的帧数（只省显存，不改训练）；默认 0 = 1024")
    a("--wstat_cpu", action="store_true", help="train：腕部特征的归一化统计量（固定抽 20000 个训练帧）在 CPU 上算（需 PROBWM_FEATS_MMAP=1 的分片库）；"
                                              "默认在 GPU 上算，启动时要多约 30 GB 显存（与批大小无关），小显存卡上打开；结果只差浮点舍入")
    a("--wpf", type=int, default=0, help="降频档：帧—路点比，IDM 的一帧 = 这么多个路点（须是 STRIDE 的倍数，如 4）。数据根不变（帧 k ↔ a_{2k}），按 delta = wpf / 2 隔帧取"
                                         "（同 train --delta）：历史 / 未来 / 本体 / 腕部帧间隔都是 wpf 个路点，动作块 = wpf·H 个路点（--wpf 4 配 --H 4 = 16 个路点）；存进 ckpt args，"
                                         "推理时 Policy / robotwin_eval 从 IDM ckpt 读（每次决策执行 wpf 个路点、滚动缓存每次平移 1 帧）；WM 须用 train --delta wpf/2 训；默认 0 = 2（原行为）")
    e = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    if e.cmd == "train":
        e.wm = e.wm or "runs/goal/abs_a13/ckpt.pt"
        e.out, e.fsteps = e.out or f"runs/goal/idm/{e.kind}.pt", e.fsteps or 10
        train(e)
    else:
        evaluate(e)


if __name__ == "__main__":
    main()
