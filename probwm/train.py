import argparse
from argparse import Namespace
import contextlib
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from . import fm
from .data import Feats
from .model import WM


def get_args(argv=None):
    p = argparse.ArgumentParser()
    a = p.add_argument
    a("--data", default="data"); a("--encoder", default="dinov3_vits16"); a("--text", default="flan-t5-base")
    a("--suites", default="libero_goal", help="逗号分隔；空字符串表示全部"); a("--synthetic", action="store_true")
    a("--mode", default="abs", choices=["abs", "res", "det", "cv"], help="cv：目标为相对匀速外推 z_t + j(z_t − z_{t-1}) 的残差（/rs）")
    a("--K", type=int, default=4); a("--delta", type=int, default=1)
    a("--d", type=int, default=512); a("--depth", type=int, default=8); a("--heads", type=int, default=8)
    a("--ctx_depth", type=int, default=2); a("--head_w", type=int, default=1024)
    a("--alpha", type=float, default=3.0); a("--p_uni", type=float, default=0.3)
    a("--sigma", type=float, default=1.0); a("--p_drop", type=float, default=0.1)
    a("--p_orig", type=float, default=0.0, help="增广表时，训练用原指令的概率（其余均匀取改写）")
    a("--txt_noise", type=float, default=0.0, help="训练时给文本嵌入加的高斯噪声（相对每个 token 的 RMS）")
    a("--prior", action="store_true", help="单帧无条件先验 p(z_t)：目标为当前帧，历史置零、语言全丢（观测净化用，需 --n_fut 1）")
    a("--lam_pc", type=float, default=0.0, help=">0：改写一致性损失（第二条改写 + 独立文本噪声下的速度场一致），需要增广表")
    a("--p_warm", type=float, default=0.0, help=">0：训练时模拟热启动的样本比例"); a("--warm_hi", type=float, default=0.4)
    a("--init", default="", help="从已有 ckpt 初始化（新增参数保持零初始化）")
    a("--init_skip_mismatch", action="store_true", help="(explore) --init 时跳过形状不同的参数（如换文本编码器后 t_in 的输入维），保持新初始化")
    a("--init_offs", action="store_true", help="(xden2) --init 的源 ckpt 是 --tgt_offs 稀疏时距 WM（如 XF5 的 1..8,16,24,32,48,64）时，fpos 时间槽按时距对齐：目标槽取源里同一时距的槽，"
                                              "源里没有的时距取最近的（canon_slot 的 t+0 = 源的当前帧历史槽 K−1）；不加时按 chunk 源的旧映射（稀疏源的远槽会错位）；默认关")
    a("--n_fut", type=int, default=1, help=">1：一次联合生成 n_fut 帧（FlowWM 式）")
    a("--xtok_drop", type=float, default=0.0, help="(rteasy) 配 --tgt_tok：训练时每个样本以这个概率把只作上下文的额外视角 token 整组置 0（标准化空间，= 视角缺省）；默认 0 = 关")
    a("--xin_zero", action="store_true", help="(rteasy) 配 --tgt_tok：额外视角用单独的输入层（零初始化）与零位置嵌入，热启动时从无输入开始学；默认关 = 与预测视角共用 c_in")
    a("--xtok_last", action="store_true", help="(pro) 配 --tgt_tok：只作上下文的额外视角只取当前帧（历史槽 K−1）的 token，其余历史帧只给预测视角；默认关 = K 帧都给")
    a("--fuse", default="", help="(pro 11-18 FU) 冻结的腕部融合模块 F_φ（idm --fuse 训出的 IDM ckpt，取其中的 fuse）：多视角库（头部 + 腕部）每帧现算 s = F(h, w)，WM 的上下文与目标都换成 s（每帧只剩头部的 T 个 token）")
    a("--tgt_tok", type=int, default=0, help=">0：多视角特征库（info.json 的 views）里每帧只预测前这么多个 token（第一个视角），其余视角只进上下文")
    a("--tgt_off", type=int, default=1, help=">1：目标从 z_{t+tgt_off} 开始（远点 WM：--n_fut 1 --tgt_off 8 只预测 0.8 s 后的一帧；历史间隔仍为 1 帧）")
    a("--sam", type=float, default=0.0, help=">0：SAM（sharpness-aware minimization）的半径 ρ，每步两次前向 / 反向")
    a("--tgt_stride", type=int, default=1, help="目标帧间隔：--n_fut 2 --tgt_off 4 --tgt_stride 4 → 关键帧 t+4、t+8"); a("--lam_tc", type=float, default=0.0, help="时间一致性损失权重")
    a("--bs", type=int, default=256, help="每卡"); a("--lr", type=float, default=3e-4); a("--wd", type=float, default=0.01)
    a("--steps", type=int, default=50000); a("--warmup", type=int, default=1000); a("--ema", type=float, default=0.999)
    a("--compile", action="store_true"); a("--log", type=int, default=200); a("--save", type=int, default=5000)
    a("--keep", default="", help="逗号分隔的步数：到这些步时另存一份 ckpt_s<步>.pt（同步数对比用，不改学习率日程）")
    a("--resume_every", type=int, default=0, help=">0：每这么多步存一个续训点（<out>/resume.pt，含优化器 / EMA / 日程）；同命令重跑自动接着训")
    a("--accum", type=int, default=1, help="梯度累积：每步 accum 个 --bs 小批（显存放不下整批时，如稠密 32 槽）")
    a("--out", default="runs/dev"); a("--seed", type=int, default=0)
    a("--n_demo", type=int, default=0, help=">0：WM 只用每个任务这么多条训练 demo 的视频（idm.labeled）")
    a("--trainval", action="store_true", help="val demo 也参与训练（标准 LIBERO：每任务 50 条全用；val 只作监控）")
    a("--canon", default="", help="逗号分隔的配对数据根（camaug.py 格式）：训练特征规范化模型 z_obs → z_can（probwm/canon.py；"
                                  "需 --K 1 --n_fut 1；上下文 = z_obs，目标 = z_can，语言为空；--p_warm 1 为从加噪 z_obs 出发的桥）")
    a("--canon_norm", default="runs/goal/chunk8_t5l_paug_tnoise/ckpt.pt", help="--canon：特征用这个 WM ckpt 的 norm")
    a("--canon_pid", type=float, default=0.2, help="--canon：恒等对（输入为干净帧）的比例")
    a("--canon_wm", action="store_true", help="--canon 且 K>1：规范目标 WM（历史 = 扰动帧，目标 = 干净的未来帧，带指令；可 --init 主 WM 微调）；"
                                             "不加时 K>1 为时序规范化器（目标 = 干净的当前帧，语言为空）")
    a("--obj", default="", help="物体 token（probwm/objtok.py）：data 根下的覆盖率文件，如 objseg/cov.npy（oracle）或 objseg/pred_<名>.npy")
    a("--obj_mode", default="cat", choices=["cat", "ground"], help="cat：7 个类别槽位；ground：2 个槽位 = 指令的操作物体 / 目标（按任务的 bddl 目标取类别）")
    a("--p_odrop", type=float, default=0.1, help="训练时每个物体 token 置为不可见的概率")
    a("--p_main", type=float, default=0.0, help=">0：数据根的 index.npz 带 aug（如 vidmix merge 的干净 + 生成 demo）时，每个 batch 里干净帧（aug == 0）的比例（同 vidmix --p_main）")
    a("--lw_motion", type=float, default=0.0, help="FM 损失的逐 patch 加权：+ α·（该 patch 目标帧相对当前帧的变化 / 样本平均，截到 5）")
    a("--lw_slot", action="store_true", help="(ctrl) 配 --lw_motion：变化量按每个目标槽（帧）内部的均值归一、权重也逐槽归一到均值 1 → 只在帧内重分配（变化 / 末端区域），不改槽间配比；需目标恰为 n_fut 个整帧；默认关")
    a("--lw_obj", type=float, default=0.0, help="FM 损失的逐 patch 加权：+ β·（当前帧里本任务操作物体 / 目标的覆盖率，--lw_cov 文件）")
    a("--lw_cov", default="objseg/cov.npy", help="--lw_obj 用的覆盖率文件（data 根下或绝对路径；7 类别或定位头 2 通道）")
    a("--aux_eef", type=float, default=0.0, help=">0：辅助头预测 chunk 内各帧末端位移（robot0_eef_pos，相对当前帧，单位 dm）的损失权重")
    a("--tgt_offs", default="", help="任意稀疏目标时距（逗号分隔，如 1,8,16 或 1,2,4,8,16,24），一起作为 n_fut 个目标槽（各自的时间槽嵌入）；需 --n_fut = 个数")
    a("--roll", type=float, default=0.0, help="滚动去噪训练（fm.loss roll）：以这个概率用滚动噪声剖面（近槽干净、远槽噪声大），其余同步；推理 closed_loop --roll")
    a("--roll_churn", type=float, default=0.0, help="--roll 的样本以 0.5 的概率逐槽多加 U[0, 这个值] 的噪声（均匀时间 u 上），覆盖推理时的回噪")
    a("--far_clamp", type=int, default=0, help="--tgt_offs：anchor 只要求有这个时距以内的未来；更远的目标槽超出 demo 末尾时取最后一帧（任务已完成的稳态）；0 = 要求全部时距")
    a("--roll_sort", action="store_true", help="--roll 的剖面改成逐槽独立 U(0,1) 按槽降序排（任意单调：近槽轻、远槽重）；多尺度滚动用")
    a("--goal_first", type=float, default=0.0, help="(rteasy 臂 G) 以这个概率把训练样本的噪声剖面换成目标先行两阶段（closed_loop --goal_first）的状态："
                                                     "阶段 1 近槽（时距 < 9）纯噪声、远槽任意，阶段 2 远槽干净、近槽任意（fm.loss gf）；需 --tgt_offs（近槽在前）+ --roll；默认 0 = 关")
    a("--alpha_far", type=float, default=0.0, help="(rteasy 臂 L) >0：滚动样本里远槽（时距 ≥ 9）的 shift 参数改用这个（近槽仍 --alpha；同步样本不变）；"
                                                    "闭环 msroll 读 ckpt 的这个值逐槽加噪，远槽跨决策保留更多上次的估计；需 --tgt_offs（近槽在前）+ --roll --roll_sort；默认 0 = 关")
    a("--lead", type=float, default=0.0, help="(rteasy 臂 G2) 以这个概率把训练样本换成远槽领先的多步去噪（closed_loop --lead_k）的中间状态（fm.loss lead）；"
                                               "需 --tgt_offs（近槽在前）+ --roll；默认 0 = 关")
    a("--slot_noise", action="store_true", help="逐槽独立的噪声层级（diffusion forcing 式），配 --tgt_offs 的稀疏时距；推理可用交错去噪（closed_loop --stagger）")
    a("--aux_offs", default="", help="(A) 训练专用的辅助目标槽：远时距帧（逗号分隔，如 16,24），与 chunk 一起去噪，chunk token 看不到它们；推理不变")
    a("--aux_kf", action="store_true", help="(A) 再加一个辅助槽：下一个接触关键帧")
    a("--ev_slot", type=int, default=0, help="(explore) --tgt_offs 的槽之后再加 1 个普通目标槽 = 下一个语义事件（之后第一次夹爪指令翻转那一帧，没有就取 demo 末帧；时距可变），"
                                             "参与滚动训练 / 推理（closed_loop 每次决策从纯噪声与其他槽一起去噪）；值 = 名义时距（--hw 权重用）；需 --n_fut = len(tgt_offs) + 1")
    a("--ev_roll", action="store_true", help="(explore) --ev_slot 的滚动版：roll_sort 剖面里事件槽按它的实际时距（到下一次夹爪翻转的帧数）和其他槽一起排序定层级；"
                                             "闭环里事件槽的估计跨决策保留、按估计时距的名次加噪回去再细化（见 closed_loop.Policy.rolling）")
    a("--aux_notext", action="store_true", help="(A) 消融：辅助槽交叉注意力读丢掉语言的上下文（区分表征学习 vs 条件利用）")
    a("--hw", type=float, default=0.0, help="时距加权：第 j 个目标槽（j = 1..n_fut）的损失权重 ∝ j^hw，均值归一为 1（0 = 不加权，1 = 线性，2 = 平方）；"
                                            "同步噪声时在训练循环里经逐 token 的 pw 乘上，逐槽噪声（--slot_noise / --roll）时交给 fm.loss 的逐槽权重，两者互斥")
    a("--far_w", type=float, default=1.0, help="(cap) 远槽加权：在 --hw 的逐槽权重上把远槽（--tgt_offs 里时距 ≥ 9 的槽）再乘这个系数、重新均值归一为 1；"
                                               "近槽内部、远槽内部的配比不变，只改近 / 远的总配比（XF5 的 hw 0.5 下远槽占 64%，×2 → 78%，×4 → 88%）。"
                                               "只接逐槽噪声路径（--roll / --slot_noise）+ --hw；val 仍按原 --hw 权重报告（与旧臂可比）；默认 1 = 不改")
    a("--aux_ae", type=float, default=0.0, help="(cap) 远槽末端位移辅助头的损失权重：目标流最后一层每个远槽（--tgt_offs 里时距 ≥ 9）的隐状态（槽内 token 平均）→ 小 MLP"
                                                "→ 该槽时刻相对当前帧的末端位移（--ae_keys 各取 xyz，单位 dm；超出末尾同 --far_clamp 取最后一帧）；逐槽按 (1 − τ) 加权（越接近噪声越要靠推断）；0 = 关")
    a("--ae_keys", default="left_ee_f,right_ee_f", help="(cap) --aux_ae 的状态键（meta/states/<键>.npy，前 3 维 = xyz；RoboTwin 的 DR 复本用 scripts/cap_eefill.py 从源 clean demo 补齐的 *_f）")
    a("--moe", default="", help="(cap) 目标流各块的 FFN 换成 MoE：'E,k,m[,cf]' = E 个专家、逐 token top-k、专家宽 m·d、训练容量因子 cf（默认 2；如 8,2,2：激活 FFN 算力同稠密 4d，"
                                 "总参数约 +4 倍 FFN）；训练按容量丢超出的分派（可编译），推理不丢；负载均衡损失系数 --moe_lb；上下文编码层仍是稠密 FFN；空 = 关")
    a("--moe_lb", type=float, default=0.01, help="(cap) --moe 的 Switch 式负载均衡损失系数")
    a("--moe_lang", action="store_true", help="(cap MoE-T) --moe 的路由器另吃池化的语言嵌入（同 WM 的文本特征，按 mask 平均、CFG 丢掉时为 0；零初始化）；默认关")
    a("--moe_shared", action="store_true", help="(cap MoE-T) --moe 加一个所有 token 都走的共享专家（宽同专家）；默认关")
    a("--moe_bias", type=float, default=0.0, help="(cap MoE-T) >0：无辅助损失的负载均衡（DeepSeek-V3 偏置法，每步按负载 ± 这个步长调选专家用的偏置，不进梯度；"
                                                  "配 --moe_lb 0；单卡用——DDP 会广播 buffer）；默认 0 = 关")
    a("--lam_aux", type=float, default=1.0, help="(A) 辅助槽的损失权重（相对 chunk token）")
    a("--kf_pred", default="", help="J2：--kfcond 训练时一部分样本的关键帧换成这个关键帧 WM（train --keyframe）的预测（scheduled sampling）")
    a("--p_kpred", type=float, default=0.5, help="--kf_pred：用预测关键帧的比例")
    a("--joint_kf", action="store_true", help="J3：一个 WM 同时去噪 {下一个接触关键帧, chunk 1..n_fut-1}（第 0 槽 = 关键帧；--n_fut 9 = 关键帧 + 8 帧）")
    a("--kfcond", action="store_true", help="两级：chunk WM 以下一个接触关键帧的（真实）特征为条件（额外一组上下文 token），配 --p_kdrop；测试时关键帧由 --keyframe WM 预测")
    a("--pair_loss", type=float, default=0.0, help="--keyframe：配对反事实损失的权重 λ（同组配对 demo 的预测关键帧差 ≈ 真实差）"); a("--pair_bs", type=int, default=32)
    a("--pair_json", default="", help="配对组号的 gen.json（默认 <数据根>/meta/gen3k.json）")
    a("--p_gen", type=float, default=0.0, help="配 --p_main：任务号 ≥ --gen_tid 的 demo（新任务，如组合泛化的生成数据）单独成一组，在 batch 里占这个比例（从非干净部分扣）")
    a("--gen_tid", type=int, default=10, help="--p_gen：任务号下限（合并根里原任务在前、新任务追加在后）")
    a("--p_lg", type=float, default=0.0, help="配 --p_main：生成 demo（编号 ≥ 1000）在 batch 里的固定比例，其余增广（DR 复本）占 1 − p_main − p_lg")
    a("--hist_noise", type=float, default=0.0, help="训练时历史上下文加噪：τ_h ~ U(0, 值)，ctx ← (1 − τ_h)ctx + τ_h ε（削弱历史捷径；推理用干净历史）")
    a("--hist_drop", type=float, default=0.0, help="训练时较早的 K−1 个历史帧各自以这个概率置 0（当前帧保留）")
    a("--obs_corrupt", type=float, default=0.0, help="(pro) 训练式吸收（预注册 logs/pro.md 11-15 16:50，11-15 17:00 修订）：每个样本以这个概率给上下文的随机连续 1–K 帧加一种通用损坏——"
                                                     "整帧丢弃（标准化空间置 0）/ 冻结（换成这段之前的一帧，段从第 0 帧起时用第 0 帧）/ 黑屏（--obs_black 的特征）/ 随机块遮挡"
                                                     "（16×16 网格上一个 4–10 × 4–10 的矩形置 0）/ 逐 patch 损坏（随机 10–50% 的 patch 换成噪声或同集上下文里别一帧的对应 patch）/ "
                                                     "时延（这段帧换成 1–--obs_delay 帧之前的帧，早于集开头时用集开头），六种等概率；目标不变（仍是干净帧）；只在训练 batch 上，默认 0 = 关")
    a("--obs_kinds", default="", help="(pro X3cF，11-16 预注册) --obs_corrupt 只抽这几类（逗号分隔：drop / freeze / black / box / patch / delay，均分）；默认空 = 六类均分（原行为）")
    a("--obs_delay", type=int, default=8, help="(pro) --obs_corrupt 时延类的最大延迟（帧，均匀取 1..这个值）")
    a("--t0_head", type=int, default=0, help="(pro 11-17 X3cSG) >0：独立的 t+0 小头（这么多层），读上下文编码器里当前帧的 token 并 stop-gradient，回归同状态的干净当前帧"
                                                 "（DR 配对同 --canon_slot 的 pair_clean）；WM 主体的目标槽与损失不变（不加 t+0 槽）；默认 0 = 关")
    a("--t0_w", type=float, default=1.0, help="(pro X3cSG) --t0_head 的损失权重（标准化空间 MSE）")
    a("--t0_drop", type=float, default=0.0, help="(pro) 训练式吸收：每个样本以这个概率把上下文的当前帧（第 K 帧）置 0，t+0 槽仍预测干净的当前帧（配 --canon_slot）；与 --obs_corrupt 独立；默认 0 = 关")
    a("--obs_black", default="", help="(pro) --obs_corrupt 黑屏用的特征：npy (T, D)，编码器对全黑图像的原始特征（scripts/pro_blackfeat.py 写；训练时按本数据的 norm 标准化）")
    a("--kf_aug", type=float, default=0.0, help="--kfcond：关键帧条件增广，训练时 τ_c ~ U(0, 值) 加噪并把 τ_c 作为输入（闭环 --kf_tau 用固定的小 τ_c）")
    a("--p_kdrop", type=float, default=0.2, help="--kfcond：训练时把关键帧条件丢掉（置 0）的概率")
    a("--keyframe_src", default="gripper", choices=["gripper", "motion"], help="关键帧事件：gripper = 夹爪指令翻转；motion = 特征轨迹运动速度的局部极小（不需要动作 / 事件标注）")
    a("--keyframe", action="store_true", help="接触关键帧 WM：目标 = 下一次夹爪指令翻转那一帧的特征（可变时距；需 --n_fut 1），条件照旧（历史 + 语言 [+ where]）")
    a("--where", action="store_true", help="where 瓶颈：WM 以下一个接触点（下一次夹爪开合时的末端，投影到 16×16）的高斯热图为条件（逐 patch 加到历史和目标 token 上）；配合大的 --p_drop")
    a("--where_sigma", type=float, default=1.0, help="热图高斯宽度（patch）"); a("--where_cam", default="runs/goal/obj/eefuv.pt", help="取里面存的 agentview 相机矩阵")
    a("--where_file", default="", help="--where：热图中心改用这个逐帧文件（数据根下或绝对路径，(N, 3) = 网格 x, y, 可见；probwm/wherekf.py heat 出），"
                                        "定义 = 当前阶段（夹爪开：操作物体；合：放置目标）类别覆盖率的质心，与闭环 --where_src oracle 一致；不可见时热图全 0")
    a("--p_wdrop", type=float, default=0.0, help="--where：训练时整张热图置 0 的概率（让模型也会'没有热图'）")
    a("--rope", action="store_true", help="2D RoPE（16×16 网格上的相对位置，作用于上下文自注意力、目标自注意力和交叉注意力）代替绝对空间位置嵌入")
    a("--p_hdrop", type=float, default=0.0, help="训练时丢掉视觉上下文（历史帧 + 物体 token）的概率（与语言 dropout 独立；两路分开的引导用）")
    a("--norm_from", default="", help="特征 norm 取这个 WM ckpt 的（不按本数据根重算；同 vidmix --norm_from）")
    a("--warp", default="", choices=["", "add", "pure"], help="warp-kf（probwm/warpkf.py）：去噪器的 x̂1 = 当前帧按预测的逐 patch 2D 形变 warp 后 + 残差；"
                                                             "add = 原输出头的 x̂1 + (warp − 当前帧)（形变为 0 时等于原模型），pure = warp + 残差头（输出头清零重学）")
    a("--warp_kind", default="bilinear", choices=["bilinear", "attn"], help="--warp：bilinear = 逐 patch 2D 位移的双线性反向采样；attn = 对当前帧各 patch 的软对应（全局可达）")
    a("--warp_res", type=float, default=0.0, help="--warp：残差 R = x̂1 − warp(当前帧) 的 L2 惩罚权重（逼形变解释变化）")
    a("--warp_tv", type=float, default=0.0, help="--warp：形变场的平滑（相邻 patch 差的平方）惩罚权重")
    a("--bal", type=float, default=-1.0, help="≥0：四 suite 联合训练的均衡抽样（data.bal_sampler）：每个 batch 各 suite 等量、suite 内任务均匀，"
                                              "任务内干净帧占这个比例、其余按 suite 内 DR / 生成 demo 的帧量分（缺的并回干净）；不与 --p_main 同用")
    a("--motion_p", type=float, default=0.0, help="(mika) 任务内按动作重分配 anchor：<数据根>/meta/motion.npy（scripts/mika_motion.py）标为\"之后 H 帧内在动\"的帧在本任务里占 p，"
                                                     "任务份额不变；动的帧本来就 ≥ p 的任务不变（RememberColor-Long 的 demo 95% 是静止等待）；只接 --tgt_offs 的默认抽法；默认 0 = 关")
    a("--task_w", default="", help="(rteasy) 任务权重表 JSON（{任务名: 权重} 或 {\"w\": {…}}，scripts/rteasy_tw.py）：训练 anchor 按逐帧均匀 × 所属任务权重抽（只接 --tgt_offs 的 WM）；默认关")
    a("--bal_x", type=float, default=0.0, help="(xtask) 配 --bal：index 的 xtask 列 > 0 的 demo（同一场景配别的训练原句）单独成组，在有这组帧的任务里占这个比例，"
                                              "干净 / DR / 生成三组按原比例缩（data.bal_sampler 的 p_x）；默认 0 = 不分组（原行为）")
    a("--bal_ao", type=float, default=0.0, help="(xtask3) 配 --bal：index 的 xtask 列 ≥ 500 的 demo（dscale 的 LIBERO-90 场景加物体配对）不进原分组，"
                                                 "每个样本以这个概率换成这些 demo 的帧（帧均匀）；默认 0 = 原行为")
    a("--prefetch", action="store_true", help="后台线程预取下一个训练 batch（PROBWM_FEATS_MMAP / CPU 特征时读盘与 GPU 计算重叠）")
    a("--far_slots", default="", help="(objslot) 物体级远槽：npy (N_meta, Ks, ds) 已标准化的物体 slot 向量（scripts/objslot_slots.py，meta 帧序）；"
                                      "--tgt_offs 里时距 ≥ --far_slot_min 的槽，目标换成 slot 帧 = 前 Ks 个 token 放 slot 向量（不足 D 维补 0）、其余 token 为 0；"
                                      "损失按 token 加权：slot token T/Ks（每槽总权重同稠密帧）、补零 token --far_pad_w。只支持 Feats 的 --tgt_offs 路径；闭环配 closed_loop --far_sep")
    a("--far_slot_min", type=int, default=9, help="--far_slots：时距 ≥ 它的目标槽用 slot 帧")
    a("--canon_stn", default="", choices=["", "affine", "homog"], help="(pro) 几何 canon：上下文帧先过空间变换头（model.STN，回归仿射 / 单应、"
      "grid_sample 对齐 16×16 特征网格，恒等初始化），再接原 canon；配 --canon 用，默认关")
    a("--stn_reg", type=float, default=1e-2, help="(pro) --canon_stn 的正则权重：λ·‖θ − θ_I‖²")
    a("--canon_slot", action="store_true", help="(xden) canon 并入 WM：--tgt_offs 以 0 开头（如 0,1,…,8,16,24,32），上下文仍是本 demo（DR 复本）的观测帧，"
                                                "全部目标槽（含 t+0 = 当前帧）换成同一仿真状态的干净帧（pair_clean）；闭环里 t+0 槽的输出代替独立 canon 给 IDM 的历史（closed_loop Policy.act）；默认关")
    a("--bal_dr", type=float, default=0.0, help="(pro 11-17 B3) 配 --bal：任务内干净 / DR / 生成三组里 DR 组固定占这个比例（干净仍为 --bal，生成拿剩下的）；默认 0 = 按帧量分")
    a("--hist_null", type=float, default=0.0, help="(pro 11-17 B3) 训练时每个样本以这个概率把过去 K−1 帧全部置 0（标准化空间的均值帧），只留当前观测帧，目标照旧"
                                                   "（canon_slot 时 t+0 = 同状态干净帧）：把\"单帧 → 规范帧\"并进同一套权重，供闭环 --wm_canon 的第 1 遍用；默认 0 = 关")
    a("--canon_hist", action="store_true", help="(xden2) 配 --canon_slot：t+0 输出回灌 WM 历史——上下文里过去的 K−1 帧换成同一仿真状态的干净帧（pair_clean）"
                                                "+ 模拟闭环 t+0 估计误差的噪声（--ch_noise），当前帧仍是本 demo 的观测；闭环里 WM 的历史 = 之前各步 t+0 槽的输出（closed_loop Policy.ch_hist）；默认关")
    a("--ch_noise", default="", help="--canon_hist：t+0 残差统计（scripts/xden2_res.py 的 .pt：逐维 sd，DR 复本 / 干净 demo 分开，相邻帧 AR(1) 相关 rho，可选逐通道增益 gain (2, D)：DR / 干净）；"
                                     "过去帧 = gain·干净帧 + sd·AR(1) 高斯噪声；空 = 只换干净帧、不加噪声（对照）")
    a("--ch_scale", type=float, default=1.0, help="--canon_hist：--ch_noise 的 sd 乘这个倍数")
    a("--far_pad_w", type=float, default=0.05, help="--far_slots：slot 帧里补零 token 的损失权重（学会输出 0，滚动推理时缓存的估计与训练一致）")
    a("--hold_tasks", default="", help="(demo) 逗号分隔的 task_id：这些任务的全部 demo 不进训练（train / val anchor 都去掉；留出任务评测用）；默认空 = 原行为")
    a("--hold_x", default="", help="(demo) 逗号分隔的 xtask 配对号（index 的 xtask 列 − 1，如 1,11 = x001,x011）：这些配对的 demo 不进训练、也不作示范；默认空")
    a("--demo_bank", default="", help="(demo) 示范库（scripts/demo_data.py bank 的输出 .pt：每条 demo M 帧、每帧池化成 grid² 个 token）；给了就开示范条件：每个样本配一条同任务的另一条 demo，"
                                      "作为 WM 的额外上下文 token（model.WM n_dfr / n_dtok）；默认空 = 原行为")
    a("--p_ddrop", type=float, default=0.3, help="--demo_bank：训练时整段示范丢掉的概率（与语言 --p_drop 独立；示范 CFG 用）")
    a("--p_dgoal", type=float, default=0.4, help="--demo_bank：示范在场时只留最后一帧（目标图像条件）的概率")
    a("--demo_lg", type=float, default=0.5, help="--demo_bank：任务在示范库里有布局生成（lg）demo 时，从 lg 取示范的概率（其余从数据根同任务的别的 demo 取，不取 anchor 自己那条）")
    a("--self_hist", default="", help="(mika) M,g：全历史上下文——本 episode 过去的帧（[episode 开头, t−K]，不越过当前时刻；≤ M 帧时全取，多于 M 时均匀取 M 帧）"
                                      "每帧 16×16 池化成 g×g 个 token，复用 --demo_bank 的示范 token 通路（model.WM n_dfr = M、n_dtok = g²），'进度'嵌入换成多久以前 (t − s)/1024；"
                                      "闭环 probwm/mikasa_eval 维护同规则的历史缓存（SelfHist / sh_pick）；不与 --demo_bank 同用；默认空 = 原行为")
    a("--sh_drop", type=float, default=0.1, help="--self_hist：训练时整段历史丢掉的概率")
    a("--sh_time", default="abs", choices=["abs", "rel"], help="--self_hist：历史 token 的时间嵌入——abs = 多久以前 (t − s)/1024（原行为）；"
                                                              "rel = 相对时距 (t − s)/(t − s0)（尺度无关：不再告诉 WM'开局多久了'）")
    a("--sh_stride", type=int, default=4, help="--sh_grid recent 的步长（WM 帧；HiLo-WAM 的稀疏历史是 8 帧、步长 4）")
    a("--sh_grid", default="start", choices=["start", "mid", "recent"], help="--self_hist：可用帧多于 M 时的均匀网格——start = 含两端（必取开局帧，原行为）；"
                                                                  "mid = 中点网格 s0 + ⌊(j + ½)·n/M⌋（不必取开局帧）")
    a("--sh_slotdrop", type=float, default=0.0, help="--self_hist：训练时每个样本的逐槽丢弃率 ~ U(0, p)（打乱'有效帧数 = 已过去帧数'这条时钟）；闭环不丢")
    a("--sh_tscale", default="", help="--self_hist：lo,hi——训练时每个样本的时距随机伸缩倍数 ~ U(lo, hi)，作用于 abs 时间嵌入与 --sh_alibi 的 Δt；闭环 = 1")
    a("--sh_alibi", action="store_true", help="--self_hist：时间衰减注意力（ALiBi 式）——历史 token 作为 key 时 logit 加 −m_h·Δt（Δt = 离当前的帧数，按 --sh_tscale 伸缩），"
                                               "m_h 按头取几何级数、前 2 个头为 0（保留长程检索）；上下文编码器自注意力与目标流交叉注意力都加；默认关")
    a("--mem", type=int, default=0, help="(mika C2) 记忆 token 数 S：WM 上下文多 S 个记忆槽，每个记忆步读上一步的记忆、同时写出新记忆（model.WM n_mem）；"
                                         "训练沿 demo 时间展开（--mem_bptt 步带梯度，之前从 episode 开头无梯度预热），截断 BPTT；0 = 原行为")
    a("--mem_every", type=int, default=4, help="--mem：记忆步的间隔（WM 帧；记忆只在 episode 内第 0、r、2r … 帧写，其余决策只读）")
    a("--mem_bptt", type=int, default=8, help="--mem：每个样本带梯度展开的记忆步数 L（截断 BPTT）")
    a("--mem_drop", type=float, default=0.1, help="--mem：训练时整段记忆清零的概率（这一样本的 L 步都读开头的记忆）")
    a("--mem_skip", type=float, default=0.0, help="--mem：训练时每个记忆步不写（记忆保持不变）的概率（打乱'写了几次 = 过了多久'这条时钟）；闭环不跳")
    a("--sh_jit", type=float, default=0.0, help="--self_hist：训练时选帧随机化——可用帧多于 M 时，均匀网格加随机相位、每槽再抖 ±j/2 个网格步长"
                                                "（对'线索帧在不在历史里、在哪个位置'鲁棒）；闭环仍用确定规则；0 = 确定规则（原行为）")
    a("--train_tasks", default="", help="(dscale) json 文件（scripts/dscale_sub.py）{\"tasks\": [task_id…], \"x\": [xtask 配对号…]}：只用这些任务的普通 demo 与这些 xtask 配对的 demo 训练，"
                                        "其余 demo 与 --hold_tasks / --hold_x 同样处理（train / val anchor 去掉、不进示范池）；评测的留出组仍按 --hold_tasks；默认空 = 全部")
    a("--any_far", default="", help="(hz2) 任意时距远槽：lo,hi,kmin,kmax（如 12,96,3,5）。--tgt_offs 里时距 < 9 的近槽照旧，远槽每个样本从 [lo, hi] 里不放回地抽 k 个"
                                    "（概率 ∝ 1/时距，即近似对数均匀；k 每步在 [kmin, kmax] 里随机、各 rank 相同），升序排；目标槽的时间槽嵌入按整数时距查表（model.WM n_hz = hi）。"
                                    "--n_fut = 近槽数 + kmax，--tgt_offs 的远槽只作闭环默认查询；--hw 按各样本的实际时距加权；默认空 = 原行为")
    a("--det_hw", action="store_true", help="(ctrl) --mode det（确定性回归）也按 --hw 的逐槽权重加权（与 flow 版同一损失配比；需 --roll / --slot_noise 使 sw 生效）；默认关 = 原行为")
    a("--far_drop", type=float, default=0.0, help="(faraux) 远槽只作训练辅助目标的对照：每步随机 round(p·bs) 个样本整组丢掉远槽 token（--tgt_offs 里时距 ≥ 9 的槽），"
                                                  "只带近槽去噪（近槽看不到远槽，滚动剖面按近槽数抽），近槽 token 的损失权重不变（远槽项直接去掉）；其余样本照常。"
                                                  "推理不带远槽用 closed_loop --near_only；需 --tgt_offs 与 --roll / --slot_noise；默认 0 = 原行为")
    a("--sync_slots", action="store_true", help="(gawm P1) 同步去噪训练：逐槽权重（--hw 按真实时距）同滚动版，但每个样本的所有槽同一噪声层级（不用滚动剖面）；"
                                                "推理不加 --roll（每次决策 scratch4 从噪声同步采整窗）；需 --tgt_offs，不与 --roll / --slot_noise / --far_drop / --far_aux 同用；默认关")
    a("--far_aux", action="store_true", help="(gawm) 严格 IFP 式远头：--tgt_offs 里时距 ≥ 9 的远槽只作训练辅助目标——近槽 token 在每一层都看不到远槽 token（单向 mask），"
                                             "远槽看近槽与上下文；远槽目标、逐槽权重（--hw）同普通 MS-far；--roll_sort 剖面近槽 / 远槽各自排（近槽同 MSN）。"
                                             "参数形状同普通 MS-far；推理用 closed_loop --near_only（远槽不去噪、不滚动）；需 --tgt_offs（近槽在前）+ --roll --roll_sort；默认关")
    a("--uwm", type=int, default=0, help="(uwm) >0：WM 另加这么多个动作 token + 1 个本体 token（probwm/uwm.py；UWM 式联合去噪，动作有自己的 τ），动作块 / 本体的定义与标准化同 --uwm_idm；"
                                         "需 --tgt_offs 与 --roll / --slot_noise 的普通路径（RoboTwin 全连续动作）；闭环 closed_loop --wm_act；默认 0 = 原行为")
    a("--uwm_idm", default="", help="(uwm) 取动作块定义（H、本体 key、h5）与标准化统计量（astats、pmu / psd）的 IDM ckpt（idmPL）")
    a("--uwm_w", type=float, default=1.0, help="(uwm) 动作 FM 损失的权重 λ"); a("--uwm_pic", type=float, default=0.25, help="(uwm) 整组画面槽给干净真值（τ = 1）的样本比例")
    a("--uwm_alpha", type=float, default=3.0, help="(uwm) 动作层级的 shift α（同 idmPL 的 flow 头）"); a("--uwm_puni", type=float, default=0.3, help="(uwm) 动作层级不 shift 的比例")
    a("--uwm_mask", action="store_true", help="(ptdata P2) 读数据根的 meta/act_mask.npy（逐帧，1 = 有兼容的真动作）：动作项只在这些样本上算，其余样本本体置 0；默认关")
    a("--uwm_h5", default="", help="(ptdata P2) 动作块的 HDF5 根改用这个（默认用 --uwm_idm 的 h5）"); a("--uwm_lat", type=int, default=0, help="(ptdata P2) >0：动作 token 另出潜在动作 = 相邻帧特征残差平均池化到 g×g（只作目标；配 --uwm_mask 时只在无真动作的样本上算）")
    a("--uwm_latw", type=float, default=1.0, help="(ptdata P2) 潜在动作项的权重")
    return p.parse_args(argv)


def build(a, dims):
    D, T, Dt = dims
    n_obj = 0 if not getattr(a, "obj", "") else 2 if a.obj_mode == "ground" else 7
    n = getattr(a, "tgt_tok", 0) or T  # --tgt_tok：每帧只预测前 n 个 token（第一个视角），其余视角只作上下文
    cls, wk = WM, {}
    if getattr(a, "warp", ""):  # warp-kf：同一主干，另接形变头
        from .warpkf import WarpWM
        cls, wk = WarpWM, {"warp": a.warp, "kind": getattr(a, "warp_kind", "bilinear")}
    if getattr(a, "uwm", 0):  # (uwm) 动作组 token：在 WM 建好之后挂上（probwm/uwm.py）
        from .uwm import attach
        return attach(build(argparse.Namespace(**{**vars(a), "uwm": 0}), dims), a)
    m = cls(D, n, a.K, Dt, a.d, a.depth, a.heads, a.ctx_depth, a.head_w, getattr(a, "n_fut", 1),
              getattr(a, "p_warm", 0.0) > 0, n_obj, D + 18 if n_obj else 0,  # 18 = objtok.P_DIM
              3 * getattr(a, "n_fut", 1) if getattr(a, "aux_eef", 0.0) > 0 else 0, getattr(a, "rope", False), getattr(a, "where", False),
              n_xtok=T - n, kfc=getattr(a, "kfcond", False),
              n_auxs=len([x for x in str(getattr(a, "aux_offs", "")).split(",") if x]) + int(getattr(a, "aux_kf", False)), kfa=getattr(a, "kf_aug", 0.0) > 0,
              **({"n_dfr": a.demo_m, "n_dtok": a.demo_grid ** 2} if getattr(a, "demo_bank", "") or getattr(a, "self_hist", "") else {}),
              **({"n_hz": int(a.any_far.split(",")[1])} if getattr(a, "any_far", "") else {}),
              **({"n_ae": 3 * len(a.ae_keys.split(","))} if getattr(a, "aux_ae", 0.0) > 0 else {}),
              **({"t0h": a.t0_head} if getattr(a, "t0_head", 0) else {}),
              **({"moe": (lambda v: (int(v[0]), int(v[1]), float(v[2]), float(v[3]) if len(v) > 3 else 2.0))(a.moe.split(",")) + (getattr(a, "moe_lb", 0.01),)
                  + ((int(getattr(a, "moe_lang", False)), int(getattr(a, "moe_shared", False)), float(getattr(a, "moe_bias", 0.0)))
                     if getattr(a, "moe_lang", False) or getattr(a, "moe_shared", False) or getattr(a, "moe_bias", 0.0) > 0 else ())} if getattr(a, "moe", "") else {}),
              **({"alibi": True} if getattr(a, "sh_alibi", False) else {}), **({"n_mem": a.mem} if getattr(a, "mem", 0) else {}),
              **({"stn": a.canon_stn} if getattr(a, "canon_stn", "") else {}), **wk)
    if getattr(a, "far_aux", False):  # (gawm) 近槽数 = --tgt_offs 里时距 < 9 的前缀；model.trunk 按 n_main 做单向 mask（闭环 --near_only 时槽数 = n_main，不 mask）
        m.n_main = sum(int(x) < 9 for x in a.tgt_offs.split(","))
    if getattr(a, "xin_zero", False) and T - n > 0:  # (rteasy 11-16) 只作上下文的额外视角用单独的输入层（零初始化），位置嵌入也置 0：热启动时新视角从"无输入"开始学
        m.cx_in = torch.nn.Linear(D, a.d)
        torch.nn.init.zeros_(m.cx_in.weight), torch.nn.init.zeros_(m.cx_in.bias), torch.nn.init.zeros_(m.xpos)
    if getattr(a, "xtok_last", False) and T - n > 0:  # (pro 11-16 X3cW) 额外视角（腕部）只取当前帧：上下文多 n_xtok 个 token 而不是 K × n_xtok
        m.xlast = True
    return m


def hold_mask(root, data, a):
    """(demo) --hold_tasks / --hold_x → 逐帧 bool（True = 留出，不进训练）。行按 Feats.load 的 suite 筛选对齐。"""
    import numpy as np
    root = Path(root)
    ix, tasks = np.load(root / "meta/index.npz"), json.loads((root / "meta/tasks.json").read_text())
    keep = np.isin([tasks[t]["suite"] for t in ix["task_id"]], a.suites.split(",")) if a.suites else np.ones(len(ix["demo"]), bool)
    ht = [int(x) for x in str(getattr(a, "hold_tasks", "")).split(",") if x]
    hx = [int(x) for x in str(getattr(a, "hold_x", "")).split(",") if x]
    h = np.isin(ix["task_id"][keep], ht)
    if hx:
        assert "xtask" in ix.files, "--hold_x 需要 index 有 xtask 列"
        h |= np.isin(ix["xtask"][keep].astype(np.int64) - 1, hx)
    if getattr(a, "train_tasks", ""):  # (dscale) 只用子集训练：子集外的任务 / 配对并进留出（返回的 ht / hx 也带上，DemoBank 据此排除示范池）
        sub = json.loads(Path(a.train_tasks).read_text())
        xc = ix["xtask"][keep].astype(np.int64) - 1 if "xtask" in ix.files else np.full(int(keep.sum()), -1)
        h |= np.where(xc < 0, ~np.isin(ix["task_id"][keep], sub["tasks"]), ~np.isin(xc, sub["x"]))
        ht = ht + [t for t in range(len(tasks)) if t not in sub["tasks"] and t not in ht]
        hx = hx + sorted(k for k in set(xc[xc >= 0].tolist()) if k not in sub["x"] and k not in hx)
    return torch.from_numpy(h).repeat_interleave(data.length).to(data.device), ix, keep, ht, hx


class DemoBank:
    """(demo) 示范库常驻 GPU（已按 WM 的 norm 标准化，fp16）+ 按任务的示范池。sample(idx, split) → (特征 (B, M, P, D), keep (B, M))。
    池：src 0 = 数据根的 demo（同任务、不取 anchor 自己那条；含 xtask 的跨场景 demo），src 1 = lg 布局生成 demo（不同布局）；留出任务 / 留出配对的 demo 不进池。"""

    def __init__(s, path, data, a, ix, keep, ht, hx):
        import numpy as np
        b = torch.load(path, map_location="cpu")
        dev, s.a = data.device, a
        s.f = torch.empty(b["f"].shape, dtype=torch.float16, device=dev)
        mu, sd = data.mu.reshape(-1)[-b["f"].shape[-1]:], data.sd.reshape(-1)[-b["f"].shape[-1]:]
        for i in range(0, len(s.f), 1024):
            s.f[i:i + 1024] = ((b["f"][i:i + 1024].to(dev).float() - mu) / sd).half()
        task, src, dm, xt = b["task"].long(), b["src"].long(), b["demo"].long(), b["xtask"].long()
        ok = ~torch.from_numpy(np.isin(task.numpy(), ht)) & ~((src == 0) & torch.from_numpy(np.isin(xt.numpy() - 1, hx)))
        nt = len(data.task_names)
        key = src * nt + task
        o = torch.nonzero(ok).squeeze(1)
        o = o[torch.argsort(key[o] * 100000 + dm[o], stable=True)]  # 池内按 demo 编号排
        cnt = torch.bincount(key[o], minlength=2 * nt)
        s.order, s.cnt = o.to(dev), cnt.view(2, nt).to(dev)
        s.off = (torch.cumsum(cnt, 0) - cnt).view(2, nt).to(dev)
        rk = {(int(task[i]), int(dm[i])): j for j, i in enumerate(o.tolist()) if int(src[i]) == 0}  # 数据根 demo → 在整个 order 里的位置
        tid, dd = ix["task_id"][keep], ix["demo"][keep]
        pos = torch.tensor([rk.get((int(t), int(d)), -1) for t, d in zip(tid, dd)], dtype=torch.long)
        pos = torch.where(pos >= 0, pos - s.off[0].cpu()[torch.from_numpy(tid.astype(np.int64))], pos)  # 池内位置
        s.fpos = pos.repeat_interleave(data.length).to(dev)
        s.ftid = data.task_d.repeat_interleave(data.length).to(dev)
        s.M = s.f.shape[1]
        print(f"示范库 {path}：{tuple(b['f'].shape)}，可用 {len(o)} 条（数据根 {int(cnt[:nt].sum())} / lg {int(cnt[nt:].sum())}），"
              f"无示范的任务 {[t for t in range(nt) if cnt[t] + cnt[nt + t] == 0][:20]}", flush=True)

    def sample(s, idx, split=0):
        B, dev, a = len(idx), idx.device, s.a
        t, p = s.ftid[idx], s.fpos[idx]
        c0 = s.cnt[0, t] - (p >= 0).long()
        lg = ((torch.rand(B, device=dev) < a.demo_lg) & (s.cnt[1, t] > 0)) | (c0 <= 0)
        sr = lg.long()
        ex = ~lg & (p >= 0)
        c = torch.where(lg, s.cnt[1, t], c0).clamp_min(1)
        j = (torch.rand(B, device=dev) * c).long().clamp(max=c - 1)
        j = j + (ex & (j >= p)).long()
        bi = s.order[s.off[sr, t] + j]
        keep = torch.ones(B, s.M, dtype=torch.bool, device=dev)
        if split == 0:
            dg = torch.rand(B, device=dev) < a.p_dgoal
            keep[:, :-1] &= ~dg[:, None]
            keep &= (torch.rand(B, device=dev) >= a.p_ddrop)[:, None]
        return s.f[bi], keep


def sh_pick(t, s0, K, M, jit=0.0, grid="start", stride=4):
    """(mika) --self_hist 的选帧规则（训练与闭环共用）：t、s0 (B,) 当前帧与 episode 开头的帧号 → pos (B, M) 帧号、keep (B, M)。
    可用的过去帧 = [s0, t − K]（K 帧近期历史之外）；个数 n ≤ M 时逐帧全取（前 n 个槽），n > M 时在这一段上均匀取 M 帧（含两端）。
    grid = "recent"（--sh_grid recent，HiLo 式近窗稀疏）：紧接 K 帧近期历史之前、步长 stride 的 M 帧 t − K − stride·i（i = 0…M−1，槽按时间从旧到新），
    早于 episode 开头的槽丢弃（keep = False）；不抖动。"""
    if grid == "recent":
        jr = torch.arange(M, device=t.device)[None]
        pr = t[:, None] - K - stride * (M - 1 - jr)
        kr = pr >= s0[:, None]
        return torch.where(kr, pr, s0[:, None].expand_as(pr)), kr
    n = t - K - s0 + 1
    j = torch.arange(M, device=t.device)[None]
    step = (n[:, None] - 1).clamp_min(0).float() / max(M - 1, 1)
    if grid == "mid":  # --sh_grid mid：中点网格 s0 + ⌊(j + ½ [+ 训练期相位 / 抖动])·n/M⌋，不必取开局帧
        g = j + 0.5 + ((torch.rand(len(t), 1, device=t.device) - 0.5) + jit * (torch.rand(len(t), M, device=t.device) - 0.5) if jit > 0 else 0.0)
        uni = (s0[:, None] + torch.floor(g * n[:, None].clamp_min(1).float() / M).long()).clamp(min=s0[:, None], max=(t - K)[:, None])
    elif jit > 0:  # 训练期（--sh_jit）：网格随机相位 u ∈ [−½, ½) + 每槽抖动 ±jit/2，夹回 [s0, t − K]
        g = j + (torch.rand(len(t), 1, device=t.device) - 0.5) + jit * (torch.rand(len(t), M, device=t.device) - 0.5)
        uni = (s0[:, None] + torch.round(g * step).long()).clamp(min=s0[:, None], max=(t - K)[:, None])
    else:
        uni = s0[:, None] + torch.round(j * step).long()
    pos = torch.where((n <= M)[:, None], s0[:, None] + j, uni)
    keep = j < n[:, None]
    return torch.where(keep, pos, s0[:, None].expand_as(pos)), keep


class SelfHist:
    """(mika) --self_hist M,g：每个样本本 episode 的过去帧（sh_pick）作额外上下文 token，接口同 DemoBank：sample(idx, split) → (特征 (B, M, g², D) 已标准化,
    keep (B, M), 多久以前 (B, M) = (t − s)/1024)。池化特征库预先算好常驻 GPU（N × g² × D fp16；mk 72.5 万帧、g = 4 约 8.9 GB）。"""

    def __init__(s, data, a, dev):
        from .encode import pool
        M, g = (int(x) for x in a.self_hist.split(","))
        assert data.delta == 1, "--self_hist 只支持 --delta 1"
        N, T, D = len(data.f), data.f.shape[1], data.f.shape[2]
        k = int(round(T ** 0.5)) // g
        sdev = "cpu" if os.environ.get("PROBWM_SH_CPU") else dev  # (gawm XF5M) PROBWM_SH_CPU=1：池化历史库放 CPU（钉页内存），每步只把取到的帧搬上卡；默认常驻 GPU（原行为）
        s.f = torch.empty(N, g * g, D, dtype=torch.float16, device=sdev, pin_memory=sdev == "cpu")
        get = getattr(data, "get", None) or data.f.__getitem__
        for i in range(0, N, 8192):
            s.f[i:i + 8192] = pool(get(torch.arange(i, min(i + 8192, N), device=data.device)), k).to(sdev)
        s.data, s.M, s.K, s.p, s.jit = data, M, data.K, a.sh_drop, getattr(a, "sh_jit", 0.0)
        s.tm, s.grid, s.sd = getattr(a, "sh_time", "abs"), getattr(a, "sh_grid", "start"), getattr(a, "sh_slotdrop", 0.0)
        s.stride = getattr(a, "sh_stride", 4)  # (mika) --sh_grid recent 的步长
        s.ts = [float(x) for x in a.sh_tscale.split(",")] if getattr(a, "sh_tscale", "") else None
        s.v2 = s.tm != "abs" or s.grid != "start" or s.sd > 0 or s.ts is not None or getattr(a, "sh_alibi", False)  # 任一 v2 选项：输出第 4 项 Δt（帧数，已伸缩）
        print(f"self_hist：M {M}、每帧 {g}×{g} token，池化特征库 {tuple(s.f.shape)}（{s.f.numel() * 2 / 1e9:.1f} GB）", flush=True)

    def sample(s, idx, split=0):
        s0 = s.data.fstart[idx]
        pos, keep = sh_pick(idx, s0, s.K, s.M, s.jit if split == 0 else 0.0, s.grid, s.stride)
        if split == 0 and s.p > 0:
            keep = keep & (torch.rand(len(idx), device=idx.device) >= s.p)[:, None]
        fp = s.f[pos.cpu()].to(idx.device, non_blocking=True) if s.f.device.type == "cpu" else s.f[pos]  # (gawm XF5M) 库在 CPU 时只搬取到的帧
        if not s.v2:
            return fp, keep, (idx[:, None] - pos).float() / 1024
        if split == 0 and s.sd > 0:  # 逐槽丢弃（每个样本的丢弃率 ~ U(0, sd)）
            keep = keep & (torch.rand(keep.shape, device=idx.device) >= s.sd * torch.rand(len(idx), 1, device=idx.device))
        sc = (s.ts[0] + (s.ts[1] - s.ts[0]) * torch.rand(len(idx), 1, device=idx.device)) if (split == 0 and s.ts) else 1.0
        dt = (idx[:, None] - pos).float() * sc
        return fp, keep, sh_prog(idx, s0, pos, dt, s.tm), dt


def sh_prog(t, s0, pos, dt, tm):  # (mika) --sh_time：历史 token 时间嵌入的输入（训练与闭环共用）——abs = Δt/1024；rel = (t − s)/(t − s0)（尺度无关）
    return dt / 1024 if tm == "abs" else (t[:, None] - pos).float() / (t - s0).clamp_min(1)[:, None].float()


class MemSeq:
    """(mika C2) --mem 的训练样本：每个样本取一条 demo 里记忆步网格（episode 开头起每 r 帧）上的连续 L 个记忆步 τ_0..τ_{L−1}（最后一步的 t+1..t+H 在 demo 内），
    之前的网格点（从 episode 开头起）作无梯度预热。step(wm, a, tx, amp) → 本步的总损失（L 步的 FM 损失一起算）。历史帧早于 episode 开头时用第一帧补（同闭环）。"""

    def __init__(s, data, a, n_fut):
        s.data, s.a, s.r, s.L, s.nf = data, a, a.mem_every, a.mem_bptt, n_fut
        to = [int(x) for x in a.tgt_offs.split(",")] if getattr(a, "tgt_offs", "") else list(range(1, n_fut + 1))  # 目标时距（MSF 的远槽 16/24/32 同普通 --tgt_offs 路径）
        s.fc = getattr(a, "far_clamp", 0)  # --far_clamp：anchor 只要求这个时距以内的未来，更远的目标超出 demo 末尾时取最后一帧（同普通路径）
        an = data.anchors_h(0, min(max(to), s.fc or 10 ** 6))  # 只有近槽 1..8 时（MSNM）与原来的 anchors_h(0, n_fut) 相同
        s.to = torch.tensor(to, device=an.device)
        s0 = data.fstart[an]
        s.an = an[((an - s0) % s.r == 0) & ((an - s0) // s.r >= s.L - 1)]
        assert data.delta == 1 and len(s.an), "--mem：没有可用的样本（或 --delta ≠ 1）"
        s.offs = torch.arange(-(data.K - 1), 1, device=an.device)
        print(f"mem：S {a.mem}、每 {s.r} 帧一个记忆步、BPTT {s.L} 步、清零 {a.mem_drop}、跳写 {getattr(a, 'mem_skip', 0.0)}；可用样本 {len(s.an)}", flush=True)

    def ctx(s, t, s0):
        return s.data.get((t[:, None] + s.offs).clamp(min=s0[:, None]))

    def step(s, wm, tx, amp, B):
        from . import fm
        d, a, dev = s.data, s.a, s.an.device
        t = s.an[torch.randint(len(s.an), (B,), device=dev)]
        s0, task = d.fstart[t], d.ftask[t]
        txt, tm = tx(task, 0)
        tau0 = t - (s.L - 1) * s.r
        nb = (tau0 - s0) // s.r  # 预热步数（网格上 τ_0 之前的点）
        m0 = wm.mem0.expand(B, -1, -1).float()
        mem = m0.detach()
        with torch.no_grad(), amp():
            for j in range(int(nb.max())):
                tj = tau0 - (nb.max() - j) * s.r
                act = tj >= s0  # 这个样本的网格已进 episode
                if not act.any():
                    continue
                kv = wm.context(s.ctx(tj.clamp(min=s0), s0), txt, tm, mem=mem, mem_only=True)
                w = act & (torch.rand(B, device=dev) >= getattr(a, "mem_skip", 0.0))  # 没进 episode / 跳写：记忆保持
                mem = torch.where(w[:, None, None], kv.mem.float(), mem)
        clear = torch.rand(B, device=dev) < a.mem_drop
        kvs, x1s = [], []
        with amp():
            for k in range(s.L):
                tk = tau0 + k * s.r
                c = s.ctx(tk, s0)
                m_in = torch.where(clear[:, None, None], m0, mem)  # 清零的样本 L 步都读开头的记忆（梯度仍流到 mem0）
                kv = wm.context(c, txt, tm, torch.rand(B, device=dev) < a.p_drop, mem=m_in)
                w = torch.rand(B, device=dev) >= getattr(a, "mem_skip", 0.0)
                mem = torch.where(w[:, None, None], kv.mem.float(), m_in)
                fi = tk[:, None] + s.to
                tg = d.get(torch.minimum(fi, d.fend[tk][:, None] - 1) if s.fc else fi).flatten(1, 2)
                kvs.append(kv), x1s.append(fm.target(a.mode, tg, c[:, -1], d.rs, c[:, -2]))
            kc = type(kvs[0])((torch.cat([k_[i][0] for k_ in kvs]), torch.cat([k_[i][1] for k_ in kvs])) for i in range(len(kvs[0])))
            return kc, torch.cat(x1s)


def load_data(a, dev, norm=None):
    if getattr(a, "canon", ""):
        from .canon import Pairs
        return Pairs(a.canon.split(","), norm or torch.load(a.canon_norm, map_location="cpu")["norm"], dev, a.encoder,
                     p_id=a.canon_pid, text=Path(a.data) / f"text/{a.text}.pt" if getattr(a, "canon_wm", False) else None)
    if a.synthetic:
        return Feats.synthetic(a.K, a.delta, dev, norm=norm)
    return Feats.load(a.data, a.encoder, a.text, a.K, a.delta, dev, a.suites.split(",") if a.suites else None, norm)


def pair_clean(data, root, suites, dev):
    """(xden) --canon_slot：每帧 → 同一仿真状态的干净帧下标（data 帧序），干净帧映到自己。
    index 没有 aug 列（RoboTwin oursC，render_robotwin legacy）：demo 编号 ≥ 1000 的是 DR 复本 1000·v + d，对应同任务的干净 demo d % 1000；
    有 aug 列（LIBERO 的 vidmix merge 根，如 data_m4lg3k）：aug 行里与某条干净 demo 同 (任务, demo 编号) 的是 DR 复本（camaug 保留编号），
    对不上的（生成 / 布局 demo，编号 ≥ 1000，本身是干净渲染）映到自己。配上的须逐帧等长。"""
    import numpy as np
    root = Path(root)
    ix, tasks = np.load(root / "meta/index.npz"), json.loads((root / "meta/tasks.json").read_text())
    keep = np.isin([tasks[t]["suite"] for t in ix["task_id"]], suites.split(",")) if suites else np.ones(len(ix["demo"]), bool)  # 同 Feats.load
    tid, dm = ix["task_id"][keep], ix["demo"][keep]
    has_aug = "aug" in ix.files
    aug = ix["aug"][keep] > 0 if has_aug else dm >= 1000
    st, ln = data.start.numpy(), data.length.numpy()
    row = {(int(t), int(d)): r for r, (t, d, g) in enumerate(zip(tid, dm, aug)) if not g}
    m, n, n_id = np.arange(int(ln.sum())), 0, 0
    for r in np.nonzero(aug)[0]:
        c = row.get((int(tid[r]), int(dm[r]) if has_aug else int(dm[r]) % 1000))
        if c is None and has_aug:  # 生成 / 布局 demo：干净渲染，目标就是自己
            n_id += 1
            continue
        assert c is not None and ln[c] == ln[r], f"DR demo {(int(tid[r]), int(dm[r]))} 没有等长的干净 demo"
        m[st[r]:st[r] + ln[r]] = np.arange(st[c], st[c] + ln[c])
        n += 1
    print(f"canon_slot：{n} 条 DR demo 的目标换成同状态的干净帧（干净 demo {len(row)} 条；没有配对、映到自己的增广 demo {n_id} 条）", flush=True)
    return torch.from_numpy(m).to(dev)


def corrupt_hist(ctx, a):  # --hist_noise / --hist_drop：只在训练 batch 上污染历史上下文，削弱"照着历史外推"的捷径，让 chunk 更依赖关键帧 / 语言条件
    B = len(ctx)
    if getattr(a, "hist_noise", 0.0) > 0:  # 每个样本 τ_h ~ U(0, hist_noise)：ctx ← (1 − τ_h)·ctx + τ_h·ε（K 帧全加，含当前帧）
        t = (torch.rand(B, device=ctx.device) * a.hist_noise)[:, None, None, None]
        ctx = ((1 - t) * ctx.float() + t * torch.randn(ctx.shape, device=ctx.device)).to(ctx.dtype)
    if getattr(a, "hist_drop", 0.0) > 0:  # 较早的 K−1 帧各自以 hist_drop 的概率置 0（标准化后的均值帧），当前帧保留
        m = torch.rand(B, ctx.shape[1] - 1, device=ctx.device) < a.hist_drop
        ctx = torch.cat([torch.where(m[..., None, None], torch.zeros_like(ctx[:, :-1]), ctx[:, :-1]), ctx[:, -1:]], 1)
    return ctx


def corrupt_obs(ctx, a, black, delayed=None):  # (pro) --obs_corrupt / --t0_drop：只在训练 batch 上损坏上下文观测（标准化空间），目标仍是干净帧；black (T, D) 标准化后的黑屏特征；
    # delayed(sel, d) → (len(sel), K, T, D)：样本 sel 的上下文整体提前 d 帧（时延类用，在取 batch 的地方给，能读到上下文窗口之前的帧）
    B, K, T = ctx.shape[:3]
    dev, kk = ctx.device, torch.arange(K, device=ctx.device)
    if a.obs_corrupt > 0:
        on = torch.rand(B, device=dev) < a.obs_corrupt
        ks = getattr(a, "obs_kinds", "")
        if ks:  # (pro X3cF) 只抽这几类（均分）；没有取更早帧的入口时时延类换成冻结
            km = {"drop": 0, "freeze": 1, "black": 2, "box": 3, "patch": 4, "delay": 5}
            al = torch.tensor([km[x] for x in ks.split(",")], device=dev)
            kind = al[torch.randint(len(al), (B,), device=dev)]
            if delayed is None:
                kind = torch.where(kind == 5, torch.ones_like(kind), kind)
        else:
            kind = torch.randint(6, (B,), device=dev)  # 0 丢弃 1 冻结 2 黑屏 3 块遮挡 4 逐 patch 损坏 5 时延
            if delayed is None:  # 没有取更早帧的入口时不抽时延类（在 0–4 里重抽）
                kind = torch.where(kind == 5, torch.randint(5, (B,), device=dev), kind)
        L = torch.randint(1, K + 1, (B,), device=dev)
        s0 = (torch.rand(B, device=dev) * (K - L + 1).float()).long()
        run = on[:, None] & (kk[None] >= s0[:, None]) & (kk[None] < (s0 + L)[:, None])  # (B, K) 被损坏的帧
        g = int(round(T ** 0.5))
        hh, ww = torch.randint(4, 11, (B,), device=dev), torch.randint(4, 11, (B,), device=dev)
        y0, x0 = (torch.rand(B, device=dev) * (g - hh + 1).float()).long(), (torch.rand(B, device=dev) * (g - ww + 1).float()).long()
        yy, xx = torch.arange(T, device=dev) // g, torch.arange(T, device=dev) % g
        box = (yy[None] >= y0[:, None]) & (yy[None] < (y0 + hh)[:, None]) & (xx[None] >= x0[:, None]) & (xx[None] < (x0 + ww)[:, None])  # (B, T)
        frz = ctx[torch.arange(B, device=dev), (s0 - 1).clamp_min(0)]  # (B, T, D) 段之前的一帧
        z = torch.zeros_like(ctx)
        alt = torch.where((kind == 1)[:, None, None, None], frz[:, None].expand_as(ctx), z)
        alt = torch.where((kind == 2)[:, None, None, None], black.to(ctx.dtype)[None, None].expand_as(ctx), alt)
        alt = torch.where((kind == 3)[:, None, None, None], torch.where(box[:, None, :, None], z, ctx), alt)
        # 逐 patch：每样本比例 f ~ U(.1, .5)，各 patch 独立以 f 入选（各帧同一组 patch）；一半样本换成标准正态噪声，一半换成同集上下文里另一帧（随机、不同于本帧）的对应 patch
        f = 0.1 + 0.4 * torch.rand(B, device=dev)
        pm = torch.rand(B, T, device=dev) < f[:, None]
        oth = (kk[None] + torch.randint(1, K, (B, 1), device=dev)) % K  # (B, K) 每帧取另一帧（K = 1 时退化为本帧）
        src = torch.where((torch.rand(B, device=dev) < 0.5)[:, None, None, None], torch.randn_like(ctx, dtype=torch.float).to(ctx.dtype),
                          ctx[torch.arange(B, device=dev)[:, None], oth])
        alt = torch.where((kind == 4)[:, None, None, None], torch.where(pm[:, None, :, None], src, ctx), alt)
        sel = torch.nonzero(on & (kind == 5)).squeeze(1)
        if len(sel):  # 时延：这段帧换成整体提前 d ~ U{1..obs_delay} 帧的上下文的对应帧
            d = torch.randint(1, getattr(a, "obs_delay", 8) + 1, (len(sel),), device=dev)
            alt[sel] = delayed(sel, d).to(ctx.dtype)
        ctx = torch.where(run[..., None, None], alt, ctx)
    if a.t0_drop > 0:
        d0 = torch.rand(B, device=dev) < a.t0_drop
        ctx = torch.cat([ctx[:, :-1], torch.where(d0[:, None, None], torch.zeros_like(ctx[:, -1]), ctx[:, -1])[:, None]], 1)
    return ctx


class Step(torch.nn.Module):  # 让 DDP / compile 包住 context + 去噪前向
    def __init__(s, wm, a):
        super().__init__()
        s.wm, s.a, s.sw = wm, a, None
        s.offs = torch.tensor([float(x) for x in a.tgt_offs.split(",")]) if getattr(a, "ev_roll", False) else None  # --ev_roll：固定槽的时距
        hw = getattr(a, "hw", 0.0)
        if hw and (getattr(a, "roll", 0) > 0 or getattr(a, "slot_noise", False) or getattr(a, "sync_slots", False)):  # 时距加权（--hw）的逐槽噪声版：逐槽权重 ∝ 实际时距^hw（稀疏 --tgt_offs 用真实时距），均值 1
            k = torch.tensor(([float(x) for x in a.tgt_offs.split(",")] + [float(a.ev_slot)] * bool(getattr(a, "ev_slot", 0))) if getattr(a, "tgt_offs", "")
                             else [float(j) for j in range(1, a.n_fut + 1)]) ** hw
            s.sw = k / k.mean()
        if s.sw is not None and getattr(a, "canon_slot", False):  # (xden) --canon_slot 的 t+0 槽时距为 0：按时距 1 计权（否则权重为 0）
            k = torch.tensor([max(float(x), 1.0) for x in a.tgt_offs.split(",")]) ** hw
            s.sw = k / k.mean()
        if getattr(a, "far_w", 1.0) != 1.0:  # (cap) --far_w：远槽（时距 ≥ 9）在 --hw 权重上再乘系数，重新均值归一
            assert s.sw is not None and getattr(a, "tgt_offs", "") and not getattr(a, "ev_slot", 0) and not getattr(a, "canon_slot", False) \
                and not getattr(a, "any_far", ""), "--far_w 只接 --hw + 逐槽噪声 + 固定 --tgt_offs（不与 --ev_slot / --canon_slot / --any_far 同用）"
            of = torch.tensor([float(x) for x in a.tgt_offs.split(",")])
            k = s.sw * torch.where(of >= 9, torch.tensor(float(a.far_w)), torch.tensor(1.0))
            s.sw = k / k.mean()
        s.aej = None
        if getattr(a, "aux_ae", 0.0) > 0:  # (cap) --aux_ae：远槽（--tgt_offs 里时距 ≥ 9）的槽序号
            assert getattr(a, "tgt_offs", "") and (getattr(a, "roll", 0) > 0 or getattr(a, "slot_noise", False)) and not (
                getattr(a, "any_far", "") or getattr(a, "ev_slot", 0) or getattr(a, "far_drop", 0.0) or getattr(a, "uwm", 0) or getattr(a, "aux_eef", 0.0)
                or getattr(a, "prefetch", False) or getattr(a, "canon_slot", False) or getattr(a, "warp", "") or getattr(a, "demo_bank", "")), \
                "--aux_ae 只接 --tgt_offs + 逐槽噪声的普通路径"
            s.aej = torch.tensor([j for j, x in enumerate(a.tgt_offs.split(",")) if int(x) >= 9])
        s.fd = getattr(a, "far_drop", 0.0)
        if s.fd > 0:  # (faraux) 近槽须是 --tgt_offs 的前缀（模型按槽序取时间槽嵌入，近槽单独前向 = 只取前 n_near 个槽）
            of = [int(x) for x in getattr(a, "tgt_offs", "").split(",") if x]
            s.n_near = sum(o < 9 for o in of)
            assert 0 < s.n_near < len(of) == a.n_fut and all(o < 9 for o in of[:s.n_near]) and (getattr(a, "roll", 0) > 0 or getattr(a, "slot_noise", False)) \
                and not (getattr(a, "ev_slot", 0) or getattr(a, "any_far", "") or getattr(a, "aux_offs", "") or getattr(a, "aux_kf", False) or getattr(a, "warp", "")
                         or getattr(a, "lam_pc", 0.0) or getattr(a, "aux_eef", 0.0) or getattr(a, "lw_motion", 0.0) or getattr(a, "lw_obj", 0.0)
                         or getattr(a, "far_slots", "") or getattr(a, "canon_slot", False)), \
                "--far_drop 只接 --tgt_offs（近槽在前）+ --roll / --slot_noise 的普通路径"
        assert not getattr(a, "sync_slots", False) or (getattr(a, "tgt_offs", "") and not (getattr(a, "roll", 0) > 0 or getattr(a, "slot_noise", False) or s.fd
                                                        or getattr(a, "far_aux", False) or getattr(a, "uwm", 0) or getattr(a, "any_far", ""))), \
            "--sync_slots 需要 --tgt_offs，不与 --roll / --slot_noise / --far_drop / --far_aux / --uwm / --any_far 同用"
        s.fa = 0
        if getattr(a, "far_aux", False):  # (gawm) 严格 IFP 式远头：近槽 / 远槽的 roll_sort 剖面分开抽（fm.loss split）；单向 mask 在 model.trunk（build 设 n_main）
            of = [int(x) for x in getattr(a, "tgt_offs", "").split(",") if x]
            s.fa = sum(o < 9 for o in of)
            assert 0 < s.fa < len(of) == a.n_fut and all(o < 9 for o in of[:s.fa]) and getattr(a, "roll", 0) > 0 and getattr(a, "roll_sort", False) and s.fd == 0 \
                and not (getattr(a, "ev_slot", 0) or getattr(a, "any_far", "") or getattr(a, "aux_offs", "") or getattr(a, "aux_kf", False) or getattr(a, "warp", "")
                         or getattr(a, "far_slots", "") or getattr(a, "canon_slot", False) or getattr(a, "uwm", 0) or getattr(a, "mem", 0)), \
                "--far_aux 只接 --tgt_offs（近槽在前）+ --roll --roll_sort 的普通路径，不与 --far_drop / aux_offs / any_far / ev_slot / uwm / mem 同用"
        s.gfn = 0
        s.afv, s.ldn = None, 0
        if getattr(a, "alpha_far", 0.0) > 0 or getattr(a, "lead", 0.0) > 0:  # (rteasy 臂 L / G2) 近槽须是 --tgt_offs 的前缀；只接 forward 里那次普通 fm.loss
            of = [int(x) for x in getattr(a, "tgt_offs", "").split(",") if x]
            ng = sum(o < 9 for o in of)
            assert 0 < ng < len(of) == a.n_fut and all(o < 9 for o in of[:ng]) and getattr(a, "roll", 0) > 0 and s.fd == 0 and s.fa == 0 \
                and (getattr(a, "roll_sort", False) or not getattr(a, "alpha_far", 0.0)) \
                and not (getattr(a, "ev_slot", 0) or getattr(a, "any_far", "") or getattr(a, "uwm", 0) or getattr(a, "mem", 0) or getattr(a, "sync_slots", False)), \
                "--alpha_far / --lead 只接 --tgt_offs（近槽在前）+ --roll（--alpha_far 还要 --roll_sort）的普通路径"
            if getattr(a, "alpha_far", 0.0) > 0:
                s.afv = torch.tensor([a.alpha] * ng + [a.alpha_far] * (len(of) - ng))
            s.ldn = ng if getattr(a, "lead", 0.0) > 0 else 0
        if getattr(a, "goal_first", 0.0) > 0:  # (rteasy 臂 G) 近槽须是 --tgt_offs 的前缀；只接 forward 里那次普通 fm.loss（逐槽路径）
            of = [int(x) for x in getattr(a, "tgt_offs", "").split(",") if x]
            ng = sum(o < 9 for o in of)
            assert 0 < ng < len(of) == a.n_fut and all(o < 9 for o in of[:ng]) and getattr(a, "roll", 0) > 0 and s.fd == 0 and s.fa == 0 \
                and not (getattr(a, "ev_slot", 0) or getattr(a, "any_far", "") or getattr(a, "uwm", 0) or getattr(a, "mem", 0) or getattr(a, "sync_slots", False)), \
                "--goal_first 只接 --tgt_offs（近槽在前）+ --roll 的普通路径，不与 --far_drop / --far_aux / ev_slot / any_far / uwm / mem / sync_slots 同用"
            s.gfn = ng
        if getattr(a, "uwm", 0):  # (uwm) 联合损失（forward 里不做 import，免得 torch.compile 断图）
            from .uwm import loss as uloss
            s.uloss = uloss

    def fdrop(s, kv, x1, zt, sw):
        """(faraux) --far_drop：随机 nd = round(p·B) 个样本只带近槽前向（x1 只取前 n_near 槽；model.trunk 槽数 < n_fut 时只用前 ns 个时间槽嵌入、无 mask），
        其余样本全部槽照常。各样本的损失 = Σ_在场 token w·err / (n_fut·T)，即近槽项与全槽样本同权、远槽项去掉。"""
        B, T, n, nn_ = len(x1), s.wm.pos.shape[1], s.a.n_fut, s.n_near
        nd = int(round(s.fd * B))
        pm = torch.rand(B, device=x1.device).argsort()  # batch 里干净 / 生成 anchor 是分段拼的，随机挑
        kv, x1, zt = [(k[pm], v[pm]) for k, v in kv], x1[pm], zt[pm]  # 普通 list（本路径不用 KV 的 heat / aux_kv）
        cut = lambda q, a_, b_: [(k[a_:b_], v[a_:b_]) for k, v in q]
        lf = lambda kv_, x_, z_, ns, w_: fm.loss(s.wm, kv_, x_, s.a.mode, s.a.alpha, s.a.p_uni, s.a.sigma, s.a.lam_tc, z_, s.a.p_warm, s.a.warm_hi,
                                                 slots=ns, roll=getattr(s.a, "roll", 0.0), churn=getattr(s.a, "roll_churn", 0.0),
                                                 roll_sort=getattr(s.a, "roll_sort", False), sw=w_)
        l = 0.0
        if nd > 0:
            l = l + nd * lf(cut(kv, 0, nd), x1[:nd, :nn_ * T], zt[:nd], nn_, None if sw is None else sw[:nn_]) * (nn_ / n)
        if nd < B:
            l = l + (B - nd) * lf(cut(kv, nd, B), x1[nd:], zt[nd:], n, sw)
        return l / B

    def forward(s, ctx, txt, tmask, drop, x1, txt2=None, tm2=None, obj=None, hdrop=None, pw=None, ya=None, heat=None, kf=None, evd=None, demo=None, hz=None, ua=None, ae=None, t0=None):
        sw_, fk_ = (None if s.sw is None else s.sw.to(x1.device)), {}
        if ua is not None:  # (uwm) ua = (标准化动作 token (B, H, 28), 本体 pfeat (B, 56))：画面 + 动作联合 FM（probwm/uwm.py loss）
            assert txt2 is None and pw is None and hz is None and evd is None and ya is None and demo is None and s.fd == 0
            return s.uloss(s.wm, s.wm.context(ctx, txt, tmask, drop, obj, hdrop), x1, ua[0], ua[1], s.a, sw_, *ua[2:])[0]  # ua[2:]：(ptdata P2) 掩码、潜在动作目标
        if hz is not None:  # (hz2) --any_far：各样本目标槽的实际时距 (B, ns) → 时间槽嵌入；--hw 按实际时距逐样本加权（均值 1）
            fk_["fpos"] = s.wm.hz_pos(hz)
            if getattr(s.a, "hw", 0.0):
                k_ = hz.float() ** s.a.hw
                sw_ = k_ / k_.mean(1, keepdim=True)
        kv2 = None if txt2 is None else s.wm.context(ctx, txt2, tm2, drop, obj, hdrop, heat=heat, kf=kf, demo=demo)
        kv = s.wm.context(ctx, txt, tmask, drop, obj, hdrop, aux=ya is not None, heat=heat, kf=kf, demo=demo, **({"t0": True} if t0 is not None else {}))
        t0p = getattr(kv[0] if isinstance(kv, tuple) else kv, "t0", None)  # (pro X3cSG) t+0 小头的输出（上下文 detach 过）
        if getattr(s.a, "aux_notext", False):  # 辅助槽读丢掉语言的上下文
            (kv[0] if ya is not None else kv).aux_kv = s.wm.context(ctx, txt, tmask, torch.ones_like(drop), obj, hdrop, heat=heat, kf=kf, demo=demo)
        if ya is not None:  # 辅助：末端位移
            kv, pa = kv
            la = ((pa.float() - ya) ** 2).mean()
        if s.fd > 0:  # (faraux) 不与 txt2 / pw / hz / evd 同用（__init__ 已排除对应 flag）
            assert txt2 is None and pw is None and hz is None and evd is None and ya is None
            return s.fdrop(kv, x1, ctx[:, -1, :s.wm.pos.shape[1]], sw_)
        aef = None
        if ae is not None:  # (cap) --aux_ae：ae (B, n_far, 3·臂数) = 远槽时刻的末端位移（dm）；远槽隐状态（槽内 token 平均）→ s.wm.ae，逐槽按 (1 − τ) 加权
            j_ = s.aej.to(x1.device)

            def aef(h, ts):
                hf = h.unflatten(1, (ts.shape[1], -1))[:, j_].mean(2)
                e = ((s.wm.ae(hf).float() - ae) ** 2).mean(-1)
                return s.a.aux_ae * (e * (1 - ts[:, j_].float())).mean()
        l = fm.loss(s.wm, kv, x1, s.a.mode, s.a.alpha, s.a.p_uni, s.a.sigma,
                    s.a.lam_tc, ctx[:, -1, :s.wm.pos.shape[1]], s.a.p_warm, s.a.warm_hi, kv2, getattr(s.a, "lam_pc", 0.0), pw,
                    slots=(s.a.n_fut if hz is None else hz.shape[1]) if getattr(s.a, "slot_noise", False) or getattr(s.a, "roll", 0) > 0 or getattr(s.a, "sync_slots", False) else 0,
                    roll=getattr(s.a, "roll", 0.0), churn=getattr(s.a, "roll_churn", 0.0), roll_sort=getattr(s.a, "roll_sort", False), sw=sw_,
                    dist=None if evd is None else torch.cat([s.offs.to(x1.device)[None].expand(len(evd), -1), evd[:, None]], 1), **fk_,
                    **({"split": s.fa} if s.fa else {}), **({"det_sw": True} if getattr(s.a, "det_hw", False) else {}),
                    **({"sync": True} if getattr(s.a, "sync_slots", False) else {}),
                    **({"gf": s.a.goal_first, "gf_near": s.gfn} if s.gfn else {}),
                    **({"af": s.afv.to(x1.device)} if s.afv is not None else {}), **({"lead": s.a.lead, "lead_near": s.ldn} if s.ldn else {}),
                    **({"ae": aef} if aef is not None else {}))
        if getattr(s.a, "canon_stn", ""):  # (pro) 几何 canon 的正则：变换参数相对恒等的偏移（取自这次前向）
            l = l + s.a.stn_reg * s.wm.stn.pen()
        if getattr(s.a, "warp", ""):  # warp-kf 的正则（残差 L2、形变平滑），取自 fm.loss 里那次前向
            l = l + s.wm.reg_loss(s.a.warp_res, s.a.warp_tv)
        if t0p is not None:  # (pro X3cSG) t+0 小头：对同状态干净当前帧的 MSE；梯度只进小头
            lt = ((t0p.float() - t0[:, :t0p.shape[1]].float()) ** 2).mean()
            s.t0loss = lt.detach()
            l = l + s.a.t0_w * lt
        return l if ya is None else l + s.a.aux_eef * la


def main(argv=None):
    a = get_args(argv)
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank() if ddp else 0
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(a.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    data = load_data(a, dev, torch.load(a.norm_from, map_location="cpu")["norm"] if getattr(a, "norm_from", "") else None)
    hold = dbank = None
    if getattr(a, "hold_tasks", "") or getattr(a, "hold_x", "") or getattr(a, "demo_bank", "") or getattr(a, "train_tasks", ""):  # (demo) 留出任务 / 配对：train、val anchor 都去掉（在下面各种抽样分组之前）
        hm, *hold = hold_mask(a.data, data, a)
        for sp in (0, 1):
            data.anchors[sp] = data.anchors[sp][~hm[data.anchors[sp]]]
        if rank == 0:
            print(f"留出任务 {hold[2]}、xtask 配对 {hold[3]}：留出帧 {int(hm.sum())}，剩 anchor {len(data.anchors[0])} / {len(data.anchors[1])}", flush=True)
    if getattr(a, "demo_bank", ""):
        assert not (getattr(a, "canon", "") or getattr(a, "prior", False) or a.synthetic), "--demo_bank 不与 --canon / --prior / --synthetic 同用"
        dbank = DemoBank(a.demo_bank, data, a, *hold)
        a.demo_m, a.demo_grid = dbank.M, int(round(dbank.f.shape[2] ** 0.5))  # 存进 args：评测 / 闭环按它重建 WM
    if getattr(a, "self_hist", ""):  # (mika) 全历史上下文：走示范 token 通路（dbank 接口），WM 按 demo_m / demo_grid 建
        assert not (getattr(a, "demo_bank", "") or getattr(a, "canon", "") or getattr(a, "prior", False) or a.synthetic), "--self_hist 不与 --demo_bank / --canon / --prior / --synthetic 同用"
        dbank = SelfHist(data, a, dev)
        a.demo_m, a.demo_grid = dbank.M, int(a.self_hist.split(",")[1])
    if getattr(a, "n_demo", 0):  # 对照："只用有标注 demo 的视频"训练 WM（与 idm.labeled 同一套嵌套子集；norm 仍按全部训练帧算）
        from .idm import labeled
        data.anchors[0] = data.anchors[0][labeled(data, a.n_demo)[data.anchors[0]]]
    if getattr(a, "trainval", False):  # 每任务 50 条 demo 全部训练（val 仍单独报告，但已见过；norm 照旧只用训练帧）
        data.anchors[0] = torch.cat([data.anchors[0], data.anchors[1]])
    dims = (data.f.shape[2], data.f.shape[1], data.temb.shape[-1])
    fuz, snorm = None, data.norm
    if getattr(a, "fuse", ""):  # (pro 11-18 FU) 冻结 F_φ：上下文 / 目标在 GPU 上现算（不另存数据集）；目标是冻结 F 的输出，不会塌
        from . import fuse as fuse_
        assert not (getattr(a, "tgt_tok", 0) or getattr(a, "canon", "") or getattr(a, "canon_slot", False) or getattr(a, "canon_hist", False) or getattr(a, "t0_head", 0)
                    or dbank is not None or getattr(a, "obj", "") or getattr(a, "kfcond", False) or getattr(a, "obs_corrupt", 0.0) > 0 or getattr(a, "t0_drop", 0.0) > 0), \
            "--fuse 只接普通稀疏时距 WM（不与 --tgt_tok / canon / t+0 槽 / 示范 / 物体 token / 关键帧条件 / 观测损坏同用）"
        fck = torch.load(a.fuse, map_location="cpu")
        fck = fck.get("fuse", fck)
        Fz = fuse_.load(fck, dev)
        T0, Tw = Fz.T, data.f.shape[1] - Fz.T
        assert Tw == fck["cfg"]["Tw"] and data.mu.dim() == 2, ("多视角库每帧的腕部 token 数与 F 不符 / 不是逐视角 norm", data.f.shape, fck["cfg"])
        ini = torch.load(a.init, map_location="cpu")["norm"] if a.init else None
        assert ini is not None and torch.allclose(data.mu[:T0].cpu(), ini["mu"].expand(T0, -1)) and torch.allclose(data.sd[:T0].cpu(), ini["sd"].expand(T0, -1)), \
            "头部 token 的 norm 须与 --init（B2XT3）的相同"
        wm_, ws_ = Fz.wnorm["mu"].float(), Fz.wnorm["sd"].float()  # 腕部：WM 合并 norm → F 的 wnorm（IDM 腕部流的）
        fconv = (data.sd[T0:].float() / ws_, (data.mu[T0:].float() - wm_) / ws_)
        fuz = lambda x: fuse_.frames(Fz, x, T0, fconv)
        dims, snorm = (dims[0], T0, dims[2]), ini  # WM 只看 s（每帧 T0 个 token）；ckpt 存头部的 norm（= B2XT3 的），闭环 / canon 照旧
        a.fuse_fp = fuse_.fp(fck)
        if rank == 0:
            print(f"--fuse {a.fuse}：F {fck['cfg']}，门 mean |tanh g| {Fz.stats()[0]:.4f}；每帧 {T0} + {Tw} → {T0} token，指纹 {a.fuse_fp:.6g}", flush=True)
    amix = None
    if getattr(a, "p_main", 0.0) > 0:  # 干净 / 生成 anchor 分开抽（只有训练 batch；val 仍是干净 val）
        import numpy as np
        ix, tasks = np.load(Path(a.data) / "meta/index.npz"), json.loads((Path(a.data) / "meta/tasks.json").read_text())
        keep = np.isin([tasks[t]["suite"] for t in ix["task_id"]], a.suites.split(",")) if a.suites else np.ones(len(ix["aug"]), bool)  # 同 Feats.load
        fa = torch.from_numpy(ix["aug"][keep].astype(bool)).repeat_interleave(data.length).to(dev)
        an = data.anchors_h(0, min(max(int(x) for x in a.tgt_offs.split(",")), getattr(a, "far_clamp", 0) or 10 ** 6) if getattr(a, "tgt_offs", "") else
                            max([getattr(a, "tgt_off", 1) + getattr(a, "tgt_stride", 1) * (a.n_fut - 1)] +
                                [int(x) for x in str(getattr(a, "aux_offs", "")).split(",") if x]))  # 同 data.batch 的未来跨度（含辅助槽）
        amix = (an[~fa[an]], an[fa[an]])
        if getattr(a, "p_lg", 0.0) > 0:  # 三路固定比例：干净 p_main、生成 demo（编号 ≥ 1000，layoutgen）p_lg、其余增广（DR 复本）
            fl = torch.from_numpy(ix["demo"][keep] >= 1000).repeat_interleave(data.length).to(dev)
            amix = (an[~fa[an]], an[fa[an] & ~fl[an]], an[fl[an]])
        if getattr(a, "p_gen", 0.0) > 0:  # 新任务（task_id ≥ gen_tid）再单独一组，放最后
            fg = torch.from_numpy(ix["task_id"][keep] >= a.gen_tid).repeat_interleave(data.length).to(dev)
            amix = tuple(g[~fg[g]] for g in amix) + (an[fg[an]],)
        if rank == 0:
            print(f"p_main {a.p_main}{f'，p_lg {a.p_lg}' if getattr(a, 'p_lg', 0.0) > 0 else ''}{f'，p_gen {a.p_gen}' if getattr(a, 'p_gen', 0.0) > 0 else ''}：各组 anchor {[len(x) for x in amix]}", flush=True)
    bal = None
    if getattr(a, "bal", -1.0) >= 0:  # 四 suite 联合训练（fullbench）：suite / 任务均衡抽 anchor
        from .data import bal_sampler
        assert amix is None and not getattr(a, "canon", ""), "--bal 不与 --p_main / --canon 同用（--bal 的值就是干净帧比例）"
        Hb = min(max(int(x) for x in a.tgt_offs.split(",")), getattr(a, "far_clamp", 0) or 10 ** 6) if getattr(a, "tgt_offs", "") else \
            max([getattr(a, "tgt_off", 1) + getattr(a, "tgt_stride", 1) * (a.n_fut - 1)] + [int(x) for x in str(getattr(a, "aux_offs", "")).split(",") if x])
        bal = bal_sampler(data, data.anchors_h(0, Hb), a.bal, a.suites.split(",") if a.suites else None, **({"p_x": a.bal_x} if getattr(a, "bal_x", 0.0) > 0 else {}),
                          **({"p_ao": a.bal_ao} if getattr(a, "bal_ao", 0.0) > 0 else {}), **({"p_dr": a.bal_dr} if getattr(a, "bal_dr", 0.0) > 0 else {}))  # (xtask3) --bal_ao；(pro B3) --bal_dr
    tws = None
    if getattr(a, "task_w", ""):  # (rteasy) 按任务难度重加权：训练 anchor 按"逐帧均匀 × 所属任务权重"抽（权重表见 scripts/rteasy_tw.py；缺的任务 1）；只接稀疏时距 WM 的默认抽法
        assert amix is None and bal is None and getattr(a, "tgt_offs", "") and not hasattr(data, "gather") and not getattr(a, "canon", ""), \
            "--task_w 只接 --tgt_offs 的 WM（不与 --p_main / --bal / --canon / vidmix 同用）"
        twd = json.loads(Path(a.task_w).read_text())
        twd = twd.get("w", twd)
        tw = torch.tensor([float(twd.get(n, 1.0)) for n in data.task_names], device=dev)
        anw = data.anchors_h(0, min(max(int(x) for x in a.tgt_offs.split(",")), getattr(a, "far_clamp", 0) or 10 ** 6))  # 同 xb 默认路径的 anchor 集合
        ftw = data.task_d.to(dev).repeat_interleave(data.length.to(dev))[anw]
        cw = tw[ftw].double().cumsum(0)  # float64：几百万帧的累积和在 float32 下会丢增量
        tws = lambda B: anw[torch.searchsorted(cw, torch.rand(B, device=dev, dtype=torch.float64) * cw[-1]).clamp_max(len(anw) - 1)]
        if rank == 0:
            s0, s1 = torch.bincount(ftw, minlength=len(tw)).float(), torch.bincount(ftw, weights=tw[ftw], minlength=len(tw))
            s0, s1 = s0 / s0.sum(), s1 / s1.sum()
            print(f"--task_w {a.task_w}：任务权重 {tw.min():.2f}–{tw.max():.2f}；anchor 份额变化最大的任务 "
                  + "，".join(f"{data.task_names[i]} {100 * s0[i]:.1f}→{100 * s1[i]:.1f}%" for i in (s1 / s0.clamp_min(1e-9)).argsort(descending=True)[:5].tolist()), flush=True)
    if getattr(a, "motion_p", 0.0) > 0:  # (mika) 任务内按"之后 H 帧内在动"重分配 anchor（任务份额不变）；复用 --task_w 的抽样钩子 tws
        assert tws is None and amix is None and bal is None and getattr(a, "tgt_offs", "") and not hasattr(data, "gather") and not getattr(a, "canon", ""), \
            "--motion_p 只接 --tgt_offs 的默认抽法（不与 --task_w / --p_main / --bal / --canon / vidmix 同用）"
        import numpy as _np
        mo = torch.from_numpy(_np.load(Path(a.data) / "meta/motion.npy")).to(dev)
        assert len(mo) == len(data.fstart), f"motion.npy {len(mo)} 帧 ≠ 特征库 {len(data.fstart)} 帧"
        anw = data.anchors_h(0, min(max(int(x) for x in a.tgt_offs.split(",")), getattr(a, "far_clamp", 0) or 10 ** 6))
        ftm, mm = data.task_d.to(dev).repeat_interleave(data.length.to(dev))[anw], mo[anw]
        nt = torch.bincount(ftm, minlength=len(data.task_names)).double()
        fm_ = torch.bincount(ftm, weights=mm.double(), minlength=len(data.task_names)) / nt.clamp_min(1)
        pm = a.motion_p
        wmv = torch.where(fm_ < pm, pm / fm_.clamp_min(1e-9), torch.ones_like(fm_))
        wid = torch.where(fm_ < pm, (1 - pm) / (1 - fm_).clamp_min(1e-9), torch.ones_like(fm_))
        cwm = torch.where(mm, wmv[ftm], wid[ftm]).cumsum(0)
        tws = lambda B: anw[torch.searchsorted(cwm, torch.rand(B, device=dev, dtype=torch.float64) * cwm[-1]).clamp_max(len(anw) - 1)]
        if rank == 0:
            print(f"--motion_p {pm}：动的 anchor 份额（重分配前 → 后）" + "，".join(f"{data.task_names[i]} {100 * fm_[i]:.0f}→{100 * max(pm, fm_[i]):.0f}%"
                                                                     for i in range(len(data.task_names)) if fm_[i] < pm), flush=True)
    if getattr(a, "obj", ""):
        from . import objtok
        objtok.attach(data, a)
    if getattr(a, "lw_obj", 0.0) > 0:  # 加权用的覆盖率：每帧本任务（操作物体, 目标）两个通道
        from . import objtok
        objtok.attach_lw(data, a)
    eef = data.state("robot0_eef_pos") if getattr(a, "aux_eef", 0.0) > 0 else None
    aeE = aeJ = None
    if getattr(a, "aux_ae", 0.0) > 0:  # (cap) --aux_ae：各臂末端 xyz 拼成 (N, 3·臂数)；远槽序号
        aeE = torch.cat([data.state(k)[:, :3] for k in a.ae_keys.split(",")], 1).to(dev)
        aeJ = torch.tensor([j for j, x in enumerate(a.tgt_offs.split(",")) if int(x) >= 9], device=dev)
        assert float(aeE.abs().sum(1).eq(0).float().mean()) < 0.01, "--ae_keys 的状态里有大量全 0 行（RoboTwin 的 DR 复本要先用 scripts/cap_eefill.py 补齐）"
    wxy = kfi = kci = kpm = None
    if getattr(a, "kf_pred", ""):  # J2：关键帧 WM（冻结）
        kpk = torch.load(a.kf_pred, map_location="cpu")
        kpa = Namespace(**kpk["args"])
        kpm = build(kpa, kpk["dims"]).to(dev).eval().requires_grad_(False)
        kpm.load_state_dict(kpk["model"])
    toffs = torch.tensor([int(x) for x in a.tgt_offs.split(",")], device=dev) if getattr(a, "tgt_offs", "") else None
    cmap = t0map = None
    if getattr(a, "t0_head", 0):  # (pro X3cSG) t+0 小头的标签：当前帧 → 同一仿真状态的干净帧（同 canon_slot 的配对）；WM 主体的目标不换
        assert not getattr(a, "canon_slot", False) and not hasattr(data, "gather"), "--t0_head 不与 --canon_slot 同用（Feats 路径）"
        t0map = pair_clean(data, a.data, a.suites, dev)
    if getattr(a, "canon_slot", False):  # (xden) 规范化槽：上下文仍是本 demo（DR 复本）的帧，各目标槽（含 t+0）取同一仿真状态的干净帧
        assert toffs is not None and int(toffs[0]) == 0 and not hasattr(data, "gather") and not getattr(a, "far_slots", ""), "--canon_slot 需要 --tgt_offs 以 0 开头（Feats 路径）"
        cmap = pair_clean(data, a.data, a.suites, dev)
    chn = None
    if getattr(a, "canon_hist", False):  # (xden2) t+0 回灌：过去 K−1 帧 = 干净配对帧 (+ 残差噪声)，当前帧不动（xb 里换）
        assert cmap is not None and a.mode == "abs" and not getattr(a, "hist_noise", 0.0) and not getattr(a, "hist_drop", 0.0), "--canon_hist 需要 --canon_slot（abs），不与 --hist_noise / --hist_drop 同用"
        if a.ch_noise:
            r_ = torch.load(a.ch_noise, map_location="cpu")
            chn = (r_["sd_dr"].float().to(dev) * a.ch_scale, r_["sd_cl"].float().to(dev) * a.ch_scale, float(r_["rho"]),
                   None if r_.get("gain") is None else r_["gain"].float().to(dev))  # gain (2, D)：DR 复本 / 干净 demo 的逐通道增益（t0 输出向均值收缩）
        if rank == 0:
            print(f"canon_hist：过去 {a.K - 1} 帧换成干净配对帧，" + ("不加噪声" if chn is None else
                  f"加噪 {a.ch_noise}（×{a.ch_scale}；RMS sd DR {chn[0].pow(2).mean().sqrt():.3f} / 干净 {chn[1].pow(2).mean().sqrt():.3f}，rho {chn[2]:.3f}"
                  f"{'' if chn[3] is None else f'，gain 均值 DR {chn[3][0].mean():.3f} / 干净 {chn[3][1].mean():.3f}'}）"), flush=True)
    fslot = fsj = fsw = None
    if getattr(a, "far_slots", ""):  # (objslot) 物体级远槽的目标 slot 向量 → GPU (N, Ks, ds) fp16（按 data.src 取本数据的帧）
        import numpy as np
        assert toffs is not None and not hasattr(data, "gather"), "--far_slots 只支持 Feats 的 --tgt_offs 路径"
        fslot = torch.from_numpy(np.ascontiguousarray(np.load(a.far_slots, mmap_mode="r")[np.asarray(data.src)])).to(dev)
        fsj = [j for j, o in enumerate(toffs.tolist()) if o >= a.far_slot_min]
        Ks, T_ = fslot.shape[1], data.f.shape[1]
        fsw = torch.ones(a.n_fut, T_, device=dev)
        fsw[fsj] = a.far_pad_w
        fsw[fsj, :Ks] = T_ / Ks
        fsw = fsw.flatten()
        if rank == 0:
            print(f"物体级远槽：{a.far_slots} {tuple(fslot.shape)}，slot 帧槽 {fsj}（时距 {[toffs[j].item() for j in fsj]}）", flush=True)
    aoffs = torch.tensor([int(x) for x in a.aux_offs.split(",")], device=dev) if getattr(a, "aux_offs", "") else None
    Ha = max([a.n_fut] + (aoffs.tolist() if aoffs is not None else []))  # 辅助槽要的未来跨度
    assert toffs is None or len(toffs) + bool(getattr(a, "ev_slot", 0)) == a.n_fut, "--tgt_offs 的个数（--ev_slot 时 + 1）须等于 --n_fut"
    afar = None
    if getattr(a, "any_far", ""):  # (hz2) 任意时距远槽：近槽 = --tgt_offs 里 < 9 的；远槽时距逐样本抽（af_hz）
        import random
        lo_, hi_, k0_, k1_ = (int(x) for x in a.any_far.split(","))
        nr_ = toffs[toffs < 9] if toffs is not None else None
        assert nr_ is not None and len(nr_) + k1_ == a.n_fut and 9 <= lo_ < hi_ and 1 <= k0_ <= k1_ <= hi_ - lo_ + 1 and not (getattr(a, "ev_slot", 0) or getattr(a, "far_slots", "")
               or getattr(a, "canon_slot", False) or getattr(a, "aux_offs", "") or hasattr(data, "gather")), "--any_far：--n_fut = 近槽数 + kmax，不与 ev_slot / far_slots / canon_slot / aux_offs 同用"
        afar = (nr_, torch.arange(lo_, hi_ + 1, device=dev), 1.0 / torch.arange(lo_, hi_ + 1, device=dev).float(), k0_, k1_, random.Random(a.seed))
        if a.compile:  # 远槽个数逐步变（k0..k1）：每种形状各编一份静态图，不转动态形状
            import torch._dynamo as dyn_  # 别写 import torch._dynamo：会把 main 里的 torch 变成局部名
            dyn_.config.automatic_dynamic_shapes = False
            dyn_.config.cache_size_limit = max(dyn_.config.cache_size_limit, 4 * (k1_ - k0_ + 1))
        if rank == 0:
            print(f"any_far：近槽 {nr_.tolist()}，远槽从 [{lo_}, {hi_}] 抽 {k0_}–{k1_} 个（∝ 1/时距），时间槽表长 {hi_}", flush=True)

    def af_hz(B, split):  # (hz2) → (B, 近槽数 + k) 各样本目标槽时距（升序）；训练 batch 的 k 由各 rank 同种子的 Random 定（各 rank 形状一致），val 用 kmax
        nr_, hs_, w_, k0_, k1_, rg_ = afar
        k = rg_.randint(k0_, k1_) if split == 0 else k1_
        far = hs_[torch.multinomial(w_.expand(B, -1), k, replacement=False)].sort(1).values
        return torch.cat([nr_[None].expand(B, -1), far], 1)
    if getattr(a, "where", False) or getattr(a, "keyframe", False) or getattr(a, "kfcond", False) or getattr(a, "joint_kf", False) or getattr(a, "aux_kf", False) \
            or getattr(a, "ev_slot", 0):  # 每帧"下一个接触事件"：之后第一次夹爪指令翻转的那一帧（没有就取 demo 最后一帧）
        import numpy as np
        from .objtok import to_grid
        gr = np.load(Path(a.data) / "meta/actions.npy", mmap_mode="r")[data.src][:, 6] > 0
        nxt = np.empty(len(gr), np.int64)
        for st, ln in zip(data.start.tolist(), data.length.tolist()):
            g = gr[st:st + ln]
            ev = np.nonzero(g[1:] != g[:-1])[0] + 1
            j = np.searchsorted(ev, np.arange(ln), side="right")
            nxt[st:st + ln] = st + (np.where(j < len(ev), ev[np.minimum(j, len(ev) - 1)], ln - 1) if len(ev) else ln - 1)
        if a.keyframe and getattr(a, "keyframe_src", "gripper") == "motion":  # 无标注事件：每帧特征变化速度（patch 平均）平滑后的局部极小
            sp = torch.zeros(len(data.f), device=dev)
            for i in range(0, len(data.f) - 1, 4096):
                j = min(i + 4096, len(data.f) - 1)
                sp[i:j] = (data.f[i + 1:j + 1].float() - data.f[i:j].float()).pow(2).mean((1, 2))
            for st, ln in zip(data.start.tolist(), data.length.tolist()):
                v = sp[st:st + ln].clone()
                v[-1] = v[-2] if ln > 1 else v[-1]
                v = torch.nn.functional.avg_pool1d(v[None, None], 5, 1, 2, count_include_pad=False)[0, 0].cpu().numpy()
                mins = np.nonzero((v[1:-1] < v[:-2]) & (v[1:-1] <= v[2:]))[0] + 1
                mins = mins[(mins >= 3) & (mins < ln - 1)]  # 开头几帧静止不算
                j = np.searchsorted(mins, np.arange(ln) + 2, side="left")  # 至少 2 帧之后的下一个极小
                nxt[st:st + ln] = st + (np.where(j < len(mins), mins[np.minimum(j, max(len(mins) - 1, 0))], ln - 1) if len(mins) else ln - 1)
            if rank == 0:
                print(f"运动极小关键帧：平均时距 {float((nxt - np.arange(len(nxt))).mean()):.1f} 帧", flush=True)
        if a.keyframe:
            assert a.n_fut == 1, "--keyframe 需要 --n_fut 1"
            kfi = torch.from_numpy(nxt).to(dev)
        if getattr(a, "kfcond", False) or getattr(a, "joint_kf", False) or getattr(a, "aux_kf", False) or getattr(a, "ev_slot", 0):
            kci = torch.from_numpy(nxt).to(dev)
        if rank == 0:
            print(f"下一个接触事件平均时距 {float((nxt - np.arange(len(nxt))).mean()):.1f} 帧", flush=True)
    pairs = None
    if getattr(a, "pair_loss", 0.0) > 0:  # 配对反事实损失（ECT 式）：layoutgen --pair 的同组 demo = 同任务、同源 demo 起始状态、不同布局
        import numpy as np
        from collections import defaultdict
        assert a.keyframe, "--pair_loss 只给 --keyframe WM"
        ix, tasks = np.load(Path(a.data) / "meta/index.npz"), json.loads((Path(a.data) / "meta/tasks.json").read_text())
        keep = np.isin([tasks[t]["suite"] for t in ix["task_id"]], a.suites.split(",")) if a.suites else np.ones(len(ix["demo"]), bool)  # 同 Feats.load
        row = {(tasks[t]["task"], int(d)): k for k, (t, d) in enumerate(zip(ix["task_id"][keep], ix["demo"][keep]))}
        grp = defaultdict(list)
        for d in json.loads(Path(a.pair_json or Path(a.data) / "meta/gen3k.json").read_text())["demos"]:
            if (d["task"], d["demo"]) in row:
                grp[(d["task"], d["pair"])].append(row[(d["task"], d["demo"])])
        f0 = data.start + (a.K - 1) * data.delta  # 每条 demo 第一个有完整历史的帧：两条 demo 的历史只差布局
        pairs = torch.tensor([[int(f0[v[0]]), int(f0[v[1]])] for v in grp.values() if len(v) >= 2], device=dev)
        if rank == 0:
            print(f"配对反事实损失：{len(pairs)} 对，λ {a.pair_loss}，每步 {a.pair_bs} 对", flush=True)
    if getattr(a, "where", False):  # where 热图：下一个接触事件时的末端，投影到 agentview 正向图
        cam = torch.load(a.where_cam, map_location="cpu", weights_only=False)["cam"]
        wxy = torch.from_numpy(to_grid(data.state("robot0_eef_pos").cpu().numpy()[nxt], cam)).to(dev)
        if getattr(a, "where_file", ""):  # 阶段类别质心（与闭环 oracle 同定义）；不可见 → 网格外很远 → 热图全 0
            wf = Path(a.where_file) if Path(a.where_file).is_absolute() else Path(a.data) / a.where_file
            w3 = torch.from_numpy(np.load(wf)[data.src]).to(dev).float()
            wxy = torch.where(w3[:, 2:] > 0, w3[:, :2], torch.full_like(w3[:, :2], 1e3))
            if rank == 0:
                print(f"where_file {wf}：可见 {float((w3[:, 2] > 0).float().mean()):.3f}", flush=True)
        gg = (torch.arange(16, device=dev, dtype=torch.float) + 0.5) / 16 * 2 - 1
        gyx = torch.stack([gg[:, None].expand(16, 16).reshape(-1), gg[None].expand(16, 16).reshape(-1)], 1)  # (256, (y, x))
        heat_of = lambda xy: torch.exp(-((gyx[None, :, 1] - xy[:, None, 0]) ** 2 + (gyx[None, :, 0] - xy[:, None, 1]) ** 2)
                                       / (2 * (a.where_sigma * 2 / 16) ** 2))
        if rank == 0:
            print(f"where：{len(wxy)} 帧的下一个接触点，网格坐标均值 {wxy.mean(0).tolist()}", flush=True)
    wm = build(a, dims).to(dev)
    if a.init:
        sd = torch.load(a.init, map_location="cpu")["model"]
        if sd["fpos"].shape != wm.fpos.shape:  # 远点 / 关键帧 / 稀疏多时距 / 带辅助槽的 WM 从 chunk WM 初始化：每个目标槽取 chunk 里同一时距的槽（超出 chunk 的取最后一帧）
            o, st = getattr(a, "tgt_off", 1), getattr(a, "tgt_stride", 1)
            offs = [int(x) for x in a.tgt_offs.split(",")] if getattr(a, "tgt_offs", "") else [o + st * j for j in range(a.n_fut)]
            offs += [int(x) for x in str(getattr(a, "aux_offs", "")).split(",") if x] + [1 << 30] * int(getattr(a, "aux_kf", False) or bool(getattr(a, "ev_slot", 0)))
            n_src = sd["fpos"].shape[1] - a.K
            fp0 = sd["fpos"]  # (xden2) --init_offs 用
            sd["fpos"] = sd["fpos"][:, list(range(a.K)) + [a.K + min(k, n_src) - 1 for k in offs]]
            if getattr(a, "init_offs", False):  # (xden2) 源是稀疏时距 WM：时间槽表 = [历史 −(K−1)..0, 源 tgt_offs]，每个目标时距取表里最近的（同时距则精确对上）
                so = list(range(-(a.K - 1), 1)) + [int(x) for x in torch.load(a.init, map_location="cpu")["args"]["tgt_offs"].split(",")]
                assert len(so) == fp0.shape[1] and not (getattr(a, "aux_offs", "") or getattr(a, "aux_kf", False) or getattr(a, "ev_slot", 0)), "--init_offs：源须是不带辅助槽的 --tgt_offs WM"
                jm = [min(range(len(so)), key=lambda j: abs(so[j] - k)) for k in offs]
                sd["fpos"] = fp0[:, list(range(a.K)) + jm]
                print(f"init_offs：目标时距 {offs} ← 源时间槽 {[so[j] for j in jm]}", flush=True)
        if getattr(a, "init_skip_mismatch", False):  # (explore) 形状不同的参数（如换文本编码器后 t_in 的输入维）不加载，保持新初始化
            ws = wm.state_dict()
            sk = [k for k, v in sd.items() if k in ws and v.shape != ws[k].shape]
            sd = {k: v for k, v in sd.items() if k not in sk}
            print("init 跳过形状不同的参数:", sk, flush=True)
        miss = wm.load_state_dict(sd, strict=False)
        print("init from", a.init, "missing:", miss.missing_keys, flush=True)
        if getattr(a, "warp", "") == "pure":  # 输出头改当残差头：最后一层清零（起点 = 照搬当前帧）
            torch.nn.init.zeros_(wm.head[-1].weight), torch.nn.init.zeros_(wm.head[-1].bias)
    uA = uP = upo = uM = None
    if getattr(a, "uwm", 0):  # (uwm) 动作块（idm.load_actions，同 idmPL：每帧 2H 个路点）与本体（idmPL 的 key、帧偏移 t−ih..t）常驻 GPU；标准化统计量从 --uwm_idm 拷进 WM
        from . import idm as idm_
        from .uwm import set_stats
        ick = torch.load(a.uwm_idm, map_location="cpu")
        ia = argparse.Namespace(**ick["args"])
        assert ia.H == a.uwm and ick["n_act"] == 2 * a.uwm and ia.proprio and ia.kind == "idm" and a.mode == "abs" and toffs is not None \
            and (a.roll > 0 or a.slot_noise) and afar is None and cmap is None and dbank is None and not (a.far_slots or a.ev_slot or a.aux_offs or a.aux_kf or a.where
            or a.rope or a.p_warm or a.far_drop or a.lam_pc or a.aux_eef or a.lw_motion or a.lw_obj or a.warp or a.prefetch or a.sam or a.obj), "--uwm 只接 --tgt_offs + --roll 的普通路径"
        set_stats(wm.to(dev), {"astats": ick["astats"], "model": {k: ick["model"][k] for k in ("pmu", "psd")}})
        uA, uP, upo = idm_.load_actions(data, a, getattr(a, "uwm_h5", "") or ia.h5, 2 * a.uwm), idm_.load_prop(data, ia.proprio), idm_.poffs(ia, a.K).to(dev)
        assert uA.shape[-1] == wm.uwm.ad and uP.shape[1] * len(upo) == wm.uwm.pmu.numel(), (uA.shape, uP.shape, upo)
        if getattr(a, "uwm_mask", False):  # (ptdata P2) 逐帧真动作掩码（data 帧序）
            import numpy as np
            uM = torch.from_numpy(np.load(Path(a.data) / "meta/act_mask.npy")[data.src].astype(np.float32)).to(dev)
            if rank == 0:
                print(f"uwm_mask：有真动作的帧 {int(uM.sum())} / {len(uM)}", flush=True)
        if getattr(a, "uwm_lat", 0):
            assert [int(x) for x in a.tgt_offs.split(",")][:a.uwm] == list(range(1, a.uwm + 1)), "--uwm_lat 要求前 H 个目标槽是 t+1..t+H"
        usw = (toffs.float() ** a.hw / (toffs.float() ** a.hw).mean()) if a.hw else None  # val 用的逐槽权重（同 Step.sw）
        if rank == 0:
            print(f"uwm：动作 token {a.uwm} 个（每个 2 × {uA.shape[-1]} 维路点），本体 {ia.proprio} 偏移 {upo.tolist()}；统计量取自 {a.uwm_idm}；"
                  f"λ {a.uwm_w}、画面干净比例 {a.uwm_pic}、动作层级 shift {a.uwm_alpha} / 不 shift {a.uwm_puni}", flush=True)
    ema = copy.deepcopy(wm).requires_grad_(False)
    mseq = MemSeq(data, a, a.n_fut) if getattr(a, "mem", 0) else None  # (mika C2)
    assert mseq is None or (getattr(a, "tgt_offs", "") and not (dbank is not None or getattr(a, "canon", "") or a.accum < 1)), "--mem 只接 --tgt_offs 的普通路径，不与 --self_hist / --demo_bank / --canon 同用"
    net = Step(wm, a)
    net_sw = net.sw
    net = DDP(net, device_ids=[dev.index], **({"broadcast_buffers": False} if getattr(a, "moe_bias", 0.0) > 0 else {})) if ddp else net  # (cap MoE-T) --moe_bias：各 rank 自己累计专家负载、bias_step 里 all_reduce，偏置各 rank 同步更新，不让 DDP 用 rank 0 的 buffer 覆盖
    net = torch.compile(net) if a.compile else net
    opt = torch.optim.AdamW(wm.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd, fused=dev.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(i / a.steps, 1.0))))
    amp = (lambda: torch.autocast("cuda", torch.bfloat16)) if dev.type == "cuda" else contextlib.nullcontext
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(a), indent=1))
        print(f"params {sum(p.numel() for p in wm.parameters()) / 1e6:.1f}M  frames {len(data.f)}  "
              f"anchors {len(data.anchors[0])}/{len(data.anchors[1])}", flush=True)

    def tx(task, split):
        txt, tm = data.text(task, rand=split == 0, p_orig=getattr(a, "p_orig", 0.0))
        if split == 0 and a.txt_noise > 0:
            txt = txt.float()
            txt = txt + a.txt_noise * txt.pow(2).mean(-1, keepdim=True).sqrt() * torch.randn_like(txt) * tm[..., None]
        return txt, tm

    def ch_ctx(ctx, idx):  # (xden2) --canon_hist：过去 K−1 帧 → 同一仿真状态的干净帧；--ch_noise 时 = gain·干净 + sd·AR(1) 高斯噪声（DR 复本 / 干净 demo 各用各的 sd）
        past = data.get(cmap[idx[:, None] + data.offs[:-1]]).float()
        if chn is not None:
            sd_dr, sd_cl, rho, g = chn
            sd = torch.where((cmap[idx] != idx)[:, None, None], sd_dr, sd_cl)[:, None]
            e = torch.randn(past.shape, device=past.device)
            for j in range(1, e.shape[1]):  # 相邻帧残差相关 rho（闭环里相邻步的 t+0 输出共用缓存与上下文）
                e[:, j] = rho * e[:, j - 1] + (1 - rho ** 2) ** 0.5 * e[:, j]
            past = (past if g is None else torch.where((cmap[idx] != idx)[:, None], g[0], g[1])[:, None, None] * past) + sd * e
        return torch.cat([past.to(ctx.dtype), ctx[:, -1:]], 1)

    def xb(split):
        if getattr(a, "canon", "") and (a.K > 1 or a.n_fut > 1):  # 时序规范化器 / 规范目标 WM（canon.Pairs.seq_batch）
            ctx, tgt, task = data.seq_batch(a.bs, split, a.K, a.n_fut, 1 if a.canon_wm else 0)
            txt, tm = tx(task, split) if a.canon_wm else \
                (torch.zeros(a.bs, 1, 1, device=dev), torch.zeros(a.bs, 1, dtype=torch.bool, device=dev))
            return ctx, txt, tm, fm.target(a.mode, tgt, ctx[:, -1], data.rs), None, None, None, None, None
        if getattr(a, "canon", ""):  # 特征规范化：上下文 = z_obs（K=1），目标 = z_can，语言为空（一个 null token）
            o, c = data.batch(a.bs, split)
            return (o[:, None], torch.zeros(a.bs, 1, 1, device=dev), torch.zeros(a.bs, 1, dtype=torch.bool, device=dev),
                    fm.target(a.mode, c, o, data.rs), None, None, None, None, None)  # res：(z_can − z_obs)/rs
        idx = None
        if amix is not None and split == 0:
            nm = int(round(a.bs * a.p_main))
            nl = [int(round(a.bs * a.p_lg))] if getattr(a, "p_lg", 0.0) > 0 else []
            ng = [int(round(a.bs * a.p_gen))] if getattr(a, "p_gen", 0.0) > 0 else []
            ns = [nm, a.bs - nm - sum(nl) - sum(ng)] + nl + ng
            idx = torch.cat([g[torch.randint(len(g), (k,), device=dev)] for g, k in zip(amix, ns)])
        if bal is not None and split == 0:
            idx = bal(a.bs)
        if tws is not None and split == 0:  # (rteasy) --task_w
            idx = tws(a.bs)
        kw = {"stride": a.tgt_stride} if getattr(a, "tgt_stride", 1) != 1 else {}  # 只在需要时传：别的数据类（vidmix.Mixed）的 batch 没有 stride
        if toffs is not None and hasattr(data, "gather"):  # vidmix.HostFeats（特征在主机内存）：主 / 其余按 p_main 抽，目标按时距 gather
            ctx, tgt, task, idx = data.batch(a.bs, split, idx=idx, offs=toffs)
        elif toffs is not None:  # 任意稀疏时距：自己抽 anchor（要有 max 时距的未来），目标按时距取
            if idx is None:
                an_ = data.anchors_h(split, min(int(toffs.max()), getattr(a, "far_clamp", 0) or 10 ** 6))
                idx = an_[torch.randint(len(an_), (a.bs,), device=dev)]
            ctx, _, task, idx = data.batch(a.bs, split, idx=idx)
            if uA is not None:  # (uwm) 这批 anchor 的动作 token 与本体（开头不足的本体帧重复第一帧，同 idm.zin），训练循环经 xb.ua 交给 Step
                from .uwm import astd, pfeat as upfeat
                xb.ua = (astd(wm, uA[idx]), upfeat(wm, uP[torch.maximum(idx[:, None] + upo, data.fstart[idx][:, None])]))
            if afar is not None:  # (hz2) 各样本自己的远槽时距
                xb.hz = af_hz(len(idx), split)
                fi = idx[:, None] + xb.hz * data.delta
            else:
                fi = idx[:, None] + toffs * data.delta
            fic_ = torch.minimum(fi, data.fend[idx][:, None] - 1) if getattr(a, "far_clamp", 0) else fi
            if aeE is not None:  # (cap) --aux_ae 的标签：远槽时刻相对当前帧的末端位移（dm），超出末尾同 far_clamp
                xb.ae = 10 * (aeE[fic_[:, aeJ]] - aeE[idx][:, None])
            tgt = data.get(fic_ if cmap is None else cmap[fic_]).flatten(1, 2)
            if t0map is not None:  # (pro X3cSG) t+0 小头的标签（当前帧 idx 的干净配对帧）
                xb.t0 = data.get(t0map[idx])  # get：特征在 GPU 时同 data.f[...]；PROBWM_FEATS_CPU=1 时也能用；(xden) --canon_slot：目标换成同一状态的干净帧
            if uA is not None and (uM is not None or getattr(a, "uwm_lat", 0)):  # (ptdata P2) 真动作掩码（无真动作的样本本体置 0）+ 潜在动作目标
                from .uwm import lat_target
                am_ = None if uM is None else uM[idx]
                if am_ is not None:
                    xb.ua = (xb.ua[0], xb.ua[1] * am_[:, None])
                z_ = lat_target(torch.cat([ctx[:, -1:], tgt.unflatten(1, (a.n_fut, -1))[:, :a.uwm]], 1), a.uwm_lat) if getattr(a, "uwm_lat", 0) else None
                xb.ua = xb.ua + (am_, z_)
            if getattr(a, "canon_hist", False):  # (xden2) t+0 回灌：过去 K−1 帧换成干净配对帧（+ 残差噪声），当前帧仍是观测
                ctx = ch_ctx(ctx, idx)
            if fslot is not None:  # (objslot) 远槽目标换成 slot 帧
                fic = torch.minimum(fi, data.fend[idx][:, None] - 1) if getattr(a, "far_clamp", 0) else fi
                tg = tgt.unflatten(1, (a.n_fut, -1)).clone()
                tg[:, fsj] = 0
                tg[:, fsj, :fslot.shape[1], :fslot.shape[2]] = fslot[fic[:, fsj]].to(tg.dtype)
                tgt = tg.flatten(1, 2)
        else:
            if idx is None and aoffs is not None:  # 辅助远时距：anchor 要有足够的未来
                an_ = data.anchors_h(split, Ha)
                idx = an_[torch.randint(len(an_), (a.bs,), device=dev)]
            ctx, tgt, task, idx = data.batch(a.bs, split, idx=idx, n_fut=a.n_fut - int(getattr(a, "joint_kf", False)), off=getattr(a, "tgt_off", 1), **kw)
            if aoffs is not None:  # (A) 训练专用辅助槽：远时距帧接在 chunk 后面
                tgt = torch.cat([tgt, data.f[idx[:, None] + aoffs * data.delta].flatten(1, 2)], 1)
            if getattr(a, "aux_kf", False):  # (A) 辅助槽：下一个接触关键帧
                tgt = torch.cat([tgt, getattr(data, "get", data.f.__getitem__)(kci[idx])], 1)
            if getattr(a, "joint_kf", False):  # J3：第 0 槽 = 下一个接触关键帧，其后是 chunk
                tgt = torch.cat([getattr(data, "get", data.f.__getitem__)(kci[idx]), tgt], 1)
        if toffs is not None and getattr(a, "ev_slot", 0):  # (explore) 事件槽：下一次夹爪指令翻转那一帧，接在稀疏时距槽后面
            tgt = torch.cat([tgt, getattr(data, "get", data.f.__getitem__)(kci[idx])], 1)
            if getattr(a, "ev_roll", False):  # 事件槽的实际时距（帧），训练循环经 xb.evd 交给 Step → fm.loss(dist=...)
                xb.evd = (kci[idx] - idx).float() / data.delta
        if fuz is not None:  # (pro FU) 上下文与各目标槽：每帧 [头部; 腕部] → s（冻结 F，no_grad）
            with torch.no_grad():
                ctx, tgt = fuz(ctx), fuz(tgt.unflatten(1, (-1, dims[1] + Tw))).flatten(1, 2)
        if getattr(a, "tgt_tok", 0):  # 多视角特征库里只预测第一个视角；上下文保留全部视角
            tgt = tgt.unflatten(1, (a.n_fut, -1))[:, :, :a.tgt_tok].flatten(1, 2)
        if kfi is not None:  # 接触关键帧：目标换成下一个接触事件那一帧
            tgt = getattr(data, "get", data.f.__getitem__)(kfi[idx])  # get：特征可放 CPU（PROBWM_FEATS_CPU=1）
        txt, tm = tx(task, split)
        if getattr(a, "prior", False):
            return torch.zeros_like(ctx), txt, torch.zeros_like(tm), ctx[:, -1].float(), None, None, None, None, None
        two = tx(task, split) if split == 0 and a.lam_pc > 0 else ()
        obj = data.objtok(idx, ctx[:, -1], task, a.p_odrop if split == 0 else 0.0) if getattr(a, "obj", "") else None
        pw = ya = None
        if split == 0 and tgt.shape[1] > a.n_fut * ctx.shape[2] and getattr(a, "lam_aux", 1.0) != 1.0:  # 辅助槽的损失权重
            pw = torch.ones(tgt.shape[:2], device=dev)
            pw[:, a.n_fut * ctx.shape[2]:] = a.lam_aux
        if split == 0 and (getattr(a, "lw_motion", 0.0) > 0 or getattr(a, "lw_obj", 0.0) > 0):
            pw = torch.ones(tgt.shape[:2], device=dev)
            if a.lw_motion > 0:
                mo = (tgt.float() - ctx[:, -1, :getattr(a, "tgt_tok", 0) or ctx.shape[2]].float().repeat(1, a.n_fut, 1)).pow(2).mean(-1)  # --tgt_tok：只比预测视角
                if getattr(a, "lw_slot", False):  # (ctrl) 逐槽归一：只在每一帧内部把权重挪向变化大的 patch，各槽总权重不变（槽间配比仍由 --hw 定）
                    m3 = mo.view(len(mo), a.n_fut, -1)
                    pw = pw + a.lw_motion * (m3 / m3.mean(2, keepdim=True).clamp_min(1e-6)).clamp(max=5).flatten(1)
                else:
                    pw = pw + a.lw_motion * (mo / mo.mean(1, keepdim=True).clamp_min(1e-6)).clamp(max=5)
            if a.lw_obj > 0:
                pw = pw + a.lw_obj * data.lwcov[idx].float().div(255).amax(1).repeat(1, a.n_fut)
            pw = pw / pw.mean(1, keepdim=True) if not getattr(a, "lw_slot", False) else (lambda q: (q / q.mean(2, keepdim=True)).flatten(1))(pw.view(len(pw), a.n_fut, -1))
        if split == 0 and getattr(a, "hw", 0.0) and not (getattr(a, "slot_noise", False) or getattr(a, "roll", 0) > 0 or getattr(a, "sync_slots", False)):
            # 时距加权（同步噪声）：chunk 里越远的帧权重越大（∝ j^hw，总权重不变），只作用在 chunk 的 n_fut 个槽；逐槽噪声时不走这里
            jw = torch.arange(1, a.n_fut + 1, device=dev, dtype=torch.float) ** a.hw
            jw = (jw / jw.mean()).repeat_interleave(ctx.shape[2])
            pw = torch.ones(tgt.shape[:2], device=dev) if pw is None else pw
            pw[:, :len(jw)] = pw[:, :len(jw)] * jw
        if fsw is not None:  # (objslot) slot 帧的逐 token 权重（训练 / val 都用）
            pw = fsw[None].expand(len(tgt), -1) if pw is None else pw * fsw[None]
        if split == 0 and eef is not None:
            fut = idx[:, None] + torch.arange(1, a.n_fut + 1, device=dev) * data.delta
            ya = 10 * (eef[fut] - eef[idx][:, None]).flatten(1)  # dm
        heat = heat_of(wxy[idx]) if wxy is not None else None
        if heat is not None and split == 0 and getattr(a, "p_wdrop", 0.0) > 0:
            heat = heat * (torch.rand(len(heat), 1, device=dev) >= a.p_wdrop)
        kf = None
        if kci is not None and getattr(a, "kfcond", False):  # 关键帧条件：真实的下一个接触关键帧特征；训练时按 p_kdrop 置 0
            kf = getattr(data, "get", data.f.__getitem__)(kci[idx])
            if kpm is not None and split == 0:  # J2：一部分换成关键帧 WM 的预测（以同样的历史 + 指令）
                sel = torch.nonzero(torch.rand(len(kf), device=dev) < a.p_kpred).squeeze(1)
                if len(sel):
                    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                        kp = fm.rollout(kpm, ctx[sel], txt[sel], tm[sel], 1, kpa.mode, alpha=kpa.alpha, sigma=kpa.sigma, rs=data.rs, cfg=3.0, steps=4)[0][:, 0]
                    kf = kf.clone()
                    kf[sel] = kp.to(kf.dtype)
            kt = None
            if split == 0 and getattr(a, "kf_aug", 0.0) > 0:  # 条件增广（级联扩散式）：kf ← (1 − τ_c)·kf + τ_c·ε，τ_c ~ U(0, kf_aug)，τ_c 也给 WM
                kt = torch.rand(len(kf), device=dev) * a.kf_aug
                kf = ((1 - kt)[:, None, None] * kf.float() + kt[:, None, None] * torch.randn(kf.shape, device=dev)).to(kf.dtype)
            if split == 0 and a.p_kdrop > 0:
                dk = torch.rand(len(kf), device=dev) < a.p_kdrop
                kf = torch.where(dk[:, None, None], torch.zeros_like(kf), kf)
                kt = None if kt is None else torch.where(dk, torch.zeros_like(kt), kt)
            if getattr(a, "kf_aug", 0.0) > 0:
                kf = (kf, torch.zeros(len(kf), device=dev) if kt is None else kt)
        if ob_black is not None and split == 0:  # (pro) --obs_corrupt / --t0_drop：在取 batch 处做（时延类要读上下文窗口之前的帧）；目标已按干净帧取好（abs）
            ctx = corrupt_obs(ctx, a, ob_black, lambda sel, d: data.get(torch.maximum(idx[sel][:, None] + data.offs - d[:, None] * data.delta, data.fstart[idx[sel]][:, None])))
        if getattr(a, "hist_null", 0.0) > 0 and split == 0:  # (pro 11-17 B3) 按样本把过去 K−1 帧整体置 0，只留当前观测帧
            hn_ = torch.rand(len(ctx), device=ctx.device) < a.hist_null
            ctx = torch.cat([torch.where(hn_[:, None, None, None], torch.zeros_like(ctx[:, :-1]), ctx[:, :-1]), ctx[:, -1:]], 1)
        if getattr(a, "xtok_drop", 0.0) > 0 and split == 0:  # (rteasy) 只作上下文的额外视角（--tgt_tok 之后的 token）按样本整组置 0（标准化空间的 0 = 腕部缺省，同 DR 帧的填充）
            xd = torch.rand(len(ctx), device=ctx.device) < a.xtok_drop
            ctx = torch.cat([ctx[:, :, :a.tgt_tok], torch.where(xd[:, None, None, None], torch.zeros_like(ctx[:, :, a.tgt_tok:]), ctx[:, :, a.tgt_tok:])], 2)
        n = getattr(a, "tgt_tok", 0) or ctx.shape[2]
        if dbank is not None:  # (demo) 示范接在最后（训练循环 / val 处取出）
            return ctx, txt, tm, fm.target(a.mode, tgt, ctx[:, -1, :n], data.rs, ctx[:, -2, :n]), obj, pw, ya, heat, kf, *two, dbank.sample(idx, split)
        return ctx, txt, tm, fm.target(a.mode, tgt, ctx[:, -1, :n], data.rs, ctx[:, -2, :n]), obj, pw, ya, heat, kf, *two

    def pair_term():  # ‖(x̂1_A − x̂1_B) − (k_A − k_B)‖²：同一组 τ / 噪声下，预测关键帧的差 要等于 真实关键帧的差（惩罚"不管布局都给同一个关键帧"）
        pp = pairs[torch.randint(len(pairs), (a.pair_bs,), device=dev)].flatten()  # A0, B0, A1, B1, …
        ctx_, _, task_, _ = data.batch(len(pp), 0, idx=pp)
        txt_, tm_ = tx(task_[0::2], 0)  # 同组同任务：两边用同一条（加噪的）指令
        txt_, tm_ = txt_.repeat_interleave(2, 0), tm_.repeat_interleave(2, 0)
        x1_ = data.f[kfi[pp]].float()
        u_ = torch.rand(a.pair_bs, device=dev)
        t_ = torch.where(torch.rand(a.pair_bs, device=dev) < a.p_uni, u_, fm.shift(u_, a.alpha)).repeat_interleave(2)
        e_ = torch.randn(a.pair_bs, *x1_.shape[1:], device=dev).repeat_interleave(2, 0)
        xt_ = (1 - t_[:, None, None]) * e_ + t_[:, None, None] * x1_
        x1h = xt_ + (1 - t_[:, None, None]) * wm(xt_, t_, wm.context(ctx_, txt_, tm_)).float()
        return (((x1h[0::2] - x1h[1::2]) - (x1_[0::2] - x1_[1::2])) ** 2).mean()

    ob_black = None  # (pro) --obs_corrupt / --t0_drop：标准化后的黑屏特征 (T, D)（只开 --t0_drop 时为 0，只作开关）
    if getattr(a, "obs_corrupt", 0.0) > 0 or getattr(a, "t0_drop", 0.0) > 0:
        assert a.mode == "abs" and not getattr(a, "tgt_tok", 0) and not getattr(a, "canon_hist", False) and getattr(data, "mu", None) is not None \
            and not hasattr(data, "gather") and hasattr(data, "offs") and getattr(data, "fstart", None) is not None, \
            "--obs_corrupt / --t0_drop 只接 abs、单视角、Feats 路径（目标与上下文无关），不与 --canon_hist 同用"
        assert a.obs_corrupt <= 0 or a.obs_black, "--obs_corrupt 要 --obs_black（全黑图像的特征）"
        if a.obs_black:
            import numpy as np
            fb = torch.from_numpy(np.load(a.obs_black)).float().to(dev)
            ob_black = (fb - data.mu.float()) / data.sd.float()
            assert ob_black.shape == data.get(data.anchors[0][:1]).shape[1:], (ob_black.shape, "黑屏特征的形状与数据帧不同")
        else:
            ob_black = torch.zeros(1, device=dev)
        print(f"obs_corrupt {a.obs_corrupt}（六类，时延 ≤ {getattr(a, 'obs_delay', 8)} 帧）、t0_drop {a.t0_drop}；黑屏特征（标准化）均值 {ob_black.mean():.3f}、RMS {ob_black.pow(2).mean().sqrt():.3f}", flush=True)
    nb = lambda: xb(0)
    if getattr(a, "prefetch", False):  # 后台线程先取好下一个 batch（同一 CUDA 流，顺序不变；读盘释放 GIL）
        from concurrent.futures import ThreadPoolExecutor
        pf = ThreadPoolExecutor(1)
        fut = [pf.submit(xb, 0)]

        def nb():
            b = fut[0].result()
            fut[0] = pf.submit(xb, 0)
            return b
    t0, i0, rp = time.time(), 0, out / "resume.pt"
    assert a.accum == 1 or not (getattr(a, "sam", 0.0) > 0 or pairs is not None), "--accum 不与 --sam / 成对损失同用"
    assert not getattr(a, "ev_roll", False) or (getattr(a, "ev_slot", 0) and a.roll_sort and not getattr(a, "prefetch", False) and not getattr(a, "sam", 0.0)), \
        "--ev_roll 需要 --ev_slot、--roll_sort，不与 --prefetch / --sam 同用"
    if getattr(a, "resume_every", 0) and rp.exists():  # --resume_every：被杀后同命令重跑，从最近的续训点接着训（各 rank 读同一份）
        r = torch.load(rp, map_location="cpu")
        wm.load_state_dict(r["wm"]), ema.load_state_dict(r["ema"]), opt.load_state_dict(r["opt"]), sched.load_state_dict(r["sched"])
        i0 = r["step"] + 1
        if rank == 0:
            print(f"续训：从第 {r['step']} 步接着训（{rp}）", flush=True)
    for i in range(i0, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):  # --accum：每步 accum 个独立抽的 bs 小批，梯度取平均（有效 batch = bs × accum）
            if mseq is not None:  # (mika C2) 记忆 token：沿 demo 时间展开 L 个记忆步，一起算 FM 损失
                kc, x1m = mseq.step(wm, tx, amp, a.bs)
                with amp():
                    l = fm.loss(wm, kc, x1m, a.mode, a.alpha, a.p_uni, a.sigma, slots=a.n_fut if getattr(a, "tgt_offs", "") else 0, roll=getattr(a, "roll", 0.0),
                                churn=getattr(a, "roll_churn", 0.0), roll_sort=getattr(a, "roll_sort", False), sw=None if net_sw is None else net_sw.to(dev))
                (l / a.accum).backward()
                rng = None
                continue
            bt = nb()
            dkw = {"demo": bt[-1]} if dbank is not None else {}  # (demo)
            if uA is not None:  # (uwm)
                dkw["ua"] = xb.ua
            ctx, txt, tm, x1, obj, pw, ya, heat, kf, *two = bt[:-1] if dbank is not None else bt
            ctx = corrupt_hist(ctx, a)  # 目标 x1 已按干净历史算好（abs 模式与历史无关）；val 用干净历史
            drop = torch.rand(a.bs, device=dev) < a.p_drop
            hd = torch.rand(a.bs, device=dev) < a.p_hdrop if getattr(a, "p_hdrop", 0.0) > 0 else None
            rng = torch.cuda.get_rng_state() if getattr(a, "sam", 0.0) > 0 else None
            with amp():
                l = net(ctx, txt, tm, drop, x1, *two, obj=obj, hdrop=hd, pw=pw, ya=ya, heat=heat, kf=kf, **({"evd": xb.evd} if getattr(a, "ev_roll", False) else {}), **({"hz": xb.hz} if afar is not None else {}),
                        **({"ae": xb.ae} if aeE is not None else {}), **({"t0": xb.t0} if t0map is not None else {}), **dkw)
                if pairs is not None:
                    lp = pair_term()
                    l = l + a.pair_loss * lp
            (l / a.accum).backward()
        if rng is not None:  # SAM（Foret et al.）：权重沿梯度推到半径 ρ 内损失最大处，在那里（同一 batch、同一组 τ / 噪声）重算梯度，回到原权重后用它更新
            ps = [p for p in wm.parameters() if p.grad is not None]
            gn = torch.stack([p.grad.norm() for p in ps]).norm().clamp_min(1e-12)
            es = torch._foreach_mul([p.grad for p in ps], (a.sam / gn).item())
            with torch.no_grad():
                torch._foreach_add_(ps, es)
            opt.zero_grad(set_to_none=True)
            torch.cuda.set_rng_state(rng)
            with amp():
                net(ctx, txt, tm, drop, x1, *two, obj=obj, hdrop=hd, pw=pw, ya=ya, heat=heat, kf=kf, **dkw).backward()
            with torch.no_grad():
                torch._foreach_sub_(ps, es)
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step(), sched.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), list(wm.parameters()), 1 - a.ema)
            if getattr(a, "moe_bias", 0.0) > 0:  # (cap MoE-T) 按本步负载调各层 MoE 的选专家偏置，EMA 模型直接抄同一份偏置
                from .model import MoE
                for mw, me in zip([q for q in wm.modules() if isinstance(q, MoE)], [q for q in ema.modules() if isinstance(q, MoE)]):
                    mw.bias_step()
                    me.ebias.copy_(mw.ebias)
        if rank == 0 and i % a.log == 0:
            with torch.no_grad(), amp():
                vb = xb(1)
                vkw = {"demo": vb[-1]} if dbank is not None else {}  # (demo) val：语言 + 完整示范
                ctx, txt, tm, x1, obj, _, _, heat, kf, *_ = vb[:-1] if dbank is not None else vb
                lv = fm.loss(ema, ema.context(ctx, txt, tm, obj=obj, heat=heat, kf=kf, **vkw), x1, a.mode, a.alpha, a.p_uni, a.sigma,
                             **({"fpos": ema.hz_pos(xb.hz)} if afar is not None else {}))  # val 只看 FM 项
                if uA is not None:  # (uwm) val 的联合损失（同训练的层级抽法）：画面项 / 动作项
                    from .uwm import loss as uloss
                    _, lvi, lva = uloss(ema, ema.context(ctx, txt, tm), x1, xb.ua[0], xb.ua[1], a, usw, *xb.ua[2:])
                    print(f"{i:>7} uwm val(ema) 画面 {lvi.item():.4f} 动作 {lva.item():.4f}", flush=True)
            t0l_ = getattr(net.module if hasattr(net, "module") else net, "t0loss", None)  # (pro X3cSG) t+0 小头的训练 MSE
            print(f"{i:>7} loss {l.item():.4f}{f' pair {lp.item():.4f}' if pairs is not None else ''}{f' t0h {t0l_.item():.4f}' if t0l_ is not None else ''} val(ema) {lv.item():.4f} lr {sched.get_last_lr()[0]:.1e} "
                  f"{(time.time() - t0) / max(i, 1) * 1e3:.0f}ms/it", flush=True)
        if rank == 0 and ((i % a.save == 0 and i > 0) or i == a.steps):
            torch.save({"model": ema.state_dict(), "args": vars(a), "norm": snorm, "dims": dims}, out / "ckpt.pt")
        if rank == 0 and str(i) in a.keep.split(","):
            torch.save({"model": ema.state_dict(), "args": vars(a), "norm": snorm, "dims": dims}, out / f"ckpt_s{i}.pt")
        if rank == 0 and getattr(a, "resume_every", 0) and i > 0 and i % a.resume_every == 0 and i < a.steps:
            torch.save({"wm": wm.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": i}, out / "resume.tmp")
            os.replace(out / "resume.tmp", rp)
    if rank == 0 and rp.exists():  # 训完了，续训点不再需要
        rp.unlink()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
