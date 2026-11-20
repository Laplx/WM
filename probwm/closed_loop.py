"""LIBERO 闭环评测：benchmark 初始状态 → 空跑 10 步让物体落稳 → 最多 300 步，成功 = env.check_success()（即 step 返回的 done）。
每 2 个环境步：agentview 256² → 上下翻转 → 冻结 DINOv3（与训练相同的层平均，用 WM ckpt 的 norm 标准化）→ K 帧历史
（开头重复第一帧补齐）→ 策略：
  wm+idm：WM 采样 ẑ_{t+1..t+H}（采样器见 idm.SAMPLERS），IDM(z_{t-ih..t}, ẑ) → 2H 个动作，执行前 --exec 个；
  bc：BC 直接输出动作块；
  foresight：chunk WM 采样 ẑ_{t+1..t+8} 取其中几帧作前瞻 F，π(z_{t-3..t}, 语言, F) → 2 个动作（probwm/foresight.py；--fzero 为 F 置零的消融）。
IDM / BC 的 ckpt 带腕部相机（idm.py --wrist n）时，env 另开 robot0_eye_in_hand 相机（同样上下翻转），用同一个编码器、
ckpt 里的 wnorm 标准化，保留最近 n 帧真实腕部画面传给 IDM / BC；WM 仍然只看 agentview。旧 ckpt 行为不变。
IDM / BC 的 ckpt 带本体感觉（idm.py --proprio k1,k2）时，worker 另外回传 env.step 之后 obs 里这些 key 的拼接（与渲染时
regenerate_obs_from_state 同一帧、同一 key），保留与图像历史同长的历史（新 episode 重复第一帧）。
--obs_sync：robosuite 1.4 的 observable 每个控制周期只在周期内某个物理子步采样一次（相位随浮点累加漂移），env.step 返回的图像和
本体感觉比执行后的真实状态滞后 0–1 个环境步（实测末端位置差中位 2–3 mm、最大 11 mm，每帧位移中位 20 mm）；训练数据用的是
regenerate_obs_from_state（当前状态）。
开启后每次回传前 sim.forward() + 强制重采样所有 observable（与 regenerate_obs_from_state 逐像素一致）；默认关，旧结果不变。
--ens m：ACT 式时间集成，当前步的动作 = 之前各次决策预测的动作块里覆盖这一步的动作按 exp(−m·i) 加权平均（i=0 为最早的块），
夹爪取加权平均的符号；对 flow 头 / --H 8 的 16 个动作的块才有重叠。
--comp：组合式指令引导（probwm/compose.py；多条件 CFG：完整指令 + LLM 子短语 + 对最近训练任务的负向条件，--comp_w 给权重）。
--prim：原语级子任务条件化（probwm/prim.py）：LLM 计划 → 各 env 当前原语的文本代替指令，按本体感觉 / 抽屉检测器切换；需要带本体感觉的 IDM。
--sguide w[,nf]：空间能量引导（probwm/objtok.py make_guide）：WM 每个 Euler 步取 chunk 最后 nf 帧的去噪估计 x̂1，末端图像位置探针（--eefuv）读出末端，
沿 ‖末端 − 目标‖² 的梯度改速度（按帧归一化到 w·‖u‖）；目标 = 定位热图（--sguide_src 定位头 ckpt / oracle 分割 + bddl 目标类别）的质心，
夹爪张开时取要操作的物体，上一次夹爪指令为合时取放置目标。IDM 照常执行。
--surp：每次决策记录"惊讶" s = mean(((z_obs − ẑ)/rs)²)（ẑ = 上一次计划里对这一帧的预测，rs = WM 的逐通道帧差尺度，在 patch 和通道上平均）
和运动 m = mean(((z_obs − z_prev)/rs)²)，写进 episode 的 surp / mot（不消耗随机数，结果与不加时逐 episode 相同）。
--recover kind,N,s_hi,m_lo[,arg[,R]]：连续 N 次决策 s > s_hi，或连续 N 次 m < m_lo（卡住 / 几乎不动）时触发恢复，持续 R 次决策（默认 4），
触发的决策序号写进 trig。kind：cfg = 这些 env 重新采样计划、CFG 换成 arg（默认 5）；second = 采 arg 个候选（默认 8）按 --select 打分取第二名；
lift = 不执行计划，末端沿 +z 抬起（动作 arg，默认 0.5）；reverse = 不执行计划，把最近 2R 个动作倒序取反重放；夹爪保持当前指令。只支持 wm+idm。
n_env 个子进程各持有一个 env（同一任务内复用，换任务才重建），所有活跃 env 的推理拼成一个 batch。
--tasks 用 benchmark 的任务编号（和 tasks.json 的顺序不同，按任务名找语言嵌入）。

  export CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4
  python -m probwm.closed_loop --policy wm+idm --sampler warm --tasks 0,7,8 --n_ep 3 --n_env 9
  python -m probwm.closed_loop --policy bc --tasks 0,7,8 --n_ep 3 --n_env 9
  全量（10 任务 × 20 ep）：--tasks 0-9 --n_ep 20 --n_env 20（B20Z 上 wm+idm warm 约 3 s/ep，bc 约 1.7 s/ep）；
  分到多个进程 / 多张卡：--shard i --nshard n，再用 python -m probwm.closed_loop --merge <各 shard 的 json> 汇总
  LIBERO-Para：--para runs/goal/para/jobs_sub.json（probwm/libero_para.py 生成；每个 job 自带语言嵌入行号）
  LIBERO-Plus：--plus runs/goal/plus/jobs_full_flan-t5-large.json --out ...（probwm/libero_plus.py；要在隔离安装的 LIBERO-Plus 环境里跑，
    见 scripts/plus.sh；worker 按任务名建 env，同官方 run_libero_eval：建 env 后 seed(0)，之后每个 episode 只 reset；噪声只加在策略看到的帧上）
  特征规范化：--canon runs/canon/<名>/ckpt.pt [--canon_steps 4 --canon_tau0 0.5]（probwm/canon.py；每个新观测先映射到规范视角 / 光照）
  WM 条件后验净化：--pfilt κ,s,pool,steps,τ0[,cfg]（probwm/postfilt.py：p(z_t | 已净化历史, 指令, 观测) 的 flow 采样，先验 = WM，
    似然 = 鲁棒的低频项；第一帧用 --prior 单帧先验；写回历史 = 递归滤波，不需要配对数据）
  时序规范化器（train --canon --K 4）：--canon 的模型 K>1 时用最近 K 帧原始观测规范化当前帧（保留一份原始历史）。
  规范目标 WM（train --canon_wm）：--wm_raw --canon <规范化器>，WM 读原始（扰动）历史、预测规范视角的未来；--canon 只用于 IDM 的历史。
  专家乘积采样：--canon <规范化器> --poe w[,steps]：第 1 帧槽位的速度 = v_canon(x | y_t) + w·(v_WM(x | 已净化历史, 指令) − v_prior(x))（--prior 单帧先验），
    其余槽位按 WM 自己走；递归（结果进历史）。
  WM 条件净化：--wmpur τ0,β（每个新观测加噪回 τ0，用 WM 以上一时刻的已净化历史 + 指令为条件去噪，与观测按 β 混合后写回历史：递归时间滤波）
"""
import os

for _k, _v in [("MUJOCO_GL", "egl"), ("PYOPENGL_PLATFORM", "egl"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1")]:
    os.environ.setdefault(_k, _v)  # 必须在导入 mujoco / LIBERO 之前（子进程继承）

import argparse
import json
import multiprocessing as mp
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

NOOP = [0.0] * 6 + [-1.0]  # 空跑：不动，夹爪张开


def bench_task(suite, tid):
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[suite]().get_task(tid)


def init_states(suite, tid):  # LIBERO 的 get_task_init_states 在 torch≥2.6 下因 weights_only 报错，这里直接读
    from libero.libero import get_libero_path
    t = bench_task(suite, tid)
    return torch.load(os.path.join(get_libero_path("init_states"), t.problem_folder, t.init_states_file),
                      weights_only=False)


def worker(conn, img, n_wait, perturb="", goals=(), wrist=False, prop=(), sync=False, plus=False, seg=False, soft=False, plus_reseed=False, trace=False,
           cam_oracle=False):  # cam_oracle：(pro) LIBERO-Plus 相机维诊断，每帧另渲染一张原始相机位姿下的 agentview（见 --cam_oracle）
    from libero.libero import get_libero_path
    from .perturb import apply, parse
    from .render_libero import make_env
    from . import envlock
    pt = parse(perturb)
    soft = soft and not (set(pt) & {"camera", "light"}) and (not plus or plus_reseed)  # 相机 / 灯光扰动原地改 sim.model（+= / *=），软 reset 不重建 model 会累加；
    dirty = False  # --plus 不 reseed 时硬 reset 的固定件随 worker 跑过的集数变，软 reset 复现不了 → 这几种退回硬 reset。dirty：本集改过 sim.model（--oracle），下一集硬 reset
    env, cur = None, None
    ppid0 = os.getppid()  # 主进程没了就退出（排队等槽时也查）

    def alive():
        if os.getppid() != ppid0:
            raise SystemExit(0)
    rng0 = {}
    if soft:  # 硬 reset 的 _load_model 也消耗 np.random（且重建 Robot 对象），软 reset 不消耗 → 之后的 np.random 序列（robot.reset 的初始关节噪声、LIBERO 的摆放采样）会和硬 reset 不同。
        from robosuite.robots import robot as _rbm  # 在类上挂钩：硬 reset 走到 Robot.reset 时记下 np.random 状态，软 reset 在同一位置重放（seed(0) 后它是确定的）
        _rb0 = _rbm.Robot.reset

        def _rb_reset(self_, *a_, **k_):
            if rng0.get("mode") == "replay":
                np.random.set_state(rng0["s"])
            elif rng0.get("mode") == "capture":
                rng0["s"] = np.random.get_state()
            return _rb0(self_, *a_, **k_)
        _rbm.Robot.reset = _rb_reset
    keep = envlock.hold("libero", int(os.environ.get("PROBWM_LIBERO_MAX", 64)), alive)  # noqa: F841 本机 LIBERO env 进程总数上限（所有线共用，持有到进程退出；11-07）
    cam = (lambda o: (o["agentview_image"][::-1].copy(), o["robot0_eye_in_hand_image"][::-1].copy())) if wrist else \
        (lambda o: o["agentview_image"][::-1].copy())  # 带腕部相机时回传 (agentview, 腕部)，与渲染一样上下翻转
    if cam_oracle:  # (pro) 每帧回传 [扰动相机画面, 原始相机画面]（同一仿真状态；临时把 agentview 的位姿换回原始的渲染一次，再换回来）
        cam0, ccache = cam, {}

        def clean():
            m, cid = env.env.sim.model, env.env.sim.model.camera_name2id("agentview")
            p0, q0 = m.cam_pos[cid].copy(), m.cam_quat[cid].copy()
            m.cam_pos[cid], m.cam_quat[cid] = ccache[cur[1].split("_view_")[0]]
            env.env.sim.forward()
            x = env.env._get_observations(force_update=True)["agentview_image"][::-1].copy()
            m.cam_pos[cid], m.cam_quat[cid] = p0, q0
            env.env.sim.forward()
            return x
        cam = lambda o: np.stack([cam0(o), clean()])
    view = (lambda o: (cam(o), np.concatenate([o[k] for k in prop]).astype(np.float32))) if prop else cam  # + 本体感觉
    if goals:  # --para：每次 step 回复额外带末端位置和"各任务目标是否成立"的位掩码（libero_goal 各场景相同，任何目标都能判断）
        from libero.libero.envs.bddl_utils import robosuite_parse_problem
        gs = [robosuite_parse_problem(f)["goal_state"] for f in goals]
        diag = lambda obs: (obs["robot0_eef_pos"].round(3).tolist(),
                            sum(1 << i for i, g in enumerate(gs) if all([env.env._eval_predicate(s) for s in g])))
    if seg:  # 物体 token 的 oracle 掩码（probwm/objtok.py）：回传 (观测, 当前状态 agentview 分割的 16×16 类别覆盖率 uint8 (7,256))
        from .objtok import geom_table, seg_cov
        view0, tab = view, [None]
        view = lambda o: (view0(o), seg_cov(env.env.sim, tab[0], img))
    # --trace：每次回话另带 (末端 xyz, 夹爪 qpos, 各物体 / 固定件 body 的 xyz)；reset 的回话还带名字表（scale 线 PRO Obj 失败归因）
    tinfo = (lambda obs, nm=False: (obs["robot0_eef_pos"].astype(np.float32), obs["robot0_gripper_qpos"].astype(np.float32),
                                    env.env.sim.data.body_xpos[list(env.env.obj_body_id.values())].astype(np.float32).copy())
             + ((list(env.env.obj_body_id),) if nm else ())) if trace else None
    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == "reset":
                key, state, *orc = arg  # orc：--oracle 的起点（专家生成时的固定件 body_pos / body_quat + 空跑后的状态 S0）
                if cur != key:  # 换任务才重建 env
                    bs_ = envlock.slot(alive).acquire()  # 建 / 关 env 在本机全局排队（几百个 EGL 上下文同时建 / 关会堵死驱动锁，11-07；每集的 hard reset 是分散的，不排队）
                    if env is not None:
                        env.close()
                    if not plus:
                        t = bench_task(*key)
                        env = make_env(os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file), img, wrist)
                    else:  # LIBERO-Plus：任务名就是 bddl 名（_view_…_initstate_… 后缀由 ControlEnv 解析）；同官方 get_libero_env，不额外 reset
                        from libero.libero.envs import OffScreenRenderEnv
                        env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), key[0], key[1] + ".bddl"),
                                                 camera_heights=img, camera_widths=img,
                                                 camera_names=["agentview", "robot0_eye_in_hand"] if wrist else ["agentview"])
                        env.seed(0)  # 官方 get_libero_env：seed(0)（= np.random.seed(0)）；每任务 1 个 episode 时紧接着 reset →
                    cur = key        # 固定件摆放（±1 cm 抖动）、噪声随机数与官方相同，各策略配对
                    if cam_oracle and key[1].split("_view_")[0] not in ccache:  # (pro) 原始相机位姿：同一基础任务把 _view_ 参数换成 0_0_100_0_0 建一次 env 读出来
                        import re as _re
                        from libero.libero.envs import OffScreenRenderEnv
                        e0 = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), key[0], _re.sub(
                            r"_view_\d+_\d+_\d+_\d+_\d+_", "_view_0_0_100_0_0_", key[1]) + ".bddl"), camera_heights=img, camera_widths=img, camera_names=["agentview"])
                        m0 = e0.env.sim.model
                        ccache[key[1].split("_view_")[0]] = (m0.cam_pos[m0.camera_name2id("agentview")].copy(), m0.cam_quat[m0.camera_name2id("agentview")].copy())
                        e0.close()
                    if soft:  # --soft_reset：本任务第一次 reset 仍是硬的（记下 robot.reset 处的 np.random 状态），之后不重建 MjSim / 渲染上下文
                        rng0["mode"] = "capture"
                    bs_.release()
                if not plus or plus_reseed:  # LIBERO 每次 reset 用全局 np.random 重摆固定件（柜子 / 灶台 ±1 cm，simstate 不含）；同任务复用 env 时
                    env.seed(0)  # 结果取决于该 worker 之前跑过几集 → 每次 reset 前重设，所有 episode、所有策略的固定件都相同（11-06）
                if soft and dirty:
                    env.env.hard_reset, rng0["mode"], dirty = True, "capture", False
                env.reset()
                if soft:
                    env.env.hard_reset, rng0["mode"] = False, "replay"  # 本任务第一次是硬 reset（记 rng0），之后都软、在 robot.reset 处重放
                    for rb in env.env.robots:  # 夹爪的 current_action 是逐步累加的（format_action），robot.reset 不清零；硬 reset 新建夹爪对象时为 0。
                        rb.gripper.current_action = np.zeros(rb.gripper.dof)  # 不清的话上一集以闭合结束时，本集开头几步夹爪指令不同（11-07 查出）
                obs = env.set_init_state(state)
                if seg:  # LIBERO 的 reset 会重建 MjSim
                    tab[0] = geom_table(env.env.sim)
                if pt:  # OOD 扰动（probwm/perturb.py），按任务 + 初始状态定种子
                    apply(env, pt, key, state)
                for i in range(n_wait):  # LIBERO-Plus：中间步用 robosuite 原始 step（不加噪声），只有回传的那帧加
                    obs, _, _, _ = (env.step if not plus or i == n_wait - 1 else env.env.step)(NOOP)
                if orc and orc[0] is not None:  # 换成专家的场景：固定件同专家生成时，状态回到专家起点（plandiag.restore），再取一次观测
                    from .plandiag import restore
                    bp, bq, S0 = orc[0]
                    env.env.sim.model.body_pos[:], env.env.sim.model.body_quat[:] = bp, bq
                    dirty = True
                    restore(env, S0)
                    obs = env.env._get_observations(force_update=True)
                if sync and n_wait:
                    env.env.sim.forward()
                    obs = env.env._get_observations(force_update=True)
                conn.send((view(obs), False, 0) + ((tinfo(obs, True),) if trace else ()))
            elif cmd == "step":
                n, done = 0, False
                for i, act in enumerate(arg):
                    obs, _, done, _ = (env.step if not plus or i == len(arg) - 1 else env.env.step)(act)  # done = _check_success()
                    n += 1
                    if done:
                        break
                if sync:  # 当前状态的观测（同渲染数据），而不是控制周期开头采的
                    env.env.sim.forward()
                    obs = env.env._get_observations(force_update=True)
                conn.send((view(obs), bool(done), n) + (diag(obs) if goals else ()) + ((tinfo(obs),) if trace else ()))
            else:
                break
    except Exception as e:
        conn.send(e)
        raise
    if env is not None:
        with envlock.slot():
            env.close()


def wm_t0(fm, wm, c1, txt, tm, mode, ns, cfg=1.0, **kw):  # (pro 11-17 B3) --wm_canon_fast：第 1 遍只采前 ns 个目标槽（t+0 = 第 0 槽）、步数由 kw["steps"] 定；
    # 上下文与 CFG 同 fm.rollout（不带 obj / heat / kf 等额外条件）；返回 t+0 槽 (B, Tm, D)
    assert mode != "cv" and 1 <= ns <= wm.n_fut, ("--wm_canon_fast 不接 cv 模式的 WM；ns 取 1..n_fut", mode, ns, wm.n_fut)
    null = torch.ones(len(c1), dtype=torch.bool, device=c1.device)
    kv = wm.context(c1, txt, tm)
    kvn = wm.context(c1, txt, tm, drop=null) if cfg != 1.0 else None
    Tm = wm.pos.shape[1]
    z, _ = fm.sample(wm, kv, c1[:, -1, :Tm].repeat(1, ns, 1), mode, kv_null=kvn, cfg=cfg, **kw)
    return z[:, :Tm]


class Policy:
    def __init__(s, a, dev):
        from . import fm, idm
        from .encode import ENCODERS
        from .train import build
        s.a, s.dev, s.idm, s.fm = a, dev, idm, fm
        ck = torch.load(a.wm, map_location="cpu")
        s.wa = wa = Namespace(**ck["args"])
        if getattr(a, "wm_data", ""):  # (gawm X2F5CU) 训练数据根在别的机器 / 已删（如 9029 的 /dev/shm）：改读本机同内容的数据根（info / 文本表 / meta）；默认关
            wa.data = a.wm_data
        s.mu, s.sd, s.rs = (ck["norm"][k].to(dev) for k in ("mu", "sd", "rs"))
        root = Path(wa.data)
        info = json.loads((root / f"feats/{wa.encoder}/info.json").read_text())
        if "cat" in info:  # (rteasy 11-16) 通道拼接的特征库（data.CatChan）：各子编码器按各自 info 建，输出按通道拼接；带短窗（clip）的吃整个窗，其余吃当前帧
            subs = []
            for c_ in info["cat"]:
                si = json.loads((root / f"feats/{c_}/info.json").read_text())
                hf_, ld_ = ENCODERS[si.get("base", si.get("encoder", c_))]
                subs.append((ld_(hf_, si["layers"], si["img"], dev), bool(si.get("clip"))))
            s.enc = lambda x: torch.cat([f_(x if v_ or x.dim() == 4 else x[:, -1]) for f_, v_ in subs], -1)
        else:
            hf, load = ENCODERS[info.get("base", wa.encoder)]
            s.enc = load(hf, info["layers"], info["img"], dev, **({"fuse": info["fuse"]} if info.get("fuse") else {}),
                         **({"grid_to": info["sspool"]} if info.get("sspool") else {}))  # (gawm T1) 多层融合的特征库；(gawm 方案 C) 超采样后池化的特征库
        s.graph(s, "enc")  # --graphs：编码器按输入形状捕获 CUDA graph（probwm/cgraph.py）
        s.clip = info.get("clip")  # 视频编码器（V-JEPA 2.1）的因果短窗：帧偏移（10 Hz 数据帧），每个 env 缓存最近几帧原始画面
        if s.clip:
            assert a.exec == 2, "视频编码器的短窗按 10 Hz 数据帧（每次决策执行 2 个环境步）"
        s.proj = None
        if "proj" in info:  # PCA 降维特征（encode.py --pca）：编码后先做同一个线性映射，再用 WM 的 norm 标准化
            pj = torch.load(root / f"feats/{wa.encoder}" / info["proj"], map_location=dev)
            s.proj = (pj["mu"], pj["W"])
        s.ad = None
        if "adapter" in info:  # 表征适配器（probwm/adapter.py）：编码后先过 A，再用 WM 的 norm 标准化（与 adapter apply 同一函数）
            from .adapter import load as load_ad
            s.ad = load_ad(root / f"feats/{wa.encoder}" / info["adapter"], dev)
        t = torch.load(a.text or root / f"text/{wa.text}.pt")
        s.temb, s.tmask = t["emb"].to(dev), t["mask"].to(dev)
        if s.temb.dim() == 4:  # 训练用的增广表：闭环用第 0 条（原指令）
            s.temb, s.tmask = s.temb[:, 0], s.tmask[:, 0]
        s.names = [x["task"] for x in json.loads((root / "meta/tasks.json").read_text())]
        if a.policy == "foresight":  # BC + WM 前瞻：F 用 foresight.fsample 采样（与训练缓存同一函数）
            from . import foresight
            fck = torch.load(a.fs, map_location="cpu")
            s.net, s.na, s.st = foresight.make(fck, dev)
            s.fsample, s.use_f = foresight.fsample, bool(s.na.fo) and not a.fzero
            assert not s.use_f or Path(a.wm) == Path(fck["wm"]), f"--wm 应为训练 F 用的 {fck['wm']}"
            print(f"foresight: fo {s.na.fo}，训练 F={s.na.src}（{s.na.sampler}）；闭环 F="
                  f"{'零' if not s.use_f else f'{a.sampler} cfg {a.cfg:g}'}", flush=True)
        else:
            ick = torch.load(a.bc if a.policy == "bc" else a.idm, map_location="cpu")
            s.net, s.na, s.st = idm.make(ick, dev)
            if getattr(s.net, "flow", False):
                s.graph(s.net, "vel")  # flow 头的速度场（Euler 的噪声在外面照常抽）
        s.nw = 0 if a.policy == "foresight" else s.na.wrist  # 腕部相机帧数（0 = 只用 agentview）
        s.npf = len(idm.poffs(s.na, wa.K)) if a.policy != "foresight" and s.na.proprio else 0  # 本体感觉帧数
        if s.npf:
            assert a.plan == 1, "本体感觉只有真实帧，不支持 --plan > 1"
        if a.policy != "foresight" and a.fsteps:
            s.net.steps = a.fsteps
        s.enets = []  # (rteasy) --idm_ens：另几个 IDM（同配方不同训练种子），各自采样后与主 IDM 的动作块（标准化空间）取均值；默认空 = 原行为
        for pth in filter(None, getattr(a, "idm_ens", "").split(",")):
            assert a.policy == "wm+idm", "--idm_ens 只接 wm+idm"
            en, ea, est = idm.make(torch.load(pth, map_location="cpu"), dev)
            assert all((getattr(ea, k, None) or None) == (getattr(s.na, k, None) or None) for k in ("kind", "H", "ih", "wrist", "proprio", "head", "lang", "far", "far_extra", "vpp", "tau_in")), \
                f"--idm_ens {pth} 与主 IDM 的输入配置不同"  # 空值一律当作没设（新 ckpt 的 args 多了 vpp="" / tau_in="" 等默认项，旧 ckpt 没有这些键）
            assert all(torch.allclose(est[k].float().cpu(), s.st[k].float().cpu()) for k in s.st if torch.is_tensor(s.st[k])), f"--idm_ens {pth} 的动作统计量与主 IDM 不同"
            if getattr(en, "flow", False):
                s.graph(en, "vel")
            if a.fsteps:
                en.steps = a.fsteps
            s.enets.append(en)
        if getattr(a, "idm_graph", False):  # (rteasy 11-18) --idm_graph：IDM 骨干（到 flow 头的 kv，idm.Net.ctx）按输入形状捕获 CUDA graph，forward 换成 Net.infer（采样同原推理分支）；默认关
            from .cgraph import wrap as gwrap
            if not hasattr(s, "gpool"):
                s.gpool = torch.cuda.graph_pool_handle()
            for n_ in [s.net] + s.enets:
                assert getattr(n_, "flow", False), "--idm_graph 只接 flow 头 IDM"
                n_.forward = n_.infer
                s.accel(n_, "idm", "ctx")  # --compile：骨干先编译再捕获（默认关 = 不动）
                gwrap(n_, "ctx", s.gpool)
        s.ms = None  # (rteasy 11-19) --idm_ms：集成成员各占一条 CUDA stream 与主 IDM 并发（issue 顺序同顺序版 → 噪声消耗相同、逐位一致）；默认关
        if getattr(a, "idm_ms", False) and s.enets:
            from .cgraph import Graphed
            assert not getattr(a, "idm_ens_par", False), "--idm_ms 与 --idm_ens_par 二选一"
            for n_ in s.enets:  # 并发重放的图不能共用显存池（共用池只保证顺序重放安全），也不能共用捕获流（cuBLAS workspace 按 (handle, stream) 分配，
                p_, cs_ = torch.cuda.graph_pool_handle(), torch.cuda.Stream(dev)  # 同一捕获流捕获的图共用同一份 workspace，B=1 的小矩阵乘走 split-K 会互相覆盖）：
                for k_ in ("vel", "ctx"):  # 每个成员的 vel / ctx 图各开一个池、一条捕获流；主 IDM 留在当前流、用原来的池与默认捕获流
                    g_ = getattr(n_, k_, None)
                    if isinstance(g_, Graphed):
                        assert not g_.g, "--idm_ms：成员的图须在换池之前没有捕获过"
                        g_.pool, g_.cs = p_, cs_
            s.ms = [torch.cuda.Stream(dev) for _ in s.enets]
        s.ens_par = None  # (rteasy 11-18) --idm_ens_par：主 IDM 与集成成员（同结构）参数叠起来，torch.func.vmap(functional_call) 一次前向算全部（各成员噪声独立，同顺序版）；默认关
        if getattr(a, "idm_ens_par", False) and s.enets:
            import copy
            from torch.func import functional_call, stack_module_state, vmap
            pr_, bf_ = stack_module_state([s.net] + s.enets)  # 叠的是原模块的参数（--graphs 只换了实例上的 vel，参数不变）
            base_ = copy.deepcopy(idm.make(ick, "cpu")[0]).to("meta")  # 不带 CUDA graph 的同结构模板（参数由 functional_call 换进来）
            base_.steps = s.net.steps

            def ens_par(*x, **k):  # → (M, B, n_act, ad)，M = 1 + 成员数，顺序同 [主 IDM] + enets
                base_.fixed_eps = getattr(s.net, "fixed_eps", None)
                return vmap(lambda p_, b_: functional_call(base_, (p_, b_), x, k), randomness="different")(pr_, bf_)
            s.ens_par = ens_par
        s.fu = None
        if a.policy == "wm+idm" and "fuse" in ick:  # (pro 11-18 FU) IDM 带腕部融合模块 F_φ：encode 在 canon 之后把当前帧 [头部; 腕部] 融成 s，WM 上下文与 IDM 历史都读 s
            from . import fuse as fuse_
            s.fu = fuse_.load(ick["fuse"], dev)
            assert abs(float(wa.__dict__.get("fuse_fp", 0) or 0) - fuse_.fp(ick["fuse"])) < 1e-6 * max(1.0, fuse_.fp(ick["fuse"])), \
                "WM（train --fuse）与 IDM 的融合模块不是同一个"
            assert all(torch.equal(s.fu.wnorm[k], ick["wnorm"][k].to(dev)) for k in ("mu", "sd")) and not getattr(a, "idm_ens", ""), "F 的腕部 norm 须同 IDM 的腕部流；暂不接 --idm_ens"
        if s.nw:
            s.wmu, s.wsd = (ick["wnorm"][k].to(dev) for k in ("mu", "sd"))
            s.wpk = int(round((256 / (ick["args"].get("wT", 0) or 256)) ** 0.5))  # (pro) 腕部流是池化过的（idm --wkeys wp2 → wT 64）时闭环同样 k×k 池化；默认 1 = 原行为
            s.wT = int(ick["args"].get("wT", 0) or 0)  # (pro) 腕部流的 token 数（whist 用；旧 ckpt 没有 = 与 WM 的 T 相同）
            assert a.plan == 1, "腕部相机只有真实帧，不支持 --plan > 1"
            # --canon / --poe 只作用在 agentview 上（WM 和 IDM 的场景视角），腕部画面原样交给 IDM（2026-11-02）
            assert not a.wmpur and not a.pfilt, "--wmpur / --pfilt 只支持不带腕部相机的策略"
            assert s.ad is None, "适配器特征只支持不带腕部相机的策略"
        if a.policy == "wm+idm" or (a.policy == "foresight" and s.use_f):
            s.wm = build(wa, ck["dims"]).to(dev).eval()
            s.wm.load_state_dict(ck["model"])
            if getattr(wa, "any_far", ""):  # (hz2) 任意时距远槽 WM（train --any_far）：远槽时距 = --any_far（默认 ckpt 的 --tgt_offs 远槽），时间槽嵌入按时距查表
                near = [int(x) for x in wa.tgt_offs.split(",") if int(x) < 9]
                q = near + ([int(x) for x in a.any_far.split(",")] if getattr(a, "any_far", "") else [int(x) for x in wa.tgt_offs.split(",") if int(x) >= 9])
                lo, hi = (int(x) for x in wa.any_far.split(",")[:2])
                assert all(lo <= h <= hi for h in q[len(near):]), f"--any_far 的时距须在训练范围 [{lo}, {hi}] 内：{q}"
                wa.tgt_offs, s.wm.n_fut = ",".join(map(str, q)), len(q)
                s.wm.fq = s.wm.hz_pos(torch.tensor(q, device=dev)[None]).detach()
                print(f"any_far WM：查询时距 {q}", flush=True)
            if getattr(a, "near_only", False):  # (faraux) 只用近槽推理：带远槽训练的 WM（train --far_drop）只取 --tgt_offs 里时距 < 9 的前缀槽 → 与 MSN 同一条推理路径
                of = [int(x) for x in wa.tgt_offs.split(",")]  # （model.trunk 槽数 = n_fut 时只用前 n_fut 个时间槽嵌入；远槽不参与任何一次去噪）
                near = [o for o in of if o < 9]
                assert near and of[:len(near)] == near and not getattr(wa, "any_far", "") and not getattr(wa, "ev_slot", 0), "--near_only 需要近槽在前的 --tgt_offs WM"
                wa.tgt_offs, s.wm.n_fut = ",".join(map(str, near)), len(near)
                print(f"near_only：只用近槽 {near}（训练时 --far_drop {getattr(wa, 'far_drop', 0.0)}）", flush=True)
            s.graph(s.wm, "context", "forward")
        s.ewm = []  # (rteasy) --wm_ens：另几个 WM（同特征 / norm / 时距槽 / 文本），各自滚动、各自维护缓存，喂给 IDM 的计划按槽平均；默认空 = 原行为
        for pth in [x for x in (getattr(a, "wm_ens", "") or "").split(",") if x]:
            ek = torch.load(pth, map_location="cpu")
            ea = Namespace(**ek["args"])
            assert all(torch.allclose(ek["norm"][k].to(dev), v) for k, v in (("mu", s.mu), ("sd", s.sd), ("rs", s.rs))), f"--wm_ens {pth} 的 norm 与主 WM 不同"
            assert all(getattr(ea, k, None) == getattr(wa, k, None) for k in ("tgt_offs", "mode", "K", "encoder", "text", "alpha", "n_fut")), f"--wm_ens {pth} 的配置与主 WM 不同"
            em = build(ea, ek["dims"]).to(dev).eval()
            em.load_state_dict(ek["model"])
            s.graph(em, "context", "forward")
            s.ewm.append([em, ea, None, None])  # [模型, args, 滚动缓存 rw, rv]
        s.K, s.dims, s.prev = wa.K, (ck["dims"][1], ck["dims"][0]), None
        s.xw = False  # (pro) LIBERO 闭环的多视角 WM（agentview + 腕部 8×8 上下文）由 main() 打开（robotwin_eval 自己拼双腕，不走这里）
        if a.policy == "wm+idm" and idm.fextra(s.na):  # IDM --far_extra：WM 的目标槽须正好是 t+1..t+H 加这些远时距（MS-far：1..8,16,24,32）
            assert [int(x) for x in getattr(wa, "tgt_offs", "").split(",") if x] == list(range(1, s.na.H + 1)) + idm.fextra(s.na), "IDM --far_extra 与 WM 的 tgt_offs 不一致"
        s.ev_c = None  # train --ev_roll 的事件槽状态（Policy.rolling 里按需建）
        s.mem_in = s.mem_out = None  # (mika C2) train --mem 的 WM：评测器每次决策前设上一个记忆步的记忆；只接 MS-far --roll 路径
        assert not getattr(wa, "mem", 0) or (a.roll and getattr(wa, "tgt_offs", "") and not getattr(a, "idm_tau", "") and not getattr(a, "replan", "")), \
            "--mem 的 WM 只接 --roll 的 MS-far msroll 路径（不与 --idm_tau / --replan 同用）"
        s.sh_demo = None  # (mika) train --self_hist 的 WM：每次决策前由评测器（probwm/mikasa_eval）设成 (特征, keep, 多久以前)，只接 MS-far --roll 路径
        assert not getattr(wa, "self_hist", "") or ((a.roll or (getattr(a, "det_cfg", False) and wa.mode == "det") or getattr(a, "sh_sync", False)) and getattr(wa, "tgt_offs", "") and not getattr(a, "idm_tau", "") and not getattr(a, "prog", "") and not getattr(a, "abs_far", False)), \
            "--self_hist 的 WM 只接 --roll 的 MS-far msroll 路径（mika MSFH-DET：--mode det 的 WM 配 --det_cfg 走同步路径；MSFH-sync：flow WM 配 --sh_sync 走同步路径；历史 token 都经 fm.rollout demo= 传入）"
        s.cslot, s.chist, s._pend = bool(getattr(wa, "canon_slot", False)), None, None  # (xden) train --canon_slot 的 WM：t+0 槽 = 当前帧的规范化估计（见 act）；--idm_tau pre 的待做 WM 步
        assert not s.cslot or (a.roll and getattr(wa, "tgt_offs", "").split(",")[0] == "0" and not a.wm_raw), "--canon_slot 的 WM 只接 --roll 的 MS-far 路径（tgt_offs 以 0 开头），不与 --wm_raw 同用"
        assert not getattr(a, "idm_tau", "") or (a.roll and getattr(wa, "tgt_offs", "") and not s.cslot and (getattr(s.na, "tau_in", "") or a.idm_tau == "stale")), \
            "--idm_tau pre 需要 --roll 的 MS-far WM 和 idm --tau_in 训的噪声感知 IDM（stale 只要 MS-far WM）"
        s.chh, s.stl = bool(getattr(wa, "canon_hist", False)), getattr(a, "idm_stale", "")  # (xden2) train --canon_hist 的 WM（历史 = 之前各步 t+0 输出，见 ch_hist）；--idm_stale
        assert not s.chh or s.cslot, "--canon_hist 的 WM 须同时是 --canon_slot"
        assert not getattr(a, "t0_ignore", False) or (s.cslot and not s.chh and not getattr(a, "idm_stale", "")), "--t0_ignore 只接 --canon_slot（非 --canon_hist）的 WM，不与 --idm_stale 同用"
        assert not getattr(a, "wm_canon", False) or (s.cslot and not s.chh and getattr(a, "t0_ignore", False) and a.roll and not a.canon and not getattr(a, "idm_stale", "")), \
            "--wm_canon 只接 --canon_slot（非 --canon_hist）的 WM、配 --t0_ignore（IDM 读规范化历史）与 --roll，不与独立 --canon / --idm_stale 同用"
        assert not (getattr(a, "wm_canon_fast", "") or getattr(a, "wm_canon_crn", False) or getattr(a, "wm_canon_cfg", -1.0) >= 0) or getattr(a, "wm_canon", False), \
            "--wm_canon_fast / --wm_canon_crn / --wm_canon_cfg 须配 --wm_canon"
        assert not s.stl or (a.roll and getattr(wa, "tgt_offs", "") and not getattr(a, "idm_tau", "") and not (getattr(a, "prog", "") or getattr(a, "abs_far", False)
                             or getattr(a, "replan", "") or getattr(a, "cfg_far", -1.0) >= 0 or getattr(a, "far_sep", False) or getattr(wa, "ev_slot", 0))), \
            "--idm_stale 只接 --roll 的 MS-far msroll 路径，不与 --idm_tau / --replan / --cfg_far / --far_sep 同用"  # (gawm X2F5CU) self_hist 的 WM 现在也可：历史 token 经 rolling_pre / roll_commit 传入
        s.vpp = getattr(s.na, "vpp", "") if a.policy == "wm+idm" else ""  # (vpp) idm --vpp 训的 IDM：另读 WM 在 msroll 那一步前向里的近槽隐状态（见 rolling_vpp）
        if s.vpp:
            assert a.roll and getattr(wa, "tgt_offs", "") and a.plan == 1 and not (getattr(a, "idm_tau", "") or getattr(s, "stl", "") or s.cslot or getattr(a, "replan", "") or getattr(a, "prog", "")
                                                                                     or getattr(a, "abs_far", False) or getattr(a, "cfg_far", -1.0) >= 0 or getattr(a, "far_sep", False)
                                                                                     or getattr(wa, "self_hist", "") or getattr(wa, "ev_slot", 0) or getattr(wa, "any_far", "")
                                                                                     or getattr(a, "near_only", False) or a.kf_wm), \
                "--vpp 的 IDM 只接 --roll 的 MS-far msroll 路径（--plan 1；不与 --idm_tau / --idm_stale / canon_slot / --replan / --prog / --abs_far / --cfg_far / --far_sep / 两级 / any_far / near_only 同用）"
            fp = idm.vpp_fp(s.wm)
            assert abs(fp - s.na.vpp_fp) <= 1e-4 * max(1.0, abs(fp)), f"--vpp 的 IDM 绑定训练它的 WM（{s.na.wm}，指纹 {s.na.vpp_fp:.6g}），这个 WM 的指纹 {fp:.6g}：换 WM 要重训"
            s.vbuf = s.vhid = None
            s.graph(s.wm, "trunk")  # --graphs：forward 拆成（捕获的）trunk + eager 的输出头，隐状态从 trunk 的输出取

            def fwd(x, tau, kvs, tau0=None, fpos=None):  # 与 WM.forward 同一串计算（head(trunk(·))）；s.vbuf 不为 None 时记下 trunk 的输出
                hh = s.wm.trunk(x, tau, kvs, tau0, fpos)
                if s.vbuf is not None:
                    s.vbuf.append(hh)
                return s.wm.head(hh)
            s.wm.forward = fwd
        s.rw = s.pp = s.kfp = None  # --roll 的窗口、--pcons 的上一次计划、两级的上一次关键帧（各 env 一份）
        s.p3 = None
        s.ptr = None  # --trace：同一个世界坐标末端探针，读计划各槽（含远槽）与当前帧
        if getattr(a, "trace", ""):
            from .objtok import load_eefuv
            s.ptr = load_eefuv(a.wp_probe, dev)
        if getattr(a, "wp_exec", ""):  # --wp_exec：世界坐标末端探针（objtok eefuv out=3），把 WM 计划的 t+1 槽解成末端航点
            from .objtok import load_eefuv
            s.p3 = load_eefuv(a.wp_probe, dev)
        assert not a.roll or (a.policy == "wm+idm" and a.plan == 1 and a.n_cand == 1 and not a.sampler.startswith("rtc") and not a.stagger
                              and getattr(wa, "mode", "abs") == "abs" and getattr(wa, "tgt_off", 1) == 1 and getattr(wa, "tgt_stride", 1) == 1), \
            "--roll 只接 chunk WM / 稀疏时距 WM（--tgt_offs）的普通 wm+idm 路径（--plan 1）"
        s.wact = ""  # (uwm) --wm_act：见 uwm_init（robotwin_eval 在建好 Policy 后设参数再调）
        if getattr(a, "wm_act", ""):
            s.uwm_init()
        # 规范化需要原始历史（时序规范化器）或 WM 读原始历史（--wm_raw）时，规范化挪到主循环里做
        s.rawh = bool(a.canon) and (a.wm_raw or a.poe or torch.load(a.canon, map_location="cpu")["args"]["K"] > 1)
        assert not a.wm_raw or a.canon, "--wm_raw 需要 --canon（给 IDM 的历史用）"
        # (pro 11-17) 视频编码器可配腕部相机（s.nw）：两个视角各自缓存最近几帧、各自取 (t−1, t) 帧对（见 encode），同 camaug --wrist 的编码
        s.dl = getattr(wa, "delta", 1)  # WM 帧间隔（train --delta；V-JEPA 槽位 WM 为 2）：历史 / 本体感觉按 10 Hz 缓存，每 dl 帧取一个
        assert s.dl == 1 or (a.exec == 2 and a.plan == 1 and not (s.nw or a.wmpur or a.pfilt)), "--delta > 1 只支持每 10 Hz 帧决策一次"
        s.hcfg = tuple(float(x) for x in a.hcfg.split(",")) if a.hcfg else None  # 历史 / 文本两路分开的引导（fm.rollout hcfg；WM 需 train --p_hdrop）
        assert not s.hcfg or getattr(wa, "p_hdrop", 0) > 0, "--hcfg 需要用 --p_hdrop 训练过的 WM"
        s.sg = None
        if getattr(wa, "where", False) and not a.sguide:  # where WM：借 --sguide 0 的目标机制出热图
            a.sguide, a.sguide_src = "0", a.where_src
        if a.sguide:  # 空间能量引导：末端图像位置探针 + 定位热图的目标
            from . import objtok as ot
            v = a.sguide.split(",")
            s.sg, s.ot = (float(v[0]), int(v[1]) if len(v) > 1 else 4, float(v[2]) if len(v) > 2 else 1.0, len(v) > 3 and v[3] == "jac"), ot
            s.uvp = ot.load_eefuv(a.eefuv, dev)
            s.sgh = None if a.sguide_src == "oracle" else ot.load_head(a.sguide_src, dev)
            assert a.policy == "wm+idm" and (a.n_cand == 1 or s.sg[0] == 0) and not a.sampler.startswith("rtc"), \
                "--sguide 只接普通 wm+idm 采样路径（--n_cand > 1 时只用来给 --select eef 提供目标，w 须为 0）"
        s.ag = None
        if a.aux_guide:  # 辅助头引导：WM 的辅助头预测 chunk 内末端位移 → 世界坐标探针的目标（objtok.make_guide_xyz）
            from . import objtok as ot
            assert getattr(wa, "aux_eef", 0) > 0 and a.policy == "wm+idm" and a.n_cand == 1 and not a.sampler.startswith("rtc"), "--aux_guide 需要 train --aux_eef 的 WM、普通采样路径"
            v = a.aux_guide.split(",")
            s.ag, s.ot, s.p3 = (float(v[0]), int(v[1]) if len(v) > 1 else 4, float(v[2]) if len(v) > 2 else 1.0, len(v) > 3 and v[3] == "jac"), ot, ot.load_eefuv(a.eefxyz, dev)
        s.kwm = None
        if a.kf_wm:  # 两级：关键帧 WM（train --keyframe）出"去哪儿"，主 WM（train --kfcond）以它为条件出近处的 chunk
            from .train import build as _b
            kck = torch.load(a.kf_wm, map_location="cpu")
            s.kwa = Namespace(**kck["args"])
            assert getattr(wa, "kfcond", False) and getattr(s.kwa, "keyframe", False), "--kf_wm 需要 --kfcond 的主 WM 与 --keyframe 的关键帧 WM"
            s.kwm = _b(s.kwa, kck["dims"]).to(dev).eval()
            s.kwm.load_state_dict(kck["model"])
            s.krs = kck["norm"]["rs"].to(dev)
        s.kg = None  # 只引导关键帧（--kf_guide / --kf_select）：目标同 --sguide（定位头热图质心，按夹爪开合取操作物体 / 放置目标），chunk WM 与 IDM 不变
        if a.kf_guide or a.kf_select > 1:
            assert s.kwm is not None and s.sg is not None and s.sg[0] == 0, "--kf_guide / --kf_select 需要 --kf_wm 和 --sguide 0（只借它的目标与探针）"
            v = a.kf_guide.split(",") if a.kf_guide else ["0"]
            s.kg = (float(v[0]), float(v[1]) if len(v) > 1 else 1.0, len(v) > 2 and v[2] == "jac")
        s.nobj = 0
        if getattr(wa, "obj", ""):  # 物体 token WM（train --obj）：--obj_src oracle = env 分割掩码，否则 = 掩码头 ckpt（objtok.MaskHead）
            from . import objtok
            s.objtok, s.nobj = objtok, 2 if wa.obj_mode == "ground" else len(objtok.CATS)
            assert a.policy == "wm+idm" and a.n_cand == 1 and not a.sampler.startswith("rtc") and not (getattr(a, "comp", "") or a.smooth or a.poe or a.wmpur or a.pfilt), \
                "物体 token 只接普通 wm+idm 采样路径"
            s.mhead = None if a.obj_src == "oracle" else objtok.load_head(a.obj_src, dev)
        s.comp = None
        if a.comp:  # 组合式指令引导：只接 chunk WM 的普通采样路径
            from .compose import Comp
            assert a.policy == "wm+idm" and a.n_cand == 1 and not a.sampler.startswith("rtc") and not a.smooth
            s.comp = Comp(a.comp, a.comp_w, dev)
        s.prim = None
        if a.prim:  # 原语计划：WM 的文本表换成原语表，ti 在主循环里换成当前原语的行
            from .prim import Prim
            assert a.policy == "wm+idm" and s.npf and not a.comp, "--prim 需要带本体感觉的 IDM（末端高度），不与 --comp 同用"
            s.prim = Prim(a.prim, a.prim_det, dev, a.n_env)
            s.temb0, s.tmask0 = s.temb, s.tmask  # 原指令表：两级（--kf_wm）时 chunk WM 仍用原指令，原语只进关键帧 WM
            s.temb, s.tmask = s.prim.emb, s.prim.mask
        r = a.recover.split(",") if a.recover else []  # kind, N, s_hi, m_lo, arg, R
        s.rv = (r[0], int(r[1]), float(r[2]), float(r[3]), float(r[4]) if len(r) > 4 else {"cfg": 5, "second": 8, "lift": 0.5}.get(r[0], 0),
                int(r[5]) if len(r) > 5 else 4) if r else None
        assert not (a.surp or r) or (a.policy == "wm+idm" and s.dl == 1 and a.n_cand == 1 and not a.sampler.startswith("rtc") and a.ens is None), \
            "--surp / --recover 只支持普通 wm+idm 路径"
        assert not r or r[0] in ("cfg", "second", "lift", "reverse"), r
        s.rp, s.rp_cd, s.rp_last = None, None, None  # (hz2) --replan：(阈值, 模式, 冷却决策数, 统计量下标)、各 env 的冷却计数、最近一次检查的 (统计量 (B,3), 有估计, 触发)
        if getattr(a, "replan", ""):
            v = a.replan.split(",")
            s.rp = (float(v[0]), v[1] if len(v) > 1 else "all", int(v[2]) if len(v) > 2 else 8, {"mean": 0, "top": 1, "cos": 2}[v[3] if len(v) > 3 else "mean"])
            assert a.roll and getattr(wa, "tgt_offs", "") and not (a.prog or a.abs_far or getattr(a, "idm_tau", "")) and s.rp[1] in ("all", "far"), \
                "--replan 只接 --roll 的 MS-far msroll 路径（模式 all / far）"
        s.ov, s.ov_st, s.ov_n, s.ov_tr, s.ov_last = None, None, None, None, None  # (pro) --ov：(模式, 规则, thr_hi, thr_lo, 封顶)、R3 滞回状态、连续不可信次数、最近可信观测、最近一次的 (统计量, 有估计, 不可信, 已替换)
        s.ov_res = []  # (pro) --ov softlog：各决策逐 patch 残差（标定 OV-soft 的 θ、s 用）
        if getattr(a, "ov", ""):
            v = a.ov.split(",")
            if v[0] in ("soft", "softlog"):  # (pro) OV-soft（预注册 logs/pro.md 11-15 16:50）：soft,θ,s 逐 patch 软门控；softlog 只算不混、存残差
                s.ov = (v[0], "", float(v[1]) if len(v) > 1 else 0.0, float(v[2]) if len(v) > 2 else 1.0, 0)
            else:
                s.ov = (v[0], v[1] if len(v) > 1 else "12", float(v[2]) if len(v) > 2 else float("inf"), float(v[3]) if len(v) > 3 else float("inf"), int(v[4]) if len(v) > 4 else 30)
                s.ov_r4 = (float(v[5]), float(v[6])) if "4" in s.ov[1] else None  # (pro PC2) R4 视觉-本体一致性：(τ_z, τ_q)，--ov wm,34,θ_hi,θ_lo,cap,τ_z,τ_q
                assert "4" not in s.ov[1] or len(v) > 6, "--ov 规则 4 要给 τ_z、τ_q（第 6、7 项）"
            r3o = s.ov[0] == "wm" and set(s.ov[1]) <= set("34")  # (pro 11-16 预测-校正) 只用 R3（WM 残差）时不靠 Pro2.0 的旗标：非 --pro2 的评测配 --ov_log 也能开，可配 --canon（比 WM 看到的那份特征）
            assert s.ov[0] in ("wm", "hold", "soft", "softlog") and set(s.ov[1]) <= set("1234") and (getattr(a, "pro2", "") or (r3o and getattr(a, "ov_log", False))) \
                and not getattr(a, "replan", "") and (not a.canon or r3o) and not s.nw, \
                "--ov 模式 wm | hold | soft | softlog、规则取自 1 / 2 / 3 / 4；只接 --pro2（只用 R3 / R4 的 wm 模式另可配 --ov_log 用于非 Pro2.0 评测、可配 --canon），不与 --replan / 腕部同用"
            assert s.ov[0] == "hold" or (a.roll and getattr(wa, "tgt_offs", "") and not (a.prog or a.abs_far or getattr(a, "idm_tau", ""))), "--ov wm / soft 只接 --roll 的 MS-far msroll 路径"

    @torch.no_grad()
    def encode(s, imgs, slots=None, fresh=None):  # uint8 (B,256,256,3) → (B,T,D) 标准化 fp16（与 Feats 一致）；带腕部相机时每项为 (agentview, 腕部) → 两组
        if getattr(s, "xw", False):  # (pro) 多视角 WM：agentview 照常（含 canon），腕部 k×k 池化、按 WM 的 norm 标准化后拼在后面（--wm_xtok_null 时腕部全 0 = 训练里的缺省）；
            # IDM 也读腕部（s.nw，如 IW8）时另返回 IDM 的腕部流（按 IDM 的 wT 池化、wnorm 标准化）
            from .encode import pool as pool_
            assert s.clip is None and s.proj is None and s.ad is None and not s.rawh, "多视角 WM 的腕部上下文不接视频编码器 / PCA / 适配器 / 时序规范化"
            x = torch.from_numpy(np.stack([q[0] for q in imgs] + [q[1] for q in imgs])).to(s.dev)
            with torch.autocast("cuda", torch.bfloat16):
                fa, fw = s.enc(x).float().chunk(2)
            T0 = fa.shape[1]
            mu, sd = (s.mu, s.sd) if s.mu.dim() == 1 else (s.mu[:T0], s.sd[:T0])
            mw, sw_ = (s.mu, s.sd) if s.mu.dim() == 1 else (s.mu[T0:], s.sd[T0:])
            z = ((fa - mu) / sd).half()
            z = s.canonize(z, (mu, sd) if s.mu.dim() == 1 else (mu[0], sd[0])) if s.a.canon else z  # 逐视角 norm 时 agentview 各 token 的统计量相同，取第一行给规范化器核对
            w = torch.zeros(len(imgs), s.dims[0] - T0, z.shape[2], dtype=z.dtype, device=s.dev) if getattr(s.a, "wm_xtok_null", False) \
                else ((pool_(fw, s.xwk).float() - mw) / sw_).half()
            if s.nw:
                wi = pool_(fw, s.wpk) if getattr(s, "wpk", 1) > 1 else fw
                return torch.cat([z, w.to(z.dtype)], 1), ((wi.float() - s.wmu) / s.wsd).half()
            return torch.cat([z, w.to(z.dtype)], 1)
        if s.nw:
            imgs = [x[0] for x in imgs] + [x[1] for x in imgs]  # 两个相机拼成一个 batch 过编码器，各用各的 norm
        x = torch.from_numpy(np.stack(imgs)).to(s.dev)
        if s.clip:  # 视频编码器：slots (B,) env 下标，fresh (B,) 新 episode → 用第一帧补满缓存（同 encode.py 里 demo 开头的补法）；带腕部时两个视角各一份缓存
            L, nv = 1 - s.clip[0], 2 if s.nw else 1
            if not hasattr(s, "fbuf"):
                s.fbuf = torch.zeros(nv, s.a.n_env, L, *x.shape[1:], dtype=x.dtype, device=s.dev)
            xs = []
            for v, xv in enumerate(x.chunk(nv)):
                b = torch.cat([s.fbuf[v, slots, 1:], xv[:, None]], 1)
                s.fbuf[v, slots] = b = torch.where(fresh.view(-1, 1, 1, 1, 1), xv[:, None].expand_as(b), b)
                xs.append(b[:, [L - 1 + o for o in s.clip]])
            x = torch.cat(xs)
        with torch.autocast("cuda", torch.bfloat16):
            f = s.enc(x).float()
        if s.proj is not None:
            f = (f - s.proj[0]) @ s.proj[1]
        if not s.nw:
            if s.ad is not None and s.a.canon and not s.rawh:  # 适配器 + 规范化：先在 DINO 空间规范化（规范化器按 DINO 的 norm 训练），再过适配器
                zd = s.canonize(((f - s.ad.mu) / s.ad.sd).half(), (s.ad.mu, s.ad.sd)).float()
                return ((s.ad(zd).float() - s.mu) / s.sd).half()
            z = (((f if s.ad is None else s.ad.raw(f)) - s.mu) / s.sd).half()
            z = s.purify(z) if s.a.purify else z
            return s.canonize(z) if s.a.canon and not s.rawh else z
        f, w = f.chunk(2)
        if getattr(s, "wpk", 1) > 1:  # (pro) IDM 的腕部流是 k×k 池化的特征库
            from .encode import pool as pool_
            w = pool_(w, s.wpk)
        z = ((f - s.mu) / s.sd).half()
        z, wn = (s.canonize(z) if s.a.canon and not s.rawh else z), ((w - s.wmu) / s.wsd).half()
        if getattr(s, "fu", None) is not None:  # (pro FU) s = F(规范化后的头部, 腕部)：之后 WM / IDM 读到的当前帧都是 s
            z = s.fu(z, wn)
        return z, wn

    @torch.no_grad()
    def purify(s, z):
        """观测净化（用户方案用在输入上）：当前帧加噪回 τ0 → 在 τ0 层沿单帧先验 p(z_t) 的 score 做 K 次 Langevin → 去噪回 τ=1。
        把受扰动（相机 / 光照）的观测拉回训练分布，再交给 WM 和 IDM。"""
        s.load_prior()
        t0, K, st, snr = (float(x) for x in s.a.purify.split(","))
        B = len(z)
        with torch.autocast("cuda", torch.bfloat16):
            kv = s.prior.context(torch.zeros(B, s.pa.K, *z.shape[1:], device=s.dev), torch.zeros(B, s.pL, s.pdt, device=s.dev),
                                 torch.zeros(B, s.pL, dtype=torch.bool, device=s.dev))
            x = s.fm.sample(s.prior, kv, z.float(), "abs", steps=int(st), alpha=s.pa.alpha, tau0=t0, n_lan=int(K), snr=snr)[0]
        return x.half()

    @torch.no_grad()
    def canonize(s, z, norm=None):  # norm：规范化器应有的 (mu, sd)，默认是 WM 的
        """特征规范化（probwm/canon.py）：受扰动观测的特征 → 规范相机 / 光照下同一状态的特征，再交给只见过干净视角的 WM 和 IDM。"""
        from .canon import canonize, load
        if not hasattr(s, "cm"):
            s.cm, s.ca, cn = load(s.a.canon, s.dev)
            s.graph(s.cm, "context", "forward")
            assert all(torch.equal(cn[k].to(s.dev), x) for k, x in zip(("mu", "sd"), norm or (s.mu, s.sd))), "规范化模型的 norm 与 WM 不一致"
        return canonize(s.cm, s.ca, z, s.a.canon_steps, s.a.canon_tau0).half()

    @torch.no_grad()
    def wmpur(s, hp, h, ti, fresh, slots=None, tb=None):
        """WM 条件净化（递归时间滤波）：hp = 上一时刻的历史 z_{t-K..t-1}（已净化），h = 当前历史（最后一帧是新观测 z_t）。
        把 z_t 复制到整个 chunk、加噪回 τ0，用 WM 以 hp + 指令（CFG 同主采样）为条件去噪到 τ=1，取第 1 帧 ẑ_t，
        返回最后一帧换成 β·ẑ_t + (1-β)·z_t 的历史；新 episode 没有上一时刻，不净化。
        --wmpur τ0,β,τ0s,n_lan,n_start,anchor：开头 n_start 帧改用更强的噪声 τ0s + n_lan 步 Langevin（snr .15）再放松到 τ0；
        anchor=1 时新 episode 的第一帧换成同一任务训练 demo 第一帧（规范视角）里与观测最像的一帧（tb = 基础任务下标）。"""
        v = [float(x) for x in s.a.wmpur.split(",")]
        t0, beta = v[0], v[1]
        t0s, nl, ns, anc = (v[2], int(v[3]), int(v[4]), int(v[5])) if len(v) > 5 else (t0, 0, 0, 0)
        h = h.clone()
        if ns:
            if not hasattr(s, "fc"):
                s.fc = torch.zeros(s.a.n_env, dtype=torch.long, device=h.device)
            s.fc[slots] = torch.where(fresh, torch.zeros_like(s.fc[slots]), s.fc[slots] + 1)
        if anc and fresh.any():
            h[fresh] = s.anchor(h[fresh, -1], tb[fresh])[:, None].expand(-1, h.shape[1], -1, -1)
        T = h.shape[2]
        for m_, tau, n_lan in ((~fresh & (s.fc[slots] > ns) if ns else ~fresh, t0, 0),
                               (~fresh & (s.fc[slots] <= ns) if ns else torch.zeros_like(fresh), t0s, nl)):
            if not m_.any():
                continue
            hq, txt, tm = hp[m_], s.temb[ti[m_]], s.tmask[ti[m_]]
            with torch.autocast("cuda", torch.bfloat16):
                kv = s.wm.context(hq, txt, tm)
                kvn = s.wm.context(hq, txt, tm, drop=torch.ones(len(hq), dtype=torch.bool, device=h.device)) if s.a.cfg != 1.0 else None
                z = s.fm.sample(s.wm, kv, h[m_, -1].repeat(1, s.wm.n_fut, 1), s.wa.mode, steps=4, alpha=s.wa.alpha, sigma=s.wa.sigma,
                                rs=s.rs, tau0=tau, n_lan=n_lan, snr=0.15, cfg=s.a.cfg, kv_null=kvn)[0][:, :T]
            h[m_, -1] = (beta * z.float() + (1 - beta) * h[m_, -1].float()).to(h.dtype)
        return h

    @torch.no_grad()
    def anchor(s, y, tb):  # 场景锚：同一任务训练 demo 的第一帧里（逐通道去空间均值后）与 y 最近的一帧
        if not hasattr(s, "abank"):
            root = Path(s.wa.data)
            ix = np.load(root / "meta/index.npz")
            f = np.load(root / f"feats/{s.wa.encoder}/feats.npy", mmap_mode="r")
            rows = [r for r in range(len(ix["demo_start"])) if ix["split"][r] == 0]
            s.abank = ((torch.from_numpy(np.stack([f[ix["demo_start"][r]] for r in rows])).to(s.dev).float() - s.mu) / s.sd).half()
            s.atask = torch.from_numpy(ix["task_id"][rows].astype(np.int64)).to(s.dev)
        c = lambda z: z.float() - z.float().mean(-2, keepdim=True)
        out = []
        for i in range(len(y)):
            b = s.abank[s.atask == tb[i]]
            out.append(b[(c(b) - c(y[i])).pow(2).mean((1, 2)).argmin()])
        return torch.stack(out)

    @torch.no_grad()
    def pfilt(s, hp, h, ti, fresh):  # 后验净化（probwm/postfilt.py）：h 的最后一帧换成 ẑ_t；新 episode 整段历史都换成 ẑ_0
        if not hasattr(s, "flt"):
            from . import postfilt
            from .train import build
            pk = torch.load(s.a.prior, map_location="cpu")
            pa = Namespace(**pk["args"])
            pm = build(pa, pk["dims"]).to(s.dev).eval()
            pm.load_state_dict(pk["model"])
            s.flt = postfilt.Filter(s.wm, s.wa, pm, pa, pk["dims"][2], s.a.pfilt)
        z = s.flt.step(h[:, -1], hp, s.temb[ti], s.tmask[ti], fresh).to(h.dtype)
        h = torch.cat([h[:, :-1], z[:, None]], 1)
        return torch.where(fresh[:, None, None, None], z[:, None].expand_as(h), h)

    def ens_ms(s, blk, n_avg=1, every=False):  # (rteasy 11-19) --idm_ms：blk(net) → 该网络的 (B, n_act·m, ad) float 块；返回等权平均（every：另返回各网络的块）
        cur = torch.cuda.current_stream()
        ev = torch.cuda.Event()
        ev.record(cur)  # 分叉点：成员流只等它之前的工作（输入都已在这之前算好）
        o0 = torch.stack([blk(s.net) for _ in range(n_avg)]).mean(0)  # 主 IDM 照顺序版先发（先消耗随机数）
        eo = []
        for en, st_ in zip(s.enets, s.ms):
            st_.wait_event(ev)
            with torch.cuda.stream(st_):
                eo.append(blk(en))
        for st_, o_ in zip(s.ms, eo):
            cur.wait_stream(st_)
            o_.record_stream(cur)
        out = torch.stack([o0] + eo).mean(0)
        return (out, [o0] + eo) if every else out

    def accel(s, obj, tag, *names):  # --compile：tag 在 --compile_on 里的模块，把这些方法 torch.compile（在 --graphs 捕获之前；默认关 = 不动）
        a = s.a
        if tag and getattr(a, "compile", "") and tag in (getattr(a, "compile_on", "") or "").split(","):
            import torch._dynamo as dyn
            for k in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit", "accumulated_recompile_limit"):
                if hasattr(dyn.config, k):  # 按输入形状各编一份（B、CFG 两路、新回合的子批）：默认上限不够
                    setattr(dyn.config, k, max(getattr(dyn.config, k), 256))
            for n in names:
                setattr(obj, n, torch.compile(getattr(obj, n), mode=None if a.compile == "default" else a.compile, dynamic=False))

    def graph(s, obj, *names):  # --graphs：把 obj 的这些方法换成按输入形状捕获的 CUDA graph（结果逐位不变，见 probwm/cgraph.py）
        tag = "enc" if obj is s else "canon" if obj is getattr(s, "cm", None) else "idm" if isinstance(obj, s.idm.Net) else "wm" if obj is getattr(s, "wm", None) else ""
        s.accel(obj, tag, *names)  # --compile（默认关 = 不动）
        if getattr(s.a, "graphs", False):
            from .cgraph import wrap
            if not hasattr(s, "gpool"):
                s.gpool = torch.cuda.graph_pool_handle()
            for n in names:
                wrap(obj, n, s.gpool)

    def load_prior(s):  # 单帧无条件先验 p(z_t)（train --prior）
        if not hasattr(s, "prior"):
            from .train import build
            pk = torch.load(s.a.prior, map_location="cpu")
            s.pa = Namespace(**pk["args"])
            s.prior = build(s.pa, pk["dims"]).to(s.dev).eval()
            s.prior.load_state_dict(pk["model"])
            s.graph(s.prior, "context", "forward")
            s.pdt = pk["dims"][2]
            from .canon import null_len
            s.pL = null_len(s.pa)  # 与训练同样个数的 null token

    @torch.no_grad()
    def poe(s, y, hp, ti, fresh, slots=None):
        """专家乘积采样（composable diffusion 式）：y (B,T,D) 原始观测，hp (B,K,T,D) 上一时刻的已净化历史。chunk 状态 x 从纯噪声出发，
        第 1 帧槽位的速度 v = v_canon(x₀ | y) + w·(v_WM(x | hp, 指令)₀ − v_prior(x₀))，其余槽位用 v_WM；新 episode 没有历史，w=0（只用规范化器）。"""
        from .canon import load
        if not hasattr(s, "cm"):
            s.cm, s.ca, cn = load(s.a.canon, s.dev)
            s.graph(s.cm, "context", "forward")
            assert s.ca.mode == "abs" and s.ca.K == 1, "--poe 需要单帧、从纯噪声出发的规范化器"
        s.load_prior()
        v = s.a.poe.split(",")
        w, st = float(v[0]), int(v[1]) if len(v) > 1 else 4
        B, T, D, dev = *y.shape, y.device
        z0 = lambda: torch.zeros(B, 1, dtype=torch.bool, device=dev)
        wv = torch.where(fresh, 0.0, w)[:, None, None]
        if len(v) > 2:  # OOD 门控：规范化位移 mean((canon(y) − y)²) 不超过 thr 的帧不用 WM 项（只用规范化器）
            # 第 3 项的其他写法（不确定性门控，权重 ∈ [0,1]）：ramp:a:b = clip((位移 − a)/(b − a))；
            # var:K:a:b = clip((u − a)/(b − a))，u = K 个规范化样本（各自从噪声出发）的逐 token 方差均值 = 规范化器自己的不确定性
            from .canon import canonize
            g = v[2]
            K = int(g.split(":")[1]) if g.startswith("var") else 1
            zs = torch.stack([canonize(s.cm, s.ca, y, getattr(s.a, "poe_gsteps", 4), 0.0) for _ in range(K)])  # --poe_gsteps：门控探针的规范化步数
            d = (zs[0] - y.float()).pow(2).mean((1, 2))
            u = zs.var(0).mean((1, 2)) if K > 1 else torch.zeros_like(d)
            if g.startswith("ramp") or g.startswith("var"):
                a0, b0 = (float(x) for x in g.split(":")[-2:])
                wt = (((u if K > 1 else d) - a0) / (b0 - a0)).clamp(0, 1)
            else:
                wt = (d > float(g)).float()
            wv = wv * wt[:, None, None]
            if not hasattr(s, "pd"):  # 每个 env 本 episode 的累计（json 里 poe_d = 平均位移、poe_on = WM 项的平均权重、poe_u = 平均规范化方差）
                s.pd = torch.zeros(s.a.n_env, 4, device=dev)
                s.ptr = [[] for _ in range(s.a.n_env)]
            s.pd[slots] = torch.where(fresh[:, None], 0.0, s.pd[slots]) + torch.stack([d, wt, torch.ones_like(d), u], 1)
            if s.a.poe_trace:  # 逐决策步的 (位移, 方差)，给不确定性门控标定阈值用
                for j, w_ in enumerate(slots.tolist()):
                    s.ptr[w_] = ([] if fresh[j] else s.ptr[w_]) + [(round(d[j].item(), 4), round(u[j].item(), 5))]
        x = torch.randn(B, s.wm.n_fut * T, D, device=dev)
        ts = s.fm.shift(torch.linspace(0, 1, st + 1), s.ca.alpha).tolist()
        pc = s.a.poe_cond  # WM 项的条件（消融）：full = 历史 + 指令；nolang = 只有历史；shuf = 历史 + 别的随机指令（每个 episode 固定）；none = 无历史无指令（WM 的单帧边际）
        if pc == "shuf":
            if not hasattr(s, "shuf"):
                s.shuf = torch.zeros(s.a.n_env, dtype=torch.long, device=dev)
            s.shuf[slots] = torch.where(fresh, torch.randint(len(s.temb), (B,), device=dev), s.shuf[slots])
            ti = s.shuf[slots]
        with torch.autocast("cuda", torch.bfloat16):
            kc = s.cm.context(y[:, None], torch.zeros(B, 1, 1, device=dev), z0())
            if getattr(s.a, "poe_lazy", False) and not bool(wv.any()):  # --poe_lazy：整批 WM 权重为 0（门全关 / 新 episode）时 WM 与 prior 两项恒为 0，
                x = x[:, :T]  # 跳过它们，只积分规范化器；噪声照原形状抽，与默认路径逐位相同（codex 11-05 CUDA B=1/4 核验）
                for t0, t1 in zip(ts[:-1], ts[1:]):
                    x = x + (t1 - t0) * s.cm(x, torch.full((B,), t0, device=dev), kc).float()
                return x.half()
            kw = s.wm.context(torch.zeros_like(hp) if pc == "none" else hp, s.temb[ti], s.tmask[ti],
                              drop=torch.ones(B, dtype=torch.bool, device=dev) if pc in ("nolang", "none") else None)
            kp = s.prior.context(torch.zeros(B, s.pa.K, T, D, device=dev), torch.zeros(B, s.pL, s.pdt, device=dev),
                                 torch.zeros(B, s.pL, dtype=torch.bool, device=dev))
            for t0, t1 in zip(ts[:-1], ts[1:]):
                tt = torch.full((B,), t0, device=dev)
                u = s.wm(x, tt, kw).float()
                u[:, :T] = s.cm(x[:, :T], tt, kc).float() + wv * (u[:, :T] - s.prior(x[:, :T], tt, kp).float())
                x = x + (t1 - t0) * u
        return x[:, :T].half()

    def pick(s, hist, txt, tm, slots, fresh, kw, C=None, rank=0):
        """测试时分支选择：每个 env 采 n_cand 个 chunk，按 --select 打分取最高的一个 → (B,n_fut,T,D)。
        dc = 隐式分类器（与指令最一致）；medoid = 与其余候选平均距离最小（多数模式）；prev = 与上一次选中的计划（平移后）最接近；
        可用 + 组合（各项在候选间先标准化再相加）；新 episode 没有 prev 时该项为 0。"""
        B, C, n, T = len(hist), C or s.a.n_cand, s.wm.n_fut, hist.shape[2]
        hk, txk, tmk = (x.repeat_interleave(C, 0) for x in (hist, txt, tm))
        null = torch.ones(len(hk), dtype=torch.bool, device=hk.device)
        kv, kvn = s.wm.context(hk, txk, tmk), s.wm.context(hk, txk, tmk, drop=null)
        z_t = hk[:, -1].repeat(1, n, 1)
        z = s.fm.sample(s.wm, kv, z_t, s.wa.mode, alpha=s.wa.alpha, sigma=s.wa.sigma, rs=s.rs, cfg=s.a.cfg, kv_null=kvn, **kw)[0]
        zc, sc = z.view(B, C, n, T, -1), 0.0
        std = lambda x: (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)
        for k in s.a.select.split("+"):
            if k == "dc":
                x1 = s.fm.target(s.wa.mode, z, z_t[:, :T], s.rs)
                sc = sc + std(s.fm.dc_score(s.wm, kv, kvn, x1, s.a.dc_t, s.wa.alpha, s.wa.sigma, s.wa.mode).view(B, C))
            elif k == "medoid":
                f = zc.flatten(2).float()
                sc = sc + std(-torch.cdist(f, f).mean(2))
            elif k == "eef":  # 空间一致性：chunk 最后一帧的探针末端位置离定位目标越近越好（目标见 sguide；无目标的样本该项为 0）
                g, ok = s.cur_goal
                uv = s.uvp(zc[:, :, -1].flatten(0, 1)).float().view(B, C, 2)
                sc = sc + std(-((uv - g[:, None]) ** 2).sum(-1)) * ok[:, None]
            elif k == "prev":
                if s.prev is None:
                    s.prev = torch.zeros(s.a.n_env, n, T, hist.shape[3], dtype=hist.dtype, device=hist.device)
                d = max(1, s.a.exec // s.idm.STRIDE)
                pv = s.prev[slots][:, None, d:].float()
                e = -(zc[:, :, :n - d].float() - pv).pow(2).mean((2, 3, 4))
                sc = sc + torch.where(fresh[:, None], torch.zeros_like(e), std(e))
        best = zc[torch.arange(B, device=z.device), sc.argmax(1) if rank == 0 else sc.argsort(1, descending=True)[:, rank]] if torch.is_tensor(sc) else zc[:, rank]  # rank 1 = 第二名（--recover second）
        if s.prev is not None:
            s.prev[slots] = best.to(hist.dtype)
        return best

    @torch.no_grad()
    def rolling_x(s, hist, txt, tm, slots, fresh, kw, offs, v):
        """MS-far 滚动的两种推理变体（codex 11-05/06 续训 20k 的机制臂；只在给了 flag 时走这里，默认 msroll 路径不变）：
        --prog P1|P2|P3（渐进去噪，codex core/progressive_expansion_20261106/code/probwm/stream_cache.py）：槽每次决策只去噪到
          end_h = shift((L − h)/(L − 1))（L = max 时距 + 1 = 33；P1 只远槽 h > 8，近槽仍到干净端；P2/P3 所有槽），部分态 (x, τ = end) 按绝对时刻缓存，
          下次在同一绝对帧精确命中时从部分态接着去噪（P1 只远槽），P3 先把部分态往噪声端退 10%；给 IDM 的仍是干净端点估计 x + (1 − τ)·v。
        --abs_far（远槽锁定绝对时刻，codex core/absolute_expansion_20261106/code/probwm/absolute_roll.py）：时钟 φ = 本 episode 的决策数 mod 8，
          远槽（第 9 槽起）取 h − φ，即同一绝对帧连续跟 8 次决策；起始层级在原各时距节点的 τ 上按真实时距线性插值，时间槽嵌入按时距线性插值（model fpos）。
        其余同 msroll 路径：干净估计缓存每次平移一帧、没有精确估计的槽借 ≤ W 帧内最近的估计（--roll 第 3 项）、只支持每次 1 步、新 episode 用 --sampler 同步采。"""
        assert int(v[0]) == 1 and (len(v) < 2 or v[1] == 0), "--prog / --abs_far 只支持 --roll 1,0[,W]"
        B, K, T, D = hist.shape
        dev, al, n, L = hist.device, s.wa.alpha, len(offs), int(offs.max()) + 1
        if s.rw is None:
            s.rw, s.rv = torch.zeros(s.a.n_env, L, T, D, device=dev), torch.zeros(s.a.n_env, L, dtype=torch.bool, device=dev)
            s.rx, s.rt = torch.zeros_like(s.rw), torch.zeros(s.a.n_env, L, device=dev)  # 部分态缓存（--prog）
            s.rclk = torch.zeros(s.a.n_env, dtype=torch.long, device=dev)  # 本 episode 的决策数（--abs_far 的时钟）
        sh = lambda z: torch.cat([z[slots][:, 1:], torch.zeros_like(z[slots][:, :1])], 1)
        c, ok, px, pt = sh(s.rw), sh(s.rv) & ~fresh[:, None], sh(s.rx), sh(s.rt)
        s.rclk[slots] = torch.where(fresh, torch.zeros_like(s.rclk[slots]), s.rclk[slots] + 1)
        h = offs.expand(B, -1).clone()
        if s.a.abs_far:
            assert offs.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32], "--abs_far 按 MSF 的 11 个槽写的"
            h[:, 8:] -= (s.rclk[slots] % 8)[:, None]
        r = torch.arange(B, device=dev)[:, None]
        cs, ex = c[r, h], ok[r, h]
        hs = ex.clone()
        if len(v) > 2 and v[2] > 0:
            d = (torch.arange(L, device=dev)[None, None] - h[:, :, None]).abs().float().masked_fill(~ok[:, None], float("inf"))
            dm, j = d.min(-1)
            use = ~hs & (dm <= v[2])
            cs, hs = torch.where(use[..., None, None], c[r, j], cs), hs | use
        rank = s.fm.shift(1 - (torch.arange(n, device=dev) + 1) / n, al)  # 原各槽的起始层级（同 msroll）
        fp = None
        if s.a.abs_far:  # 起始层级与时间槽嵌入都在原节点之间按真实时距线性插值（节点处与原值逐位相同）
            kn, hf = offs.float(), h.float()
            rt_ = torch.searchsorted(kn, hf).clamp(1, n - 1)
            fr = ((hf - kn[rt_ - 1]) / (kn[rt_] - kn[rt_ - 1]))
            t0 = torch.lerp(rank[rt_ - 1], rank[rt_], fr)
            fq = s.wm.fpos[0, s.wm.K:s.wm.K + n, 0].float()
            fp = torch.lerp(fq[rt_ - 1], fq[rt_], fr[..., None])[:, :, None]
        else:
            t0 = rank.expand(B, -1)
        tau = torch.where(hs, t0, torch.zeros_like(t0))
        x = tau[..., None, None] * cs.float() + (1 - tau[..., None, None]) * torch.randn(B, n, T, D, device=dev)
        end = torch.ones_like(tau)
        if s.a.prog:
            eh = s.fm.shift((L - h.float()) / (L - 1), al)
            end = torch.where(h <= 8, torch.ones_like(eh), eh) if s.a.prog == "P1" else eh
            u = ex & (h > 8) if s.a.prog == "P1" else ex
            xp, tp = px[r, h], pt[r, h]
            if s.a.prog == "P3":  # 先把部分态退回 10%（同前向核：x1 系数 τ → 0.9τ，噪声补到 1 − 0.9τ）
                tg = 0.9 * tp
                q = torch.where(tp > 0, tg / tp.clamp_min(1e-12), torch.ones_like(tp))
                sd = ((1 - tg) ** 2 - (q * (1 - tp)) ** 2).clamp_min(0).sqrt()
                xp, tp = q[..., None, None] * xp + sd[..., None, None] * torch.randn_like(xp), tg
            x, tau = torch.where(u[..., None, None], xp, x), torch.where(u, tp, tau)
        kv = s.wm.context(hist, txt, tm)
        kvn = s.wm.context(hist, txt, tm, drop=torch.ones(B, dtype=torch.bool, device=dev)) if s.a.cfg != 1.0 else None
        p, xe = s.fm.step1(s.wm, kv, x, tau, end, cfg=s.a.cfg, kv_null=kvn, fpos=fp)
        if fresh.any():  # 新 episode 的时钟是 0，h = 原节点，同步采样与默认路径相同
            i = fresh.nonzero().squeeze(1)
            p[i] = s.fm.rollout(s.wm, hist[i], txt[i], tm[i], n, s.wa.mode, alpha=al, sigma=s.wa.sigma, rs=s.rs, cfg=s.a.cfg, **kw)[0].float()
            e_ = end[i][..., None, None]
            xe[i] = e_ * p[i] + (1 - e_) * torch.randn_like(p[i]) if s.a.prog else p[i]
        c[r, h], ok[r, h], px[r, h], pt[r, h] = p, True, xe, end
        s.rw[slots], s.rv[slots], s.rx[slots], s.rt[slots] = c, ok, px, pt
        return p

    @torch.no_grad()
    @torch.no_grad()
    def ov_sub(s, z, slots, fresh, fl, pe=None):
        """(pro) --ov mode[,rules[,thr_hi[,thr_lo[,cap]]]]（预注册 logs/pro.md 11-15 04:40）：观测不可信时用伪观测顶上。z (B,T,D) 本次编码的观测，
        fl (B,2) worker 的 [陈旧, 黑屏]（probwm/pro2_env.py）→ 替换后的 z。不可信 = R1 陈旧 | R2 黑屏 | R3 惊讶滞回：上一次决策对当前帧的
        干净估计（滚动缓存平移后的第 0 格，同 replan_check）与观测的 mean 统计量 > thr_hi 进入、之后 < thr_lo 才退出；rules 选用哪几条（默认 12）。
        新 episode 的第一次决策总是可信；连续不可信满 cap 次（默认 30）强制信一次、计数清零。伪观测：wm = 那格干净估计（连续不可信时 = WM 沿自己的
        预测往前想象；没有估计时不替换），hold = 本集最近一次可信的真实观测。只换 WM / IDM 看到的特征，不改随机数。"""
        mode, rules, hi, lo, cap = s.ov
        B, dev = len(z), z.device
        if s.ov_st is None:
            s.ov_st = torch.zeros(s.a.n_env, dtype=torch.bool, device=dev)
            s.ov_n = torch.zeros(s.a.n_env, dtype=torch.long, device=dev)
            s.ov_tr = torch.zeros(s.a.n_env, *z.shape[1:], dtype=z.dtype, device=dev)
        d_ = getattr(s.a, "roll_shift", None) or 1
        if s.rw is not None and s.rw.shape[2] == z.shape[1]:
            zh, val = s.rw[slots, d_].to(z.dtype), s.rv[slots, d_] & ~fresh
        else:
            zh, val = torch.zeros_like(z), torch.zeros(B, dtype=torch.bool, device=dev)
        sv = ((z.float() - zh.float()) / s.rs).pow(2).mean((1, 2))
        if mode in ("soft", "softlog"):  # (pro) OV-soft：逐 patch 残差 r_p，w_p = σ((r_p − θ)/s)，z_p ← (1 − w_p) z_p + w_p ẑ_p；没有估计时不混
            ep = ((z.float() - zh.float()) / s.rs).pow(2).mean(-1)  # (B, T)
            if mode == "softlog":
                if val.any():
                    s.ov_res.append(ep[val].half().cpu())
                zf = torch.zeros(B, device=dev)
                s.ov_last = (sv, val, zf, zf)
                return z
            w = torch.sigmoid((ep - hi) / lo) * val[:, None].float()
            z = (z.float() + w[..., None] * (zh.float() - z.float())).to(z.dtype)
            s.ov_last = (sv, val, w.mean(1), (w > 0.5).float().mean(1))
            return z
        st = s.ov_st[slots] & ~fresh
        nn_ = torch.where(fresh, torch.zeros_like(s.ov_n[slots]), s.ov_n[slots])
        r3 = val & (sv > torch.where(st, torch.full_like(sv, lo), torch.full_like(sv, hi)))
        no = torch.zeros(B, dtype=torch.bool, device=dev)
        r3 = r3 if "3" in rules else no
        r4 = no
        if "4" in rules:  # (pro PC2，预注册 logs/pro.md 11-16 16:4x) 视觉-本体一致性：观测特征与前 k 个决策的观测几乎相同（Δz < τ_z）而末端移动 > τ_q → 陈旧
            tz, tq = s.ov_r4
            assert pe is not None, "--ov 规则 4 要策略带本体感觉（末端位置）"
            if getattr(s, "ov_zh", None) is None:
                s.ov_zh = torch.zeros(s.a.n_env, 12, *z.shape[1:], dtype=z.dtype, device=dev)
                s.ov_eh = torch.zeros(s.a.n_env, 12, 3, device=dev)
                s.ov_hn = torch.zeros(s.a.n_env, dtype=torch.long, device=dev)
            hn = torch.where(fresh, torch.zeros_like(s.ov_hn[slots]), s.ov_hn[slots])
            dz = ((z.float()[:, None] - s.ov_zh[slots].float()) / s.rs).pow(2).mean((2, 3))  # (B, 12)：第 j 格 = 前 j + 1 个决策的观测
            dq = (pe.float()[:, None] - s.ov_eh[slots]).norm(dim=-1)
            okh = torch.arange(12, device=dev)[None] < hn[:, None]
            hit = okh & (dz < tz) & (dq > tq)
            r4 = hit[:, 0] | (okh[:, 0] & (dz[:, 0] >= tz) & hit[:, 1:].any(1))
            s.ov_zh[slots] = torch.cat([z[:, None], s.ov_zh[slots][:, :-1]], 1)  # 存原始观测（替换前）
            s.ov_eh[slots] = torch.cat([pe.float()[:, None], s.ov_eh[slots][:, :-1]], 1)
            s.ov_hn[slots] = (hn + 1).clamp(max=12)
        s.ov_r4last = r4
        u = (fl[:, 0] if "1" in rules else no) | (fl[:, 1] if "2" in rules else no) | r3 | r4
        u = u & ~fresh & (nn_ < cap)
        s.ov_st[slots] = r3 & u
        s.ov_n[slots] = torch.where(u, nn_ + 1, torch.zeros_like(nn_))
        sub = u & val if mode == "wm" else u
        z = torch.where(sub[:, None, None], zh if mode == "wm" else s.ov_tr[slots], z)
        s.ov_tr[slots] = torch.where(u[:, None, None], s.ov_tr[slots], z)
        s.ov_last = (sv, val, u, sub)
        return z

    def replan_check(s, zh, val, z, slots, fresh):
        """(hz2) --replan thr[,mode[,cool[,stat]]] 的检测：zh (B,T,D) 上一次决策对当前帧的干净估计（滚动缓存平移后的第 0 格 = 上次的 t+1 槽），
        val (B,) 有没有这个估计（新 episode 没有），z (B,T,D) 本次观测（WM 看到的那份，--canon 时是规范化后的）。三个统计量（标准化空间）：
        mean = mean(((z − ẑ)/rs)²)（patch、通道平均，同 --surp），top = 逐 patch 的这个量取最大 16 个 patch 的均值，cos = 1 − 逐 patch 余弦的均值。
        stat 那一项 > thr、且不在冷却期（上次触发后 cool 次决策内不再触发）→ 触发。只读不改随机数：thr = inf 时结果与不加 --replan 逐位相同。"""
        thr, _, cool, st = s.rp
        if s.rp_cd is None:
            s.rp_cd = torch.zeros(s.a.n_env, dtype=torch.long, device=z.device)
        Tm = zh.shape[1]
        zf, hf = z[:, :Tm].float(), zh.float()
        ep = ((zf - hf) / s.rs).pow(2).mean(2)
        sv = torch.stack([ep.mean(1), ep.topk(min(16, Tm), 1).values.mean(1), 1 - torch.nn.functional.cosine_similarity(zf, hf, dim=-1).mean(1)], 1)
        cd = torch.where(fresh, torch.zeros_like(s.rp_cd[slots]), s.rp_cd[slots])
        trig = val & (sv[:, st] > thr) & (cd == 0)
        s.rp_cd[slots] = torch.where(trig, torch.full_like(cd, cool), (cd - 1).clamp_min(0))
        s.rp_last = (sv, val, trig)
        return trig

    @torch.no_grad()
    def rolling(s, hist, txt, tm, slots, fresh, kw, kf=None):
        """滚动去噪（--roll k[,δ[,lan[,grow]]]，fm.roll）：每个 env 保留一个 n_fut 帧的窗口，跨决策逐步去噪；新 episode 先用 --sampler 同步采一整段，
        再加噪到滚动剖面（fm.roll_init）。每次决策执行 2 个环境步（= 1 帧），窗口平移 1 帧。→ (B, n_fut, T, D) 给 IDM 的计划。
        kf (B,T,D)：两级的关键帧条件（WM --kfcond），条件 / 无条件两路的上下文都带。不做滚动训练的 WM 也可以直接这样推理（ctl_a1）。"""
        ok_ = {} if kf is None else {"kf": kf}
        v = [float(x) for x in s.a.roll.split(",")]
        n, (B, K, T, D) = s.wm.n_fut, hist.shape
        if getattr(s.wa, "tgt_offs", ""):  # 多尺度滚动（稀疏时距槽，fm.msroll）：按绝对时间缓存每帧最近一次的干净估计（缓存长 max(offs) + 1，每次决策平移一帧），
            # 各槽"到站"时从缓存取 t+k 的估计加噪回该槽层级再去噪；新 episode 用 --sampler 同步采
            offs = torch.tensor([int(x) for x in s.wa.tgt_offs.split(",")], device=hist.device)
            if getattr(s.a, "prog", "") or getattr(s.a, "abs_far", False):
                return s.rolling_x(hist, txt, tm, slots, fresh, kw, offs, v)
            if s.rw is None:  # (rteasy) 缓存只存 WM 预测的 token（train --tgt_tok 的 WM：上下文里多出的视角不进缓存；普通 WM 的 pos 长度 = T，与原来相同）
                s.rw = torch.zeros(s.a.n_env, int(offs.max()) + 1, s.wm.pos.shape[1] if hasattr(s.wm, "pos") else T, D, device=hist.device)
                s.rv = torch.zeros(s.a.n_env, int(offs.max()) + 1, dtype=torch.bool, device=hist.device)
            d_ = getattr(s.a, "roll_shift", None) or 1  # (rcx) 两次决策之间过去的 10 Hz 帧数（robocasa_eval --obs_every 时 = exec / 2）；默认 1 = 原行为
            c = torch.cat([s.rw[slots][:, d_:], torch.zeros_like(s.rw[slots][:, :d_])], 1)
            ok = torch.cat([s.rv[slots][:, d_:], torch.zeros_like(s.rv[slots][:, :d_])], 1) & ~fresh[:, None]
            rtrig = None
            if getattr(s.a, "replan", ""):  # (hz2) 惊讶触发重规划：上一次决策对当前帧的估计（缓存平移后的第 0 格）与本次观测比
                rtrig = s.replan_check(c[:, 0], ok[:, 0], hist[:, -1], slots, fresh)
                if rtrig.any():  # 丢掉陈旧计划：all = 整行缓存作废，far = 远槽（时距 ≥ 9）那段作废；msroll 之后这些行用 --sampler 同步重采（同新 episode）
                    ok = ok & ~(rtrig[:, None] & (torch.arange(c.shape[1], device=c.device)[None] >= (9 if s.rp[1] == "far" else 0)))
            of_ = getattr(s, "orc_far", None)  # (rteasy) 部分 oracle（robotwin_eval --oracle_plan tf）：{env: (专家特征, 对齐帧)}；None = 原行为
            if of_:  # 远槽（时距 ≥ 9）对应绝对时刻的缓存估计换成专家帧，之后照常加噪回各槽层级、与近槽联合去噪（近槽只经联合去噪 / 缓存借用间接受益）
                far = offs[offs >= 9]
                for i, w in enumerate(slots.tolist()):
                    if w in of_ and not bool(fresh[i]):
                        fe_, d0 = of_[w]
                        c[i, far] = fe_[torch.clamp(d0 + far, max=len(fe_) - 1)].to(c.dtype)
                        ok[i, far] = True
            shd = getattr(s, "sh_demo", None)  # (mika) train --self_hist 的 WM：本 episode 的历史 token（mikasa_eval 每次决策前设；None = 原行为）
            dk = (lambda ix=None: {} if shd is None else {"demo": shd if ix is None else tuple(x[ix] for x in shd)})
            mk = {} if getattr(s, "mem_in", None) is None else {"mem": s.mem_in}  # (mika C2) train --mem 的 WM：上一个记忆步写出的记忆（mikasa_eval 设；None = 原行为）
            kv = s.wm.context(hist, txt, tm, **dk(), **mk)
            s.mem_out = getattr(kv, "mem", None)  # (mika C2) 本次前向写出的新记忆（评测器只在记忆步网格上取用）
            cfar = getattr(s.a, "cfg_far", -1.0)  # --cfg_far ≥ 0：远槽（时距 ≥ 9）的 CFG 单独设，近槽仍用 --cfg
            kvn = s.wm.context(hist, txt, tm, drop=torch.ones(B, dtype=torch.bool, device=hist.device), **dk(), **mk) if s.a.cfg != 1.0 or cfar >= 0 else None
            cfgv = torch.where(offs >= 9, cfar, s.a.cfg) if cfar >= 0 else None
            cs, hs = c[:, offs], ok[:, offs]
            fsep = None
            if getattr(s.a, "far_sep", False):  # (objslot) 远槽是 slot 帧（train --far_slots）：缓存里远槽来的估计（平移后位置 ≥ 近槽最大时距）不给近槽用，近槽来的也不给远槽用
                fm_ = getattr(s.wa, "far_slot_min", 9)
                pos_ = torch.arange(c.shape[1], device=c.device)[None]
                fsep = torch.where((offs >= fm_)[:, None], pos_ >= int(offs[offs < fm_].max()), pos_ < int(offs[offs < fm_].max()))  # (n, L)
                hs = hs & fsep[torch.arange(len(offs), device=c.device), offs][None]
            if len(v) > 2 and v[2] > 0:  # 第 3 项 W：没有精确估计的槽，用绝对时间最近（≤ W 帧）的已有估计代替（"两个检查点之间沿用上一次的估计"，如远槽 t+16 用上次的 t+15）
                d = (torch.arange(c.shape[1], device=c.device)[None, None] - offs[None, :, None]).abs().float().expand(B, -1, -1).masked_fill(~ok[:, None], float("inf"))
                if fsep is not None:
                    d = d.masked_fill(~fsep[None], float("inf"))
                dm, j = d.min(-1)
                use = ~hs & (dm <= v[2])
                cs = torch.where(use[..., None, None], c[torch.arange(B, device=c.device)[:, None], j], cs)
                hs = hs | use
            rk, evr = None, getattr(s.wa, "ev_roll", False)
            if evr and s.ev_c is None:
                s.ev_c = torch.zeros(s.a.n_env, T, D, device=hist.device)
                s.ev_d = torch.full((s.a.n_env,), float("inf"), device=hist.device)
                s.ev_g = torch.zeros(s.a.n_env, dtype=torch.bool, device=hist.device)
                s.ev_flip = torch.zeros(s.a.n_env, dtype=torch.bool, device=hist.device)
            if evr:  # (explore) train --ev_roll：事件槽的上次估计保留，按估计时距（上次 − 1 帧）在各槽里的名次加噪回该层级再细化；新 episode / 上一段执行里夹爪翻转（事件已到）时从纯噪声
                hev = ~fresh & ~s.ev_flip[slots] & torch.isfinite(s.ev_d[slots])
                dev_ = torch.where(hev, (s.ev_d[slots] - 1).clamp_min(1), torch.full_like(s.ev_d[slots], float("inf")))
                cs, hs = torch.cat([cs, s.ev_c[slots][:, None]], 1), torch.cat([hs, hev[:, None]], 1)
                rk = torch.cat([offs.float()[None].expand(B, -1), dev_[:, None]], 1).sort(dim=1, stable=True).indices.argsort(1)
            elif getattr(s.wa, "ev_slot", 0):  # (explore) train --ev_slot：最后一槽是"下一个语义事件"（时距可变，不进绝对时间缓存），每次决策从纯噪声与其他槽一起去噪
                cs, hs = torch.cat([cs, torch.zeros_like(cs[:, :1])], 1), torch.cat([hs, torch.zeros_like(hs[:, :1])], 1)
            if cfgv is not None and len(cfgv) < cs.shape[1]:  # 事件槽（ev_slot / ev_roll）补在最后：用远槽的权重
                cfgv = torch.cat([cfgv, cfgv.new_full((cs.shape[1] - len(cfgv),), cfar)])
            rkk = getattr(s.a, "roll_k", 0) or 0
            gfirst = getattr(s.a, "goal_first", False)
            al_ = torch.where(offs < 9, float(s.wa.alpha), float(s.wa.alpha_far)) if getattr(s.wa, "alpha_far", 0.0) else s.wa.alpha  # (rteasy 臂 L) 逐槽 shift
            lk_ = getattr(s.a, "lead_k", 0) or 0  # (rteasy 臂 G2) 远槽领先的多步去噪
            if gfirst:  # (rteasy 臂 G，train --goal_first 的 WM) 目标先行两阶段：阶段 1 近槽（时距 < 9）纯噪声不动，远槽从缓存加噪到 u = 1 − 1/n（原剖面最轻的一级）
                # 走 k 步到干净；阶段 2 远槽钉在阶段 1 的结果（u = 1 不动），近槽按原滚动层级 1 − (j+1)/n − δ 加噪、走 k 步到干净。新 episode 的行两阶段都从纯噪声走 --gf_fresh 步。
                # 配 --oracle_plan tf：阶段 1 之后远槽换成专家帧（干净），即“远槽给对”的部分 oracle（G 的训练剖面里远槽干净、近槽加噪在分布内）
                assert rkk <= 1 and rk is None and cfgv is None and not getattr(s.wa, "ev_slot", 0) and fsep is None, "--goal_first 不与 --roll_k / 事件槽 / --cfg_far / --far_sep 同用"
                nr_, n_, ch_ = offs < 9, len(offs), v[1] if len(v) > 1 else 0.0
                def gf2(kv_, kvn_, c_, h_, k_, fix=None):
                    nr = nr_[None].expand(len(c_), -1)
                    us1 = torch.where(nr, 0.0, torch.where(h_, 1 - 1 / n_, 0.0))
                    x1_ = s.fm.msroll(s.wm, kv_, c_, h_, al_, k_, cfg=s.a.cfg, kv_null=kvn_, us=us1, uf=torch.where(nr, 0.0, 1.0))[0]
                    if fix is not None:
                        fix(x1_)
                    us2 = torch.where(nr, torch.where(h_, (1 - (torch.arange(n_, device=c_.device)[None] + 1) / n_ - ch_).clamp_min(0), 0.0), 1.0)
                    return s.fm.msroll(s.wm, kv_, torch.where(nr[..., None, None], c_.float(), x1_), h_, al_, k_, cfg=s.a.cfg, kv_null=kvn_, us=us2)[0]
                def ofix(x_):  # --oracle_plan tf：远槽换成按步数对齐的专家帧
                    fo = offs[~nr_]
                    for i, w in enumerate(slots.tolist()):
                        if w in of_ and not bool(fresh[i]):
                            fe_, d0 = of_[w]
                            x_[i, ~nr_] = fe_[torch.clamp(d0 + fo, max=len(fe_) - 1)].float()
                p = gf2(kv, kvn, cs, hs, int(v[0]), ofix if of_ else None)
                if fresh.any():
                    i = fresh.nonzero().squeeze(1)
                    kvf = s.wm.context(hist[i], txt[i], tm[i], **dk(i))
                    kvnf = s.wm.context(hist[i], txt[i], tm[i], drop=torch.ones(len(i), dtype=torch.bool, device=hist.device), **dk(i)) if s.a.cfg != 1.0 else None
                    p[i] = gf2(kvf, kvnf, cs[i], torch.zeros_like(hs[i]), getattr(s.a, "gf_fresh", 4))
            elif lk_ > 0:  # (rteasy 臂 G2，train --lead 的 WM) 每次决策 lead_k 步，远槽进度领先（fm.msroll lead）；新 episode 的行从纯噪声走 --gf_fresh 步、同样远槽领先
                assert rkk <= 1 and rk is None and cfgv is None and not getattr(s.wa, "ev_slot", 0), "--lead_k 不与 --roll_k / 事件槽 / --cfg_far 同用"
                lm_ = offs >= 9
                p = s.fm.msroll(s.wm, kv, cs, hs, al_, lk_, v[1] if len(v) > 1 else 0.0, cfg=s.a.cfg, kv_null=kvn, lead=lm_)[0]
                if fresh.any():
                    i = fresh.nonzero().squeeze(1)
                    kvf = s.wm.context(hist[i], txt[i], tm[i], **dk(i))
                    kvnf = s.wm.context(hist[i], txt[i], tm[i], drop=torch.ones(len(i), dtype=torch.bool, device=hist.device), **dk(i)) if s.a.cfg != 1.0 else None
                    p[i] = s.fm.msroll(s.wm, kvf, cs[i], torch.zeros_like(hs[i]), al_, getattr(s.a, "gf_fresh", 4), cfg=s.a.cfg, kv_null=kvnf, lead=lm_)[0]
            elif rkk > 1:  # (rteasy) --roll_k K：同一缓存 / 上下文批量采 K 个计划（各自独立加噪），按远槽（时距 ≥ 9）特征选一个交给 IDM 并写回缓存
                assert rk is None and cfgv is None and not getattr(s.wa, "ev_slot", 0), "--roll_k 不与事件槽 / --cfg_far 同用"
                kc = getattr(s.a, "roll_kc", 0) or rkk  # (rteasy) --roll_kc：每次前向只采这么多份（显存不够时分块；默认 K 份一次）
                rep = lambda q, m: None if q is None else s.fm.rep_kv(q, m)
                pK = torch.cat([s.fm.msroll(s.wm, rep(kv, m), cs.repeat_interleave(m, 0), hs.repeat_interleave(m, 0), al_, int(v[0]), v[1] if len(v) > 1 else 0.0,
                                            cfg=s.a.cfg, kv_null=rep(kvn, m))[0].unflatten(0, (B, m))
                                for m in [min(kc, rkk - j) for j in range(0, rkk, kc)]], 1)  # (B, K, n, T, D)
                ff = pK[:, :, offs >= 9].float().flatten(2)
                dm = torch.cdist(ff, ff).pow(2) / ff.shape[-1]  # (B, K, K) 远槽逐元素均方距离
                md = dm.sum(-1) / (rkk - 1)  # 每个计划到其余计划的平均距离
                if getattr(s.a, "roll_sel", "medoid") == "cluster":  # 最大簇：距离小于该行两两距离中位数的邻居最多的那个，平手取 medoid
                    iu = torch.triu_indices(rkk, rkk, 1, device=dm.device)
                    thr = dm[:, iu[0], iu[1]].median(1).values
                    nb = (dm < thr[:, None, None]).sum(-1) - 1
                    sel = (nb.float() - md / (md.max(1, keepdim=True).values + 1e-9)).argmax(1)
                else:  # medoid：到其余计划平均距离最小
                    sel = md.argmin(1)
                p = pK[torch.arange(B, device=pK.device), sel]
                s.plan_div = md.mean(1)  # 诊断：K 个计划远槽两两平均距离（分歧度），新 episode 的行无意义
            else:
                us_ = uf_ = None
                rus = getattr(s.a, "roll_us", -1.0)
                if rus is not None and rus >= 0:  # (pro) --roll_us u0：有缓存的槽全部加噪回同一层级 u0（训练里"同步"那一半样本的剖面，在分布内），各自走到干净
                    us_ = torch.where(hs, torch.full(hs.shape, float(rus), device=hs.device), torch.zeros(hs.shape, device=hs.device))
                if of_ and getattr(s.a, "oracle_pin", False):  # (pro) --oracle_pin：注入的专家远槽钉住（u = 1，不加噪、不动），其余槽照常
                    assert rk is None, "--oracle_pin 不接事件槽"  # 训练剖面（--roll_sort 单调：近槽不比远槽噪）里没有"远槽干净、近槽加噪"的状态 → 分布外
                    pin = torch.zeros_like(hs)
                    for i, w in enumerate(slots.tolist()):
                        if w in of_ and not bool(fresh[i]):
                            pin[i, offs >= 9] = True
                    base = us_ if us_ is not None else torch.where(hs, (1 - (torch.arange(hs.shape[1], device=hs.device)[None] + 1) / hs.shape[1]
                                                                         - (v[1] if len(v) > 1 else 0.0)).clamp_min(0), torch.zeros(hs.shape, device=hs.device))
                    us_, uf_ = torch.where(pin, 1.0, base), torch.where(pin, 1.0, torch.ones_like(base))
                t0s = getattr(s.a, "t0_sde", -1.0)
                if s.cslot and t0s is not None and t0s >= 0:  # (rteasy 11-17) --t0_sde τ0：canon_slot WM 的 t+0 槽从"当前观测加噪到 τ0"起去噪（SDEdit 式投影），代替上一次决策对当前帧的估计；
                    # 其余槽与剖面照旧。τ0 是混合系数（τ = 1 数据端），换成均匀时间 u0 = unshift(τ0, α)；α = 13 时默认滚动里 t+0 槽的层级 u = 1 − 1/n 约等于 τ ≈ 0.5
                    j0 = (offs == 0).nonzero().squeeze(1)
                    assert len(j0) == 1, "--t0_sde 要 tgt_offs 里有 0 槽"
                    cs, hs = cs.clone(), hs.clone()
                    cs[:, j0] = hist[:, -1:, :cs.shape[2]].to(cs.dtype)  # 当前观测（WM 看到的那份；多视角 WM 只取预测视角的 token）
                    hs[:, j0] = True
                    base = us_ if us_ is not None else torch.where(hs, (1 - (torch.arange(hs.shape[1], device=hs.device)[None] + 1) / hs.shape[1]
                                                                         - (v[1] if len(v) > 1 else 0.0)).clamp_min(0), torch.zeros(hs.shape, device=hs.device))
                    us_ = base.clone()
                    us_[:, j0] = float(s.fm.unshift(torch.tensor(float(t0s)), float(s.wa.alpha)))
                p, _ = s.fm.msroll(s.wm, kv, cs, hs, al_, int(v[0]), v[1] if len(v) > 1 else 0.0, cfg=s.a.cfg, kv_null=kvn, **({"rank": rk} if rk is not None else {}),
                                   **({"cfgv": cfgv} if cfgv is not None else {}), **({"us": us_} if us_ is not None else {}), **({"uf": uf_} if uf_ is not None else {}))
            if fresh.any() and not gfirst and not lk_:
                i = fresh.nonzero().squeeze(1)
                p[i] = s.fm.rollout(s.wm, hist[i], txt[i], tm[i], n, s.wa.mode, alpha=s.wa.alpha, sigma=s.wa.sigma, rs=s.rs, cfg=s.a.cfg, **dk(i), **kw)[0].float()
            if rtrig is not None and rtrig.any():  # (hz2) 触发的 env：同步从头采一整段（以当前真实历史为条件）；far 只换远槽
                i = rtrig.nonzero().squeeze(1)
                q = s.fm.rollout(s.wm, hist[i], txt[i], tm[i], n, s.wa.mode, alpha=s.wa.alpha, sigma=s.wa.sigma, rs=s.rs, cfg=s.a.cfg, **dk(i), **kw)[0].float()
                if s.rp[1] == "far":
                    fj = (offs >= 9).nonzero().squeeze(1)
                    p[i[:, None], fj[None]] = q[:, fj]
                else:
                    p[i] = q
            if evr:  # 事件槽的时距估计：与本次计划里哪个固定槽的帧最像（标准化特征的均方）就取那个槽的时距
                e = (p[:, :len(offs)].float() - p[:, -1:].float()).pow(2).mean((2, 3))
                s.ev_c[slots], s.ev_d[slots] = p[:, -1], offs.float()[e.argmin(1)]
            c[:, offs], ok[:, offs] = p[:, :len(offs)], True
            s.rw[slots], s.rv[slots] = c, ok
            return p
        if s.rw is None:
            s.rw = torch.zeros(s.a.n_env, n * T, D, device=hist.device)
        kv = s.wm.context(hist, txt, tm, **ok_)
        kvn = s.wm.context(hist, txt, tm, drop=torch.ones(B, dtype=torch.bool, device=hist.device), **ok_) if s.a.cfg != 1.0 else None
        x, p, _ = s.fm.roll(s.wm, kv, s.rw[slots], s.wa.alpha, int(v[0]), v[1] if len(v) > 1 else 0.0, int(v[2]) if len(v) > 2 else 0,
                            len(v) > 3 and v[3] > 0, cfg=s.a.cfg, kv_null=kvn)
        if fresh.any():
            i = fresh.nonzero().squeeze(1)
            q = s.fm.rollout(s.wm, hist[i], txt[i], tm[i], n, s.wa.mode, alpha=s.wa.alpha, sigma=s.wa.sigma, rs=s.rs, cfg=s.a.cfg,
                             kf=None if kf is None else kf[i], **kw)[0].flatten(1, 2).float()
            p[i], x[i] = q, s.fm.roll_init(q, n, s.wa.alpha)
        s.rw[slots] = x
        return p.unflatten(1, (n, T))

    @torch.no_grad()
    def rolling_pre(s, hist, txt, tm, slots, fresh, kw):
        """(xden) --idm_tau pre：MS-far 滚动的"先出动作、后做 WM 这一步"。缓存平移、取各槽的估计（同 rolling 的 msroll 路径，含 --roll 第 3 项 W），
        加噪回各槽层级得到去噪前的窗口（fm.msroll_init）→ (窗口 (B,n,T,D), 各槽 τ (B,n))，交给噪声感知 IDM（idm --tau_in）直接读；
        WM 的上下文和这一步去噪（结果写回缓存，供下一次决策）在 roll_commit 里做，排在动作发出之后（不在帧 → 动作的关键路径上）。
        新 episode 没有缓存：同 rolling，用 --sampler 同步从头采一整段（串行），IDM 读这份干净计划（τ = 1）。"""
        v = [float(x) for x in s.a.roll.split(",")]
        n, (B, K, T, D) = s.wm.n_fut, hist.shape
        offs = torch.tensor([int(x) for x in s.wa.tgt_offs.split(",")], device=hist.device)
        Tp = s.wm.pos.shape[1] if hasattr(s.wm, "pos") else T  # (rteasy 11-18) 同 rolling：缓存 / 窗口只存 WM 预测的 token（train --tgt_tok 的多视角 WM 上下文多出的视角不进缓存）；普通 WM Tp = T，与原来相同
        if s.rw is None:
            s.rw = torch.zeros(s.a.n_env, int(offs.max()) + 1, Tp, D, device=hist.device)
            s.rv = torch.zeros(s.a.n_env, int(offs.max()) + 1, dtype=torch.bool, device=hist.device)
        c = torch.cat([s.rw[slots][:, 1:], torch.zeros_like(s.rw[slots][:, :1])], 1)
        ok = torch.cat([s.rv[slots][:, 1:], torch.zeros_like(s.rv[slots][:, :1])], 1) & ~fresh[:, None]
        cs, hs = c[:, offs], ok[:, offs]
        if len(v) > 2 and v[2] > 0:  # 同 rolling：没有精确估计的槽借绝对时间最近（≤ W 帧）的估计
            d = (torch.arange(c.shape[1], device=c.device)[None, None] - offs[None, :, None]).abs().float().expand(B, -1, -1).masked_fill(~ok[:, None], float("inf"))
            dm, j = d.min(-1)
            use = ~hs & (dm <= v[2])
            cs, hs = torch.where(use[..., None, None], c[torch.arange(B, device=c.device)[:, None], j], cs), hs | use
        ch = v[1] if len(v) > 1 else 0.0
        x0, us = s.fm.msroll_init(cs, hs, s.wa.alpha, ch)
        xs, tau, pf = x0.unflatten(1, (n, Tp)), s.fm.shift(us, s.wa.alpha), None
        if s.a.idm_tau == "stale" or s.stl:  # --idm_tau stale / (xden2) --idm_stale：IDM 读上一次决策的干净估计（平移一帧，没有精确估计的槽同上借用），当作干净计划（τ = 1）；WM 步同样在动作之后
            xs, tau = cs.float().clone(), torch.ones_like(tau)
        shd = getattr(s, "sh_demo", None)  # (gawm X2F5CU) train --self_hist 的 WM：本次决策的历史 token（评测器先设好），新 episode 的同步采与 roll_commit 的那一步都带上；None = 原行为
        dk = (lambda ix=None: {} if shd is None else {"demo": shd if ix is None else tuple(x[ix] for x in shd)})
        if fresh.any():
            i = fresh.nonzero().squeeze(1)
            pf = s.fm.rollout(s.wm, hist[i], txt[i], tm[i], n, s.wa.mode, alpha=s.wa.alpha, sigma=s.wa.sigma, rs=s.rs, cfg=s.a.cfg, **dk(i), **kw)[0].float()
            xs, tau = xs.clone(), tau.clone()
            xs[i], tau[i] = pf, 1.0
        s._pend, s._pend_shd = (hist, txt, tm, slots, fresh, c, ok, offs, cs, hs, x0, us, int(v[0]), ch, pf), shd
        return xs, tau

    @torch.no_grad()
    def roll_commit(s):  # (xden) --idm_tau pre 的后半：WM 以本次观测为条件做这一步滚动去噪，干净估计写回缓存（新 episode 的行用同步采的那份）
        hist, txt, tm, slots, fresh, c, ok, offs, cs, hs, x0, us, k, ch, pf = s._pend
        s._pend = None
        shd = getattr(s, "_pend_shd", None)  # (gawm X2F5CU) self_hist 的历史 token（rolling_pre 存下的同一份）；None = 原行为
        dkw = {} if shd is None else {"demo": shd}
        kv = s.wm.context(hist, txt, tm, **dkw)
        kvn = s.wm.context(hist, txt, tm, drop=torch.ones(len(hist), dtype=torch.bool, device=hist.device), **dkw) if s.a.cfg != 1.0 else None
        p, _ = s.fm.msroll(s.wm, kv, cs, hs, s.wa.alpha, k, ch, cfg=s.a.cfg, kv_null=kvn, init=(x0, us))
        if pf is not None:
            p[fresh.nonzero().squeeze(1)] = pf
        c[:, offs], ok[:, offs] = p, True
        s.rw[slots], s.rv[slots] = c, ok
        if s.cslot:  # (xden2) --idm_stale：当前帧的 t+0 输出这时才有 → 换掉规范化历史里先放的那一帧（预测 / 原始观测），之后各步的 IDM（与 --canon_hist 的 WM）历史用它
            s.chist[slots, -1] = p[:, 0, :s.chist.shape[2]].to(s.chist.dtype)

    @torch.no_grad()
    def ch_hist(s, hist, txt, tm, slots, fresh, kw):
        """(xden2) train --canon_hist 的 WM 读的历史：之前各步 t+0 槽的输出（s.chist 的后 K−1 帧）+ 当前原始观测 hist[:, -1]（t+0 要从它算出来）。
        新 episode 还没有 t+0 输出：先以原始历史（第一帧复制）用 --sampler 同步采一次，t+0 槽当作这一帧的估计、复制成 K−1 帧历史，之后同 rolling 的新 episode 再同步采。"""
        if s.chist is None:
            s.chist = torch.zeros(s.a.n_env, *hist.shape[1:], dtype=hist.dtype, device=hist.device)
        hw = torch.cat([s.chist[slots, 1:], hist[:, -1:]], 1)
        if fresh.any():
            i = fresh.nonzero().squeeze(1)
            e0 = s.fm.rollout(s.wm, hist[i], txt[i], tm[i], s.wm.n_fut, s.wa.mode, alpha=s.wa.alpha, sigma=s.wa.sigma, rs=s.rs, cfg=s.a.cfg, **kw)[0][:, 0]
            hw[i] = torch.cat([e0[:, None, :hist.shape[2]].to(hist.dtype).expand(-1, hist.shape[1] - 1, -1, -1), hist[i, -1:]], 1)
        return hw

    @torch.no_grad()
    def rolling_vpp(s, hist, txt, tm, slots, fresh, kw):
        """(vpp) 同 rolling（MS-far msroll 路径，结果逐位相同），另把 msroll 那一步 WM 前向（rolling 里第一次调 WM 的输出头）的条件路
        trunk 输出前 H 个槽（t+1..t+H）存进 s.vhid (B, H, T, d)，给 idm --vpp 的 IDM——稳态不多跑 WM。
        新 episode 的行：msroll 这一步没有缓存（各槽从纯噪声起），计划是 --sampler 同步从头采的；这些行另用这份计划按滚动剖面加噪（同训练的 vpp_hidden）
        做一次 trunk 前向取隐状态（只在每个 episode 的第一次决策）。"""
        B, T, H = len(hist), hist.shape[2], s.na.H
        s.vbuf = []
        try:
            p = s.rolling(hist, txt, tm, slots, fresh, kw)
            hid = s.vbuf[0][:B, :H * T].unflatten(1, (H, T))  # CFG 时 msroll 的 batch = [条件; 无条件]
        finally:
            s.vbuf = None
        if fresh.any():
            i = fresh.nonzero().squeeze(1)
            vt = s.idm.vpp_roll_tau(s.wm, s.wa, hist.device)
            t = vt[None, :, None, None]
            x = (t * p[i].float() + (1 - t) * torch.randn(len(i), *p.shape[1:], device=hist.device)).flatten(1, 2)
            hid = hid.clone()
            hid[i] = s.wm.trunk(x, vt[None].expand(len(i), -1), s.wm.context(hist[i], txt[i], tm[i]))[:, :H * T].unflatten(1, (H, T)).to(hid.dtype)
        s.vhid = hid
        return p

    def kf_cond(s, kfz):  # 条件增广训练的 c2l（train --kf_aug）：关键帧加固定的小噪声 τ_c（--kf_tau），τ_c 一起给 WM；J1 IDM 仍读原关键帧
        if kfz is None or not getattr(s.wa, "kf_aug", 0.0) > 0:
            return kfz
        c = s.a.kf_tau
        return ((1 - c) * kfz.float() + c * torch.randn(kfz.shape, device=kfz.device)).to(kfz.dtype), torch.full((len(kfz),), c, device=kfz.device)

    @torch.no_grad()
    def kf_plan(s, hist, txt, tm, kw, slots=None, fresh=None):
        """两级的关键帧（--kf_wm）→ (B,T,D)。--kf_guide：关键帧采样时加空间能量（make_guide，nf = 1）；
        --kf_select N：每个 env 采 N 个关键帧，取探针末端离定位目标最近的一个（无目标的 env 取第一个）。
        --kf_roll τ0：关键帧滚动细化——沿用上一次决策的关键帧，加噪回 τ0 再用 1 步去噪（以新历史为条件）；新 episode 或上一段执行里夹爪开合
        （关键帧被"到达"）时才按 --sampler 从头采。--pcons 时另记相邻决策关键帧的变化（kcons，只算没有从头采的决策）。"""
        kw = {k: v for k, v in kw.items() if k != "stagger"}
        roll = lambda h, t, m, g=None: s.fm.rollout(s.kwm, h, t, m, 1, s.kwa.mode, alpha=s.kwa.alpha, sigma=s.kwa.sigma, rs=s.krs, cfg=s.a.cfg,
                                                    guide=g, **kw)[0][:, 0]
        if s.kg is None and (s.a.kf_roll > 0 or s.a.pcons):
            B, dev = len(hist), hist.device
            if s.kfp is None:
                s.kfp, s.kgr, s.kcs = (torch.zeros(s.a.n_env, *hist.shape[2:], device=dev), torch.zeros(s.a.n_env, 2, device=dev),
                                       torch.zeros(s.a.n_env, 3, device=dev))
            redo = fresh | (s.kgr[slots, 0] != s.kgr[slots, 1])  # 上一段执行里夹爪开合 = 关键帧被"到达"
            if s.a.kf_roll > 0:
                kv = s.kwm.context(hist, txt, tm)
                kvn = s.kwm.context(hist, txt, tm, drop=torch.ones(B, dtype=torch.bool, device=dev)) if s.a.cfg != 1.0 else None
                z = s.fm.sample(s.kwm, kv, s.kfp[slots], s.kwa.mode, steps=1, alpha=s.kwa.alpha, tau0=s.a.kf_roll, sigma=s.kwa.sigma, rs=s.krs,
                                cfg=s.a.cfg, kv_null=kvn)[0].float()
                if redo.any():
                    i = redo.nonzero().squeeze(1)
                    z[i] = roll(hist[i], txt[i], tm[i]).float()
            else:
                z = roll(hist, txt, tm).float()
            e = ((z - s.kfp[slots]) / s.krs).pow(2).mean((1, 2))
            s.kcs[slots] = torch.where(fresh[:, None], 0.0, s.kcs[slots] + torch.stack([e * ~redo, (~redo).float(), redo.float()], 1))
            s.kfp[slots] = z
            return z.to(hist.dtype)
        if s.kg is None:
            return roll(hist, txt, tm).to(hist.dtype)
        g, ok = s.cur_goal
        if s.a.kf_select > 1:
            B, N = len(hist), s.a.kf_select
            r = lambda x: x.repeat_interleave(N, 0)
            kc = roll(r(hist), r(txt), r(tm))
            d = ((s.uvp(kc).float().view(B, N, 2) - g[:, None]) ** 2).sum(-1) * ok[:, None]
            return kc.view(B, N, *kc.shape[1:])[torch.arange(B, device=kc.device), d.argmin(1)].to(hist.dtype)
        return roll(hist, txt, tm, s.ot.make_guide(s.uvp, g, ok, s.kg[0], 1, hist.shape[2], s.kg[1], s.kg[2])).to(hist.dtype)

    @torch.no_grad()
    def objin(s, cov, z, sl, ti):
        """cov (B,7,256) uint8 oracle 覆盖率（掩码头时为 None），z (B,T,D) 当前帧（标准化），sl (B,n_slot) 各槽位的类别，ti 指令行 → WM 的物体 token 输入。
        掩码头（objtok.MaskHead）出 7 个类别的覆盖率再按 sl 取；定位头（GroundHead）直接按指令出 (操作物体, 目标) 两张热图。"""
        if s.mhead is None:
            c = cov.float() / 255
        elif s.mhead.kind == "ground":
            c, sl = s.mhead(z, s.temb[ti], s.tmask[ti]), torch.arange(2, device=z.device).expand(len(z), 2)
        else:
            c = s.mhead(z)
        return s.objtok.tokens(c.gather(1, sl[..., None].expand(-1, -1, c.shape[-1])), z)

    @torch.no_grad()
    def sguide(s, cov, z, sl, ti, grip):
        """空间能量引导的 guide 函数：目标 = 定位热图（头 / oracle）通道 0（操作物体）或 1（放置目标，grip 为真时）的质心。"""
        c = s.sgh(z, s.temb[ti], s.tmask[ti]) if s.sgh is not None else (cov.float() / 255).gather(1, sl[..., None].expand(-1, -1, cov.shape[-1]))
        g, ok = s.ot.heat_goal(c.float())
        k, i = grip.long(), torch.arange(len(g), device=g.device)
        s.cur_goal = (g[i, k], ok[i, k].float())  # --select eef 用
        if getattr(s.wa, "where", False):  # where 热图：目标质心处的高斯（σ 同训练）；无目标时全 0
            gg = (torch.arange(16, device=z.device, dtype=torch.float) + 0.5) / 16 * 2 - 1
            gy, gx = gg[:, None].expand(16, 16).reshape(-1), gg[None].expand(16, 16).reshape(-1)
            g, ok = s.cur_goal
            s.cur_heat = torch.exp(-((gx - g[:, :1]) ** 2 + (gy - g[:, 1:]) ** 2) / (2 * (getattr(s.wa, "where_sigma", 1.0) * 2 / 16) ** 2)) * ok[:, None]
        return s.ot.make_guide(s.uvp, *s.cur_goal, s.sg[0], s.sg[1], z.shape[1], s.sg[2], s.sg[3]) if s.sg[0] > 0 else None

    def uwm_init(s):
        """(uwm) --wm_act：train --uwm 的 WM 用自己的动作 token 出动作（probwm/uwm.py）。本体历史用 IDM 的配置（同 idmPL：qpos t−3..t），解码用 IDM 的 astats（须与 WM 里存的相同）。
        pre：Policy.rolling 换成 uwm.roll_pre（画面这一步与动作第 1 步共用一次前向）；clean：rolling 不变，act 里另算。"""
        from . import uwm
        a, wa = s.a, s.wa
        s.wact, s.uwm, s.wmix = a.wm_act, uwm, getattr(a, "wm_act_mix", 1.0)
        assert hasattr(getattr(s, "wm", None), "uwm") and a.roll and getattr(wa, "tgt_offs", "") and s.npf and s.npf * s.na.pdim == s.wm.uwm.pmu.numel() \
            and torch.allclose(s.st["mu"].float(), s.wm.uwm.amu.to(s.st["mu"].device)) and torch.allclose(s.st["sd"].float(), s.wm.uwm.asd.to(s.st["sd"].device)) \
            and not (getattr(a, "idm_tau", "") or getattr(a, "idm_stale", "") or getattr(a, "prog", "") or getattr(a, "abs_far", False) or getattr(a, "replan", "")
                     or getattr(a, "cfg_far", -1.0) >= 0 or getattr(a, "far_sep", False) or getattr(a, "near_only", False) or a.kf_wm or a.stagger or a.plan != 1
                     or getattr(wa, "canon_slot", False) or getattr(wa, "self_hist", "") or getattr(wa, "ev_slot", 0) or getattr(wa, "any_far", "")), \
            "--wm_act 需要 train --uwm 的 WM、带本体的 IDM（统计量相同），只接 --roll 的 MS-far msroll 路径"
        s.uwm_fn = None
        if getattr(a, "graphs", False):  # --graphs：动作组的联合前向 / 只算动作组的前向也按输入形状捕获 CUDA graph（随机数在外面抽，结果逐位不变）
            from .cgraph import Graphed
            s.uwm_fn = (Graphed(uwm.trunk, s.gpool), Graphed(uwm.vel_a, s.gpool))
        if s.wact == "pre":
            def pre(hist, txt, tm, slots, fresh, kw, kf=None):
                p, s.uwm_wo = uwm.roll_pre(s, hist, txt, tm, slots, fresh, kw, s.uwm_ph)
                return p
            s.rolling = pre
        print(f"wm_act {s.wact}：WM 动作 token {s.wm.uwm.H} 个，{a.wm_act_steps} 步；" + ("只用 WM 的动作" if s.wmix >= 1 else f"与 IDM 平均（WM 权重 {s.wmix}）"), flush=True)

    @torch.no_grad()
    def act(s, hist, ti, slots=None, fresh=None, wh=None, ph=None, hc=None, rec=None, obj=None, guide=None):  # obj：物体 token 输入（objin）；guide：--sguide  # rec (B,) bool：--recover cfg / second 时这些 env 重新出计划  # hc：--wm_raw 时 IDM 用的规范化历史  # hist (B,K,T,D)，ti (B,) tasks.json 下标，wh (B,nw,T,D) 腕部，ph (B,npf,dp) 本体感觉 → (B,n_act,7) numpy
        txt, tm = s.temb[ti], s.tmask[ti]
        s.uwm_ph = ph  # (uwm) --wm_act pre 的 rolling 要用本体
        with torch.autocast("cuda", torch.bfloat16):
            if s.a.policy == "bc":
                out = torch.stack([s.net(hist, txt, tm, zw=wh, zp=ph).float() for _ in range(getattr(s.a, "n_avg", None) or 1)]).mean(0)  # --n_avg K：同一输入采 K 个动作块取均值（默认 1）
            elif s.a.policy == "foresight":
                F = s.fsample(s.wm, s.wa, s.rs, hist, txt, tm, s.na.fo, s.a.sampler, s.a.cfg) if s.use_f else \
                    hist.new_zeros(len(hist), len(s.na.fo), *hist.shape[2:])
                out = s.net(torch.cat([hist, F.to(hist.dtype)], 1), txt, tm)
            else:
                if s.a.crn:  # 公共随机数：每次决策用同一组噪声，避免相邻决策在不同模式间跳
                    torch.manual_seed(s.a.seed)
                m, kw = s.a.plan, s.idm.SAMPLERS[s.a.sampler]
                if getattr(s.a, "det_cfg", False):  # (ctrl) --mode det 的 WM：同步路径（不 --roll）也做 CFG（fm.sample det_cfg）
                    assert s.wa.mode == "det" and not s.a.roll, "--det_cfg 只给 --mode det 的 WM、同步路径（不加 --roll）"
                    kw = {**kw, "det_cfg": True}
                if s.a.stagger:  # 交错去噪（WM 需 train --slot_noise）："L" 或 "L,步数"（步数覆盖采样器的）
                    st = [float(x) for x in s.a.stagger.split(",")]
                    kw = {**kw, "stagger": (st[0], s.wm.n_fut), **({"steps": int(st[1])} if len(st) > 1 else {})}
                if getattr(s.a, "wm_canon", False):  # (pro 11-17 B3) 两遍推理的第 1 遍：空历史、只喂当前观测帧，取 t+0 槽 = 当前帧的规范化版本，存进规范化历史；
                    # 第 2 遍（下面照常）WM 的上下文与 IDM 的历史都用规范化历史。第 1 遍从不读自己的输出 → 没有自回馈（不同于 --canon_hist）
                    c1 = torch.cat([torch.zeros_like(hist[:, :-1]), hist[:, -1:]], 1)
                    wf_ = getattr(s.a, "wm_canon_fast", "")
                    crn_, cf_ = getattr(s.a, "wm_canon_crn", False), getattr(s.a, "wm_canon_cfg", -1.0)
                    cf_ = s.a.cfg if cf_ is None or cf_ < 0 else cf_
                    if wf_ or crn_ or cf_ != s.a.cfg:  # 快路径（前 ns 个槽、steps 步）/ (pro 11-17) 固定噪声 / 第 1 遍单独的 CFG
                        ns_, st_ = (int(x) for x in wf_.split(",")) if wf_ else (s.wm.n_fut, kw.get("steps", 4))
                        x0_ = {}
                        if crn_:  # 每个 env 一份初始噪声、整集共用（新 episode 重抽）：相邻决策的第 1 遍误差相关，规范化历史的帧间抖动小（离线差分误差 .119 → .073）
                            Tm_ = s.wm.pos.shape[1]
                            if getattr(s, "wcn", None) is None or s.wcn.shape[1] != ns_ * Tm_:
                                s.wcn = torch.randn(s.a.n_env, ns_ * Tm_, hist.shape[3], device=hist.device)
                            slc = slots if slots is not None else torch.arange(len(hist), device=hist.device)
                            frc = fresh if fresh is not None else torch.zeros(len(hist), dtype=torch.bool, device=hist.device)
                            if frc.any():
                                s.wcn[slc[frc]] = torch.randn(int(frc.sum()), ns_ * Tm_, hist.shape[3], device=hist.device)
                            x0_ = {"x0": s.wcn[slc]}
                        t0 = wm_t0(s.fm, s.wm, c1, txt, tm, s.wa.mode, ns_, alpha=s.wa.alpha, sigma=s.wa.sigma, rs=s.rs, cfg=cf_, **{**kw, "steps": st_, **x0_})[:, :hist.shape[2]].to(hist.dtype)
                    else:
                        t0 = s.fm.rollout(s.wm, c1, txt, tm, s.wm.n_fut, s.wa.mode, alpha=s.wa.alpha, sigma=s.wa.sigma, rs=s.rs, cfg=s.a.cfg, **kw)[0][:, 0, :hist.shape[2]].to(hist.dtype)
                    if getattr(s, "wch", None) is None:
                        s.wch = torch.zeros(s.a.n_env, *hist.shape[1:], dtype=hist.dtype, device=hist.device)
                    sl_ = slots if slots is not None else torch.arange(len(hist), device=hist.device)
                    fr_ = fresh if fresh is not None else torch.zeros(len(hist), dtype=torch.bool, device=hist.device)
                    h_ = torch.cat([s.wch[sl_, 1:], t0[:, None]], 1)
                    h_ = torch.where(fr_[:, None, None, None], t0[:, None].expand_as(h_), h_)  # 新 episode：历史全用第一帧（同主循环）
                    s.wch[sl_] = h_
                    hist = h_
                ptau = None  # (xden) 噪声感知 IDM 读的计划各槽层级（--idm_tau pre）
                if s.a.roll and (getattr(s.a, "idm_tau", "") in ("pre", "stale") or s.stl):  # (xden) IDM 读 WM 去噪前的窗口（stale：上一次的干净估计）；WM 这一步在动作发出之后（roll_commit）
                    p, ptau = s.rolling_pre(s.ch_hist(hist, txt, tm, slots, fresh, kw) if s.chh else hist, txt, tm, slots, fresh, kw)  # (xden2) --canon_hist：WM 读 t+0 输出组成的历史
                elif s.a.roll and s.vpp:  # (vpp) 同 rolling，另取 msroll 那一步的近槽隐状态（s.vhid）
                    p = s.rolling_vpp(hist, txt, tm, slots, fresh, kw)
                elif s.a.roll:
                    kfz = s.kf_plan(hist, txt, tm, kw, slots, fresh) if s.kwm is not None else None  # 两级：关键帧照常（或 --kf_roll 细化），chunk 滚动
                    hu_ = s.ch_hist(hist, txt, tm, slots, fresh, kw) if s.chh else hist
                    p = s.rolling(hu_, txt, tm, slots, fresh, kw, kfz)
                    if s.ewm:  # (rteasy) --wm_ens：另几个 WM 换进来各自滚动（各自的缓存），计划按槽平均
                        assert not (getattr(s.a, "roll_k", 0) or 0) > 1 and s.kwm is None, "--wm_ens 不与 --roll_k / 两级同用"
                        ps, keep = [p.float()], (s.wm, s.wa, s.rw, s.rv)
                        for e in s.ewm:
                            s.wm, s.wa, s.rw, s.rv = e
                            ps.append(s.rolling(hu_, txt, tm, slots, fresh, kw, kfz).float())
                            e[2], e[3] = s.rw, s.rv
                        s.wm, s.wa, s.rw, s.rv = keep
                        p = torch.stack(ps).mean(0).to(p.dtype)
                elif s.a.sampler.startswith("rtc"):  # 上一次计划平移 m 帧后作热启动（新 episode 退化为重复 z_t）
                    n = s.wm.n_fut
                    z0 = hist[:, -1].repeat(1, n, 1)
                    if s.prev is None:
                        s.prev = torch.zeros(s.a.n_env, n, *hist.shape[2:], dtype=hist.dtype, device=hist.device)
                    pv = s.prev[slots]
                    sh = torch.cat([pv[:, m:], pv[:, -1:].expand(-1, min(m, n), -1, -1)], 1)[:, :n].flatten(1, 2)
                    z0 = torch.where(fresh[:, None, None], z0, sh)
                    kv = s.wm.context(hist, txt, tm)
                    kvn = s.wm.context(hist, txt, tm, drop=torch.ones(len(hist), dtype=torch.bool, device=hist.device)) \
                        if s.a.cfg != 1.0 else None
                    p = s.fm.sample(s.wm, kv, z0, s.wa.mode, alpha=s.wa.alpha, sigma=s.wa.sigma, rs=s.rs, cfg=s.a.cfg,
                                    kv_null=kvn, **kw)[0].unflatten(1, (n, -1))
                    s.prev[slots] = p.to(hist.dtype)
                elif s.a.n_cand > 1:
                    p = s.pick(hist, txt, tm, slots, fresh, kw)[:, :m + s.na.H - 1]
                elif s.comp is not None:
                    p = s.comp.plan(s, hist, txt, tm, ti, kw)[:, :m + s.na.H - 1]
                else:
                    sm = tuple(float(x) for x in s.a.smooth.split(",")) if s.a.smooth else (1, 0.0)
                    if s.ag:  # 辅助头引导：目标 = 当前帧的探针末端 + 辅助头预测的位移（dm → m）
                        pa = s.wm.context(hist, txt, tm, obj=obj, aux=True)[1].float().view(len(hist), -1, 3) / 10
                        base = s.p3(hist[:, -1]).float() * s.p3.sd + s.p3.mu
                        guide = s.ot.make_guide_xyz(s.p3, base[:, None] + pa, s.ag[0], s.ag[1], hist.shape[2], s.ag[2], s.ag[3])
                    kfz = s.kf_plan(hist, txt, tm, kw, slots, fresh) if s.kwm is not None else None
                    if s.prim is not None and s.kwm is not None:  # --prim + 两级：chunk WM 用原指令
                        txt, tm = s.temb0[s.prim.ti0], s.tmask0[s.prim.ti0]
                    p, _ = s.fm.rollout(s.wm, hist, txt, tm, 1 if getattr(s.wa, "keyframe", False) else s.wm.n_fut if getattr(s.wa, "tgt_off", 1) > 1 or getattr(s.wa, "tgt_stride", 1) > 1 or getattr(s.wa, "tgt_offs", "") or getattr(s.wa, "joint_kf", False) else m + s.na.H - 1,
                                        s.wa.mode, obj=obj, hcfg=s.hcfg, guide=guide, kf=s.kf_cond(kfz), kfcfg=s.a.kf_cfg,
                                        heat=getattr(s, "cur_heat", None) if getattr(s.wa, "where", False) else None, alpha=s.wa.alpha, sigma=s.wa.sigma,
                                        rs=s.rs, cfg=s.a.cfg, smooth=(int(sm[0]), sm[1]), **kw,
                                        **({"demo": s.sh_demo} if getattr(s, "sh_demo", None) is not None else {}))  # (mika MSFH-DET) --self_hist 的 det WM 走同步路径：历史 token（None = 原行为）
                    if rec is not None and rec.any():  # 惊讶触发的恢复：换新噪声 + 更高 CFG，或 K 个候选里的第二名
                        i, kind, arg = rec.nonzero().squeeze(1), s.rv[0], s.rv[4]
                        p[i] = s.fm.rollout(s.wm, hist[i], txt[i], tm[i], m + s.na.H - 1, s.wa.mode, obj=None if obj is None else obj[i], alpha=s.wa.alpha, sigma=s.wa.sigma,
                                            rs=s.rs, cfg=arg, smooth=(int(sm[0]), sm[1]), **kw)[0] if kind == "cfg" else \
                            s.pick(hist[i], txt[i], tm[i], slots[i], fresh[i], kw, C=int(arg), rank=1)[:, :m + s.na.H - 1].to(p.dtype)
                if s.cslot and getattr(s.a, "t0_ignore", False):  # (gawm X2F5CU 精度模式) t+0 槽不用来替换观测：只从计划里去掉，IDM 历史照旧（开 --canon 时是独立 canon 的输出）
                    p, ptau = p[:, 1:], None if ptau is None else ptau[:, 1:]
                elif s.cslot:  # (xden) train --canon_slot：WM 的 t+0 槽 = 当前帧的规范化估计 → 写进 IDM 用的规范化历史（代替独立 canon），计划只留 t+1 起的槽
                    if s.chist is None:  # (pro 11-16) 只存预测视角的 token：train --tgt_tok 的多视角 WM 上下文帧带额外视角（腕部），t+0 槽只有预测视角；单视角时与原来相同
                        s.chist = torch.zeros(s.a.n_env, hist.shape[1], p.shape[2], hist.shape[3], dtype=hist.dtype, device=hist.device)
                    cur = p[:, 0, :hist.shape[2]].to(hist.dtype)
                    if s.stl == "obs":  # (xden2) --idm_stale obs：IDM 的当前帧用原始观测（这一步的 t+0 要等 WM，动作之后才有）；pred = 上一次决策对当前帧的预测（缓存平移后的 t+0 槽）
                        cur = torch.where(fresh[:, None, None], cur, hist[:, -1, :cur.shape[1]])
                    hc = torch.cat([s.chist[slots, 1:], cur[:, None]], 1)
                    hc = torch.where(fresh[:, None, None, None], cur[:, None].expand_as(hc), hc)  # 新 episode：历史全用第一帧（同主循环）
                    s.chist[slots], p = hc, p[:, 1:]
                    ptau = None if ptau is None else ptau[:, 1:]  # (xden2) --idm_stale：各槽层级同样去掉 t+0 槽
                if getattr(s.wa, "joint_kf", False):  # J3：第 0 槽是关键帧，其余是 chunk
                    kfz, p = p[:, 0].to(hist.dtype), p[:, 1:]
                if s.ptr is not None:  # --trace：计划全部槽（IDM 选槽之前）→ 末端 xyz（米），另读当前帧
                    with torch.autocast("cuda", torch.bfloat16):
                        q_ = torch.cat([hist[:, -1:], p[:, :, :hist.shape[2]].to(hist.dtype)], 1)
                        s.trx = (s.ptr(q_.flatten(0, 1)).float() * s.ptr.sd + s.ptr.mu).view(len(q_), q_.shape[1], 3).cpu().numpy()
                k = getattr(s.wa, "tgt_off", 1)
                if k > 1 and not s.idm.fars(s.na):  # 远点 WM（只预测 ẑ_{t+k}）+ 普通 IDM：z_t → ẑ_{t+k} 直线插值出 IDM 要的 H 帧（H=1 即折线 z_t + (ẑ − z_t)/k）；
                    assert m == 1 and s.na.H <= k, "远点 WM 只支持 --plan 1、H ≤ tgt_off 的 IDM"  # 目标条件 IDM（--far）直接用 ẑ_{t+k}
                    j = torch.arange(1, s.na.H + 1, device=p.device)[None, :, None, None].to(hist.dtype) / k
                    p = hist[:, -1:] + (p[:, :1].to(hist.dtype) - hist[:, -1:]) * j
                fo = s.idm.fars(s.na)
                if fo and (getattr(s.wa, "tgt_stride", 1) > 1 or getattr(s.wa, "tgt_offs", "")):  # 多目标帧的 WM（如 {t+1, t+8}：第 1 槽给 PoE 当下一帧专家）：目标条件 IDM 只取它要的帧
                    tt = [int(x) for x in s.wa.tgt_offs.split(",")] if getattr(s.wa, "tgt_offs", "") else [s.wa.tgt_off + s.wa.tgt_stride * j for j in range(s.wm.n_fut)]
                    p = p[:, [tt.index(f) for f in fo]]
                if getattr(s.a, "plan_copy", False):  # 对照：计划换成当前帧复制（= 不看计划；配 IDM --p_fdrop 训练的头即它自己的 BC 模式）
                    p = hist[:, -1:, :p.shape[2]].expand(-1, p.shape[1], -1, -1).to(p.dtype)
                if getattr(s, "orc", None):  # --oracle：有专家的 env，各槽换成专家第 d + 时距 帧的真实特征（越界取最后一帧；同 plandiag 的 D2）
                    of = [int(x) for x in s.wa.tgt_offs.split(",")] if getattr(s.wa, "tgt_offs", "") else list(range(1, p.shape[1] + 1))
                    for i, w in enumerate(slots.tolist()):
                        fe = s.orc.get(w)
                        if fe is not None:
                            p[i] = fe[torch.clamp(torch.tensor([s.orc_d[w] + o for o in of[:p.shape[1]]], device=fe.device), max=len(fe) - 1)].to(p.dtype)
                s.plan = p  # --surp：下一次决策拿来和新观测比
                if s.p3 is not None:  # --wp_exec：计划第 1 槽（t+1）→ 末端 xyz（米）
                    with torch.autocast("cuda", torch.bfloat16):
                        s.wp = (s.p3(p[:, 0].to(hist.dtype)).float() * s.p3.sd + s.p3.mu).cpu().numpy()
                if s.a.pcons:  # 计划一致性：本次计划与上一次计划在同一绝对时刻的帧上的差（按 rs 归一的均方，patch / 通道平均）；[全部重叠帧, 最近的重叠帧]
                    d = max(1, s.a.exec // 2)
                    of = [int(x) for x in s.wa.tgt_offs.split(",")] if getattr(s.wa, "tgt_offs", "") else list(range(1, p.shape[1] + 1))
                    pr = [(i, of.index(o + d)) for i, o in enumerate(of) if o + d in of]  # (本次槽, 上次槽)：上次的 o + d = 本次的 o
                    if s.pp is None:
                        s.pp, s.pcs = torch.zeros(s.a.n_env, *p.shape[1:], device=p.device), torch.zeros(s.a.n_env, 3, device=p.device)
                    if pr:
                        e = ((p[:, [i for i, _ in pr]].float() - s.pp[slots][:, [j for _, j in pr]]) / s.rs).pow(2).mean((2, 3))
                        s.pcs[slots] = torch.where(fresh[:, None], 0.0, s.pcs[slots] + torch.stack([e.mean(1), e[:, 0], torch.ones_like(e[:, 0])], 1))
                    s.pp[slots] = p.float()
                seq = torch.cat([(hist if hc is None else hc)[:, :, :p.shape[2]], p.to(hist.dtype)], 1)  # IDM 的历史帧：j=0 时是真实帧，之后接上预测帧；
                # WM 有只作上下文的额外视角（train --tgt_tok）时 IDM 只取预测视角的 token（额外视角走 --wrist 流）
                lt = (txt, tm) if s.na.lang else ()  # --lang IDM：与 WM 同一行指令嵌入（--para 时即改写）
                if s.wact:  # (uwm) WM 动作 token 的动作块（标准化空间）；--wm_act_mix ≥ 1 时下面那次 IDM 调用换成直接返回它的桩（不跑 IDM）
                    wo, net0 = s.uwm.act(s, p, ph, hist, txt, tm), s.net
                    if s.wmix >= 1:
                        s.net = s.uwm.Fixed(wo)
                nf = len(s.idm.fars(s.na)) or s.na.H + len(s.idm.fextra(s.na))  # 目标条件 IDM：未来输入 = WM 预测的远点 / 关键帧；--far_extra：近处 + 远槽
                xk = (lambda q: torch.cat([q, kfz[:, None].to(q.dtype)], 1)) if getattr(s.na, "kf_extra", False) else (lambda q: q)  # J1：再读预测关键帧
                assert not getattr(s.na, "kf_extra", False) or s.kwm is not None or getattr(s.wa, "joint_kf", False), "--kf_extra 的 IDM 需要 --kf_wm 或联合 WM"
                tk = {} if ptau is None else {"tau": ptau[:, :nf]}  # (xden) --idm_tau pre：计划各槽的层级（只支持 --plan 1）
                if s.vpp:  # (vpp) IDM 另读近槽隐状态；hid：帧输入只有历史（不读计划帧）
                    nf, tk = (0 if s.vpp == "hid" else nf), {**tk, "zh": s.vhid}
                if s.ms is not None:  # (rteasy 11-19) --idm_ms：主 IDM 在当前流，集成成员各自一条流并发，等权平均（同下面顺序版）
                    stale = bool(s.stl) or s.a.idm_tau == "stale"
                    assert not (s.wact or s.vpp or (tk and not stale)), "--idm_ms 不与 --wm_act / --vpp / --idm_tau pre 同用"
                    ins = [xk(seq[:, s.K - 1 - s.na.ih + j: s.K + j + nf]) for j in range(m)]  # 在分叉前算好（成员流只等分叉点之前的工作）
                    out = s.ens_ms(lambda n_: torch.cat([n_(x_, *lt, zw=wh, zp=ph, **tk) for x_ in ins], 1).float(), getattr(s.a, "n_avg", None) or 1)
                elif s.ens_par is not None:  # (rteasy 11-18) --idm_ens_par：主 IDM + 集成成员一次 vmap 前向，等权平均（同下面顺序版）
                    stale = bool(s.stl) or s.a.idm_tau == "stale"
                    assert not (s.wact or s.vpp or (tk and not stale)) and (getattr(s.a, "n_avg", None) or 1) == 1, "--idm_ens_par 不与 --wm_act / --vpp / --idm_tau pre / --n_avg 同用"
                    out = torch.cat([s.ens_par(xk(seq[:, s.K - 1 - s.na.ih + j: s.K + j + nf]), *lt, zw=wh, zp=ph, **tk) for j in range(m)], 2).float().mean(0)
                else:
                    out = torch.stack([torch.cat([s.net(xk(seq[:, s.K - 1 - s.na.ih + j: s.K + j + nf]), *lt, zw=wh, zp=ph, **tk) for j in range(m)], 1).float()
                                       for _ in range(getattr(s.a, "n_avg", None) or 1)]).mean(0)  # --n_avg K：同一计划下 IDM 采 K 个动作块取均值（默认 1，与原来相同）
                if s.enets and s.ens_par is None and s.ms is None:  # (rteasy) --idm_ens：同一计划、历史、本体下另几个 IDM 各采一块，与主 IDM 的块等权平均（标准化空间，统计量已核相同）
                    stale = bool(s.stl) or s.a.idm_tau == "stale"  # (gawm X2F5CU) --idm_stale / --idm_tau stale：计划是上一次的干净估计，各槽层级全 1（= 干净；没有 tauemb 的 IDM 本来就忽略）→ 可以集成
                    assert not (s.wact or s.vpp or (tk and not stale)), "--idm_ens 不与 --wm_act / --vpp / --idm_tau pre 同用"
                    out = torch.stack([out] + [torch.cat([en(xk(seq[:, s.K - 1 - s.na.ih + j: s.K + j + nf]), *lt, zw=wh, zp=ph, **tk) for j in range(m)], 1).float()
                                               for en in s.enets]).mean(0)
                if s.wact:  # (uwm) --wm_act_mix < 1：与 IDM 的块按权重平均（= (c)）
                    s.net = net0
                    out = out if s.wmix >= 1 else (1 - s.wmix) * out + s.wmix * wo.float()
        acts = s.idm.clip(s.idm.decode(out.float(), s.st), s.st)
        if getattr(s, "ev_c", None) is not None:  # train --ev_roll：这次要执行的最后一个夹爪指令与上次不同 = 事件到了，下次决策事件槽从纯噪声重来
            g = acts[:, min(s.a.exec, acts.shape[1]) - 1, 6] > 0
            s.ev_flip[slots], s.ev_g[slots] = (g != s.ev_g[slots]) & ~fresh, g
        if s.kfp is not None:  # 关键帧滚动细化：记最近两次决策执行的最后一个夹爪指令（判断"到达"）
            g = acts[:, min(s.a.exec, acts.shape[1]) - 1, 6].sign().to(s.kgr.dtype)
            s.kgr[slots] = torch.where(fresh[:, None], g[:, None].expand(-1, 2), torch.stack([g, s.kgr[slots, 0]], 1))
        if getattr(s, "_pend", None) is not None:  # (xden) --idm_tau pre：动作已拷回 CPU（帧 → 动作的关键路径到此为止），再做 WM 这一步、写回缓存给下一次决策
            out = acts.cpu().numpy()
            s.t_act = time.perf_counter()
            with torch.autocast("cuda", torch.bfloat16):
                s.roll_commit()
            return out
        return acts.cpu().numpy()  # OSC 动作截到 [-1, 1]；RoboTwin 的关节角不截


def parse_ids(x):
    return [i for p in x.split(",") for i in (range(int(p.split("-")[0]), int(p.split("-")[1]) + 1) if "-" in p else [int(p)])]


def summarize(eps, n_task_names):
    per = {}
    for e in eps:
        per.setdefault(e["task"], []).append(e["success"])
    rate = {k: float(np.mean(v)) for k, v in sorted(per.items())}
    for k, r in rate.items() if len(rate) <= 50 else ():
        print(f"  task {k} {n_task_names.get(k, '')[:45]:<45} {r:.3f} ({len(per[k])} ep)")
    print(f"  mean over tasks {np.mean(list(rate.values())):.3f}   episodes {len(eps)}")
    return rate


def main():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a("--policy", default="wm+idm", choices=["wm+idm", "bc", "foresight"])
    a("--fs", default="runs/goal/foresight/wm1.pt", help="foresight 策略 ckpt（probwm/foresight.py）")
    a("--fzero", action="store_true", help="foresight：F 置零（消融）")
    a("--wm", default="runs/goal/abs_a13/ckpt.pt", help="bc 也需要：norm、编码器、语言嵌入")
    a("--idm", default="runs/goal/idm/idm.pt"); a("--bc", default="runs/goal/idm/bc.pt")
    a("--sampler", default="scratch4"); a("--cfg", type=float, default=1.0)
    a("--cfg_far", type=float, default=-1.0, help="≥0：多尺度滚动（--roll + WM --tgt_offs）里远槽（时距 ≥ 9）的 CFG 单独用这个值，近槽仍用 --cfg；新 episode 的第一次同步采样仍用 --cfg")
    a("--suite", default="libero_goal"); a("--tasks", default="0-9", help="benchmark 任务编号，如 0,3 或 0-9")
    a("--n_ep", type=int, default=20); a("--ep0", type=int, default=0, help="从第几个初始状态开始")
    a("--shard", type=int, default=0); a("--nshard", type=int, default=1); a("--n_env", type=int, default=10)
    a("--max_steps", type=int, default=300); a("--wait", type=int, default=10); a("--exec", type=int, default=2)
    a("--img", type=int, default=256); a("--seed", type=int, default=0); a("--out", default="")
    a("--perturb", default="", help="OOD 扰动，如 camera:0.03,light:0.5,objpos:0.03（probwm/perturb.py）")
    a("--text", default="", help="换一份语言嵌入（如 perturb.py text 生成的改写指令）")
    a("--para", default="", help="job 清单 json（probwm/libero_para.py）：每个 job =（任务名, 初始状态, 嵌入行号, 标签），"
                                 "语言嵌入取清单里 emb 文件的那一行；替代 --tasks/--n_ep，并记录末端轨迹和各任务目标的达成时刻")
    a("--pro2", default="", help="(pro) LIBERO-Pro2.0 官方评测服务的调度 run_id（run_grouped_eval.py submit 的输出）：worker 换成 probwm/pro2_env.worker，"
      "每次 reset 向调度器领一个 episode、结果由服务端记录；指令按字符串现编文本嵌入（pro2_env.TextCache）；默认关")
    a("--pro2_url", default=os.environ.get("LIBERO_REMOTE_URL", "http://127.0.0.1:8000")); a("--pro2_key_env", default="LIBERO_REMOTE_API_KEY")
    a("--pro2_n", type=int, default=0, help="--pro2：run 里的 episode 数（主进程按它记总数）"); a("--pro2_runner", default="probwm")
    a("--pro2_paths", default="", help="--pro2：客户端（service-source）与其依赖目录，逗号分隔，只加进 worker 的 sys.path")
    a("--plus", default="", help="LIBERO-Plus job 清单 json（probwm/libero_plus.py）：job 格式同 --para，任务名为扰动任务名，"
                                 "清单另带初始状态（init.npz）；须给 --out")
    a("--grip_hold", type=int, default=1, help=">1：夹爪新指令要连续出现这么多次决策才切换（抑制采样带来的抖动）")
    a("--merge", nargs="*", help="只汇总这些结果 json")
    a("--plan", type=int, default=1, help=">1：WM 一次推演这么多步，IDM 逐步解码后开环执行（--exec 默认随之变为 2×plan）")
    a("--crn", action="store_true", help="每次决策用同一组随机数（抑制模式跳变）")
    a("--n_cand", type=int, default=1, help=">1：测试时分支选择，每次决策采这么多个 chunk（只支持 chunk 模型、--plan 1 的路径）")
    a("--purify", default="", help="τ0,K,steps,snr：观测净化（需 --prior 单帧先验 ckpt），如 0.6,5,4,0.15")
    a("--prior", default="runs/goal/prior/ckpt.pt")
    a("--canon", default="", help="特征规范化模型 ckpt（train --canon，probwm/canon.py）：每个新观测先规范化，再交给 WM / IDM")
    a("--canon_steps", type=int, default=4); a("--canon_tau0", type=float, default=0.0, help=">0：从加噪回 τ0 的观测出发（桥）")
    a("--wmpur", default="", help="τ0,β[,τ0s,n_lan,n_start,anchor]：WM 条件净化（递归时间滤波，见 Policy.wmpur），如 0.3,1 或 0.3,1,0.15,10,5,1")
    a("--wm_raw", action="store_true", help="规范目标 WM（train --canon_wm）：WM 读原始历史，--canon 只给 IDM 的历史")
    a("--wm_xtok_null", action="store_true", help="(pro) WM 上下文带腕部（train --tgt_tok）时把腕部 token 置 0（= 训练里 --xtok_drop 的缺省；不带腕部的同模型对照）")
    a("--cam_oracle", default="", choices=["", "wm", "idm", "both"], help="(pro) LIBERO-Plus 相机维的感知上限诊断：worker 每帧另渲染原始相机位姿下的画面（不规范化），"
      "wm = WM 的历史换成它（IDM 仍读规范化的扰动画面）、idm = 只换 IDM 的历史、both = 都换；动作仍在扰动相机的 env 里执行。默认关")
    a("--cam_route", type=int, default=0, help="(cam) 每集\"走 canon / 不走 canon\"二选一：前 N 次决策照常走 canon，同时缓存原始 / 规范化特征、本体、实际执行的动作；"
      "第 N 次决策时按自监督逆动力学（IDM 读窗口内真实转移，对已执行动作的 nmse，两路同一批采样噪声）选路，选原始的 env 把历史换成原始特征、"
      "滚动缓存作废（同新 episode 重采），之后不再过 canon；每集记 cam_route。需 --canon、IDM 带本体、--exec 2。默认 0 = 关")
    a("--poe", default="", help="w[,steps[,thr]]：规范化器与 WM 的专家乘积采样（见 Policy.poe；需 --canon、--prior）；thr：OOD 门控（数 = 位移阈值；ramp:a:b；var:K:a:b）")
    a("--soft_reset", action=argparse.BooleanOptionalAction, default=True,
      help="同任务的相邻 episode 复用 env：软 reset（不重建 MjSim / 渲染上下文），每次 reset 前 seed(0)，在 robot.reset 处重放硬 reset 时的 np.random 状态、夹爪累加指令清零；"
           "与硬 reset 逐位相同（11-07 核对 Goal 100 / PRO Pos 50 / Para 144 集：初图 hash、成功、步数逐集一致），单次 reset 3–4 s → 0.3 s。默认开（11-07 起），--no-soft_reset 关；"
           "--perturb 含 camera / light、--plus 不带 --plus_reseed 时自动退回硬 reset，--oracle 改过固定件的下一集也硬 reset")
    a("--plus_reseed", action="store_true", help="--plus 的 job（如 LIBERO-PRO）同任务重复时也在每次 reset 前 seed(0)（默认只在建 env 时 seed 一次，第 2 集起的固定件取决于 worker 的历史）")
    a("--hash0", action="store_true", help="每个 episode 记初始观测图像的 sha1 前 16 位（img0），核对两种运行方式的初态是否逐集相同")
    a("--async_reset", action="store_true", help="worker reset（建 env 要在本机全局排队，envlock）时不让整批停下来等：其余 env 照常推进，reset 完的下一次决策再加入（结果分布不变，批组成会变）")
    a("--graphs", action="store_true", help="编码器 / WM / IDM 速度场 / canon / prior 的前向按输入形状捕获 CUDA graph 重放（随机数照常 eager 抽，结果逐位不变；B=1 省 kernel 启动开销）")
    a("--poe_gsteps", type=int, default=4, help="--poe 门控：判门用的那次规范化的 Euler 步数（默认 4，与原实现相同）")
    a("--poe_lazy", action="store_true", help="--poe：整批 WM 权重为 0 时跳过 WM / prior 两个专家（结果与默认逐位相同，只省计算）")
    a("--poe_trace", action="store_true", help="--poe 门控：逐决策步记录 (规范化位移, 规范化样本方差) 到结果 json 的 poe_tr")
    a("--poe_cond", default="full", choices=["full", "nolang", "shuf", "none"], help="--poe 的 WM 项条件（消融）：full 历史 + 指令 / nolang 只有历史 / shuf 随机别的指令 / none 无历史无指令")
    a("--pfilt", default="", help="κ,s,pool,steps,τ0[,cfg]：WM 条件后验净化（probwm/postfilt.py；第一帧用 --prior），如 1,0.3,2,8,0")
    a("--smooth", default="", help="M,σ：平滑引导，条件速度对 M 份加噪（相对 RMS σ）的文本嵌入取平均")
    a("--prim", default="", help="原语计划 runs/pr/plans_<集合>.json（probwm/prim.py plan）"); a("--prim_det", default="runs/pr/det.pt")
    a("--select", default="dc", help="dc | medoid | prev | eef（需 --sguide 0），可用 + 组合，如 dc+prev"); a("--dc_t", type=int, default=4)
    a("--ens", type=float, default=None, help="m：ACT 式时间集成（权重 exp(−m·i)，i=0 为最早的块）；不给 = 只用最新的块")
    a("--fsteps", type=int, default=0, help="flow 头 IDM 的 Euler 步数（0 = ckpt 里的）")
    a("--roll_k", type=int, default=0, help="(rteasy) >1：MS-far 滚动每次决策批量采 K 个计划，按远槽特征选一个（--roll_sel）交给 IDM、写回缓存；默认关")
    a("--roll_kc", type=int, default=0, help="(rteasy) --roll_k 时每次 WM 前向只采这么多份计划（分块省显存；0 = K 份一次）")
    a("--roll_sel", default="medoid", choices=["medoid", "cluster"], help="(rteasy) --roll_k 的选法：medoid = 到其余计划平均距离最小；cluster = 最大簇（距离小于中位数的邻居最多）")
    a("--goal_first", action="store_true", help="(rteasy 臂 G) MS-far 滚动改成目标先行两阶段：先远槽（近槽纯噪声）、再近槽（远槽钉住）；配 train --goal_first 的 WM；默认关")
    a("--gf_fresh", type=int, default=4, help="(rteasy) --goal_first / --lead_k 时新 episode 从纯噪声走的 Euler 步数")
    a("--lead_k", type=int, default=0, help="(rteasy 臂 G2) >0：MS-far 滚动每次决策走这么多步、远槽进度领先（配 train --lead 的 WM）；默认 0 = 关")
    a("--wm_ens", default="", help="(rteasy) 逗号分隔的另几个 WM ckpt（同特征 / norm / 时距槽）：各自滚动、各自缓存，IDM 读的计划按槽平均；默认关")
    a("--idm_ens", default="", help="(rteasy) 逗号分隔的另几个 IDM ckpt（同配方不同训练种子）：各自采样后与 --idm 的动作块等权平均；默认关")
    a("--idm_graph", action="store_true", help="(rteasy 11-18) IDM 骨干（flow 头之前）按输入形状捕获 CUDA graph（flow 头的 vel 另由 --graphs 捕获）；默认关")
    a("--compile", default="", help="torch.compile 模式：default / max-autotune-no-cudagraphs（配 --graphs 时不要用带 inductor cudagraphs 的 max-autotune）；默认空 = 关。"
      "结果不逐位相同（差在 bf16 数值噪声量级）；首次决策要编译，每个新输入形状各编一次")
    a("--compile_on", default="wm,canon,idm,enc", help="--compile 作用的模块：wm = 上下文编码与速度场，canon，idm = flow 头 vel 与骨干 ctx（骨干要 --idm_graph），enc = 视觉编码器")
    a("--idm_ens_par", action="store_true", help="(rteasy 11-18) --idm_ens 的并行版：主 IDM 与成员参数叠起来 vmap 一次前向（各成员噪声独立）；默认关 = 顺序版")
    a("--idm_ms", action="store_true", help="(rteasy 11-19) --idm_ens 的多流版：集成成员各占一条 CUDA stream 与主 IDM 并发（成员图各用独立显存池；与顺序版逐位一致）；默认关")
    a("--obs_sync", action="store_true", help="回传当前仿真状态的观测（与训练渲染一致），见文件头")
    a("--comp", default="", help="组合式指令引导清单 runs/comp/<set>.json（probwm/compose.py build）")
    a("--comp_w", default="3,0,0", help="w_f,w_p,w_n[,wm]：完整指令 / 每个子短语 / 负向（最近训练任务；wm = WM 在线选）的权重")
    a("--hcfg", default="", help="w_h,w_t：历史 / 文本两路分开的引导（替代 --cfg；WM 需 train --p_hdrop），如 2,1")
    a("--sguide", default="", help="w[,nf[,tmax[,jac]]]：空间能量引导（见文件头），如 0.1,4,0.1,jac（tmax：只在 τ ≤ tmax 的步引导；jac：梯度穿过 WM）"); a("--sguide_src", default="runs/goal/obj/ghead_scn.pt", help="定位头 ckpt 或 oracle")
    a("--eefuv", default="runs/goal/obj/eefuv.pt", help="末端图像位置探针（objtok eefuv）")
    a("--aux_guide", default="", help="w[,nf[,tmax[,jac]]]：辅助末端轨迹头引导（WM 需 train --aux_eef；选项同 --sguide）"); a("--eefxyz", default="runs/goal/obj/eefxyz.pt")
    a("--roll", default="", help="滚动去噪 k[,δ[,lan[,grow]]]（fm.roll）：每次决策 k 个 Euler 步、回噪 δ（均匀时间 u 上）、lan 步 Langevin、grow = 1 时回噪随槽距离变大；新 episode 用 --sampler")
    a("--oracle", default="", help="目录：plandiag --save_expert 存的专家；有专家的 episode 换到专家的场景与起点，计划换成专家的真实未来特征（规划 / 执行拆分，默认关）")
    a("--wp_exec", default="", choices=["", "geo", "geo_rot", "mix"], help="可部署的相对末端航点执行：探针（--wp_probe）把 WM 计划的 t+1 槽解成末端位置，动作改成指向它的比例控制（模式同 --oracle_exec，夹爪用 IDM；需本体感觉 IDM）")
    a("--wp_probe", default="assets/eefxyz_lg2.pt", help="--wp_exec 的世界坐标末端探针")
    a("--oracle_exec", default="", choices=["", "geo", "geo_rot", "mix"], help="配 --oracle：执行侧上界——动作改成指向专家下一帧末端位置的比例控制（geo：不转腕、夹爪照专家；geo_rot：转动用 IDM；mix：IDM 平移与航点各一半，转动 / 夹爪用 IDM）")
    a("--oracle_wm", action="store_true", help="配 --oracle：只换到专家的场景与起点，计划仍用 WM（同场景下的 WM 对照）")
    a("--oracle_pin", action="store_true", help="(pro) 配 --oracle_far：注入的专家远槽钉住（层级 u = 1，不再加噪、不动），近槽照常滚动；B2 系训练剖面单调（近槽不比远槽噪），这是分布外状态")
    a("--roll_us", type=float, default=-1.0, help="(pro) ≥0：MS-far 滚动时有缓存的槽全部加噪回同一层级 u0（均匀时间；训练里同步样本的剖面），各自走到干净；配 --oracle_far 即"
      "分布内的远槽 oracle（远槽注入的专家帧加噪到 τ = shift(u0, α)）。默认 −1 = 原滚动剖面")
    a("--oracle_far", action="store_true", help="(pro) 配 --oracle：部分 oracle——只把滚动缓存里远槽（时距 ≥ 9）的估计换成按步数对齐的专家帧，"
      "近槽仍由 WM 联合去噪（同 robotwin_eval --oracle_plan tf，Policy.orc_far；只接 --roll 的稀疏时距 WM）；默认关")
    a("--prog", default="", choices=["", "P1", "P2", "P3"], help="MS-far 滚动的渐进去噪变体（codex 渐进臂的推理，见 Policy.rolling_x）；默认不用")
    a("--abs_far", action="store_true", help="MS-far 滚动：远槽锁定绝对时刻（codex 绝对时刻臂 T1 的推理，见 Policy.rolling_x）")
    a("--pcons", action="store_true", help="记录计划一致性（相邻决策的计划在重叠帧上的差，见 Policy.act），写进 episode 的 pcons / pcons1")
    a("--far_sep", action="store_true", help="(objslot) WM 的远槽是物体 slot 帧（train --far_slots）：多尺度滚动的绝对时间缓存里，远槽的估计不滑进近槽、近槽的不借给远槽")
    a("--stagger", default="", help="交错去噪：远槽领先、近槽滞后 L（0 = 同步），可写 L,步数；WM 需 train --slot_noise")
    a("--kf_cfg", type=float, default=None, help="两级：关键帧引导强度 w_k（u += (w_k − 1)(u_有关键帧 − u_关键帧置 0)；不给 = 不额外引导）")
    a("--kf_guide", default="", help="两级：只引导关键帧 w[,tmax[,jac]]（空间能量，目标同 --sguide；需 --kf_wm 和 --sguide 0）")
    a("--kf_tau", type=float, default=0.1, help="条件增广训练的 c2l（train --kf_aug）：闭环时关键帧条件的固定噪声层级 τ_c")
    a("--kf_select", type=int, default=1, help="两级：每次采这么多个关键帧，取探针末端离定位目标最近的一个（需 --kf_wm 和 --sguide 0）")
    a("--kf_roll", type=float, default=0.0, help="两级：关键帧滚动细化——沿用上一次的关键帧，加噪回这个 τ0 再 1 步去噪；夹爪开合（到达关键帧）时才从头采；0 = 每次从头采")
    a("--kf_wm", default="", help="两级：关键帧 WM ckpt（train --keyframe），主 WM 须为 train --kfcond")
    a("--where_src", default="oracle", help="where WM（train --where）的热图来源：oracle = 仿真分割 + bddl 目标类别（特权上限）；否则为定位头 ckpt")
    a("--obj_src", default="oracle", help="物体 token WM：oracle = 仿真分割掩码；否则为掩码头 ckpt（probwm/objtok.py）")
    a("--surp", action="store_true", help="记录每次决策的惊讶 / 运动（见文件头）")
    a("--recover", default="", help="kind,N,s_hi,m_lo[,arg[,R]]：惊讶 / 卡住触发的恢复（见文件头）")
    a("--trace", default="", help="(scale) 路径：逐 episode 记录末端 / 夹爪 / 各物体 body 位置（每次决策后）与计划各槽的探针末端（--wp_probe），torch.save 到这里；默认关")
    a("--idm_tau", default="", choices=["", "pre", "stale"], help="(xden) pre：噪声感知 IDM（idm --tau_in）直接读 MS-far 滚动 WM 去噪前的窗口（各槽带层级 τ），"
                                                          "WM 这一步排到动作发出之后（Policy.rolling_pre / roll_commit）；stale：任意 IDM 读上一次决策的干净估计（平移一帧，τ = 1），"
                                                          "WM 步同样在动作之后（流水线对照）；需 --roll 与 --tgt_offs 的 WM；默认关")
    a("--t0_sde", type=float, default=-1.0, help="(rteasy 11-17) ≥0：canon_slot WM 的 t+0 槽从当前观测加噪到 τ0（混合系数，τ = 1 数据端）起去噪（SDEdit 式投影），代替缓存里上次对当前帧的估计；只接默认 MS-far 滚动路径；默认关")
    a("--wm_canon", action="store_true", help="(pro 11-17 B3) 两遍推理：第 1 遍空历史、只喂当前观测帧，取 WM 的 t+0 槽作当前帧的规范化版本存进规范化历史；"
                                              "第 2 遍 WM 规划与 IDM 都读规范化历史（WM 宜 train --hist_null）。须配 --t0_ignore；默认关")
    a("--wm_canon_crn", action="store_true", help="(pro 11-17) --wm_canon 的第 1 遍用固定初始噪声：每个 env 一份、整集共用（新 episode 重抽），相邻决策的规范化误差相关、历史的帧间抖动小；默认关")
    a("--wm_canon_cfg", type=float, default=-1.0, help="(pro 11-17) --wm_canon 第 1 遍单独的 CFG（≥0 时覆盖 --cfg，第 2 遍规划仍用 --cfg）；默认 −1 = 同 --cfg")
    a("--wm_canon_fast", default="", help="(pro 11-17 B3) 'ns,steps'：--wm_canon 的第 1 遍只采前 ns 个目标槽（t+0 = 第 0 槽；1 = 只查 t+0）、steps 步（覆盖采样器的步数）；默认空 = 完整 n_fut 槽")
    a("--t0_ignore", action="store_true", help="(gawm X2F5CU 精度模式) --canon_slot 的 WM：t+0 槽只从计划里去掉，不用来替换 IDM 历史里的当前帧（配 --canon 即独立 canon + 不 stale）；默认关")
    a("--wm_data", default="", help="(gawm X2F5CU) WM 的训练数据根改用这个本机路径（info.json / 文本表 / meta/tasks.json；内容须与训练根相同，可指向迁移后的数据根）；默认空 = 用 ckpt 里的 --data")
    a("--idm_stale", default="", choices=["", "pred", "obs"], help="(xden2) (A′) 单独的开关，可与 --canon_slot / --canon_hist 的 WM 叠加：IDM 读上一次决策的干净计划（平移一帧，同 --idm_tau stale），"
                                                                "动作发出后再做 WM 的上下文 + 1 步滚动并写回缓存（新 episode 第一次同步串行）。canon_slot 的 WM 这一步的 t+0 输出也在动作之后才有，"
                                                                "IDM 规范化历史里的当前帧：pred = 上一次决策对当前帧的预测（WM 完全不在帧 → 动作的关键路径上），obs = 原始观测；"
                                                                "动作之后换成真正的 t+0 输出（之后各步的历史用它）。别的 WM 上 pred / obs 相同（= --idm_tau stale）；默认关")
    a("--replan", default="", help="(hz2) thr[,mode[,cool[,stat]]]：MS-far 滚动的惊讶触发重规划（Policy.replan_check）——上一次决策对当前帧的估计与观测的偏差 > thr 时，"
                                   "这个 env 的滚动缓存作废、同步从头重采（mode all = 全部槽，far = 只换远槽），之后 cool 次决策（默认 8）不再触发；stat = mean | top | cos；"
                                   "thr = inf 只记录不触发；默认关（RoboTwin 由 robotwin_eval --replan 传入，逐决策的统计量 / 触发写进 episode）")
    a("--ov_log", action="store_true", help="(pro) 非 --pro2 的评测也逐决策记观测失效旗标（ov_dup / ov_blank，只记不替换）：在主进程按决策帧算——"
                                            "图像与上一决策帧逐字节相同而这段不变期间末端累计移动 > 1 mm，或与上一帧不同、却与之前 12 个决策帧之一相同 → 陈旧；RGB 全零 → 黑屏。"
                                            "需要策略带本体感觉（末端位置）。看 OV-wm 的检测器在干净 / 别的扰动数据上会不会误触发；默认关")
    a("--ov", default="", help="(pro) mode[,rules[,thr_hi[,thr_lo[,cap]]]]：LIBERO-Pro2.0 观测失效时用伪观测顶上（Policy.ov_sub，预注册 logs/pro.md 11-15 04:40）——"
                               "不可信 = R1 陈旧帧 / R2 黑屏（worker 回传的旗标）/ R3 极端惊讶（滞回 thr_hi 进、thr_lo 出），rules 如 12、123、3；连续不可信满 cap 次（默认 30）强制信一次；"
                               "mode wm = WM 对当前帧的干净估计，hold = 最近一次可信的真实观测。只接 --pro2；默认关（旗标照样逐决策记进 episode 的 ov_dup / ov_blank）。"
                               "OV-soft（预注册 logs/pro.md 11-15 16:50）：soft,θ,s = 逐 patch 残差 r_p 过 σ((r_p − θ)/s) 软混入干净估计；softlog = 不混、只把 r_p 存到 <out>.ovres.npy 供标定")
    a("--any_far", default="", help="(hz2) 任意时距远槽 WM（train --any_far）推理时查询的远槽时距，逗号分隔，如 16,24,32 或 16,24,32,48,64（须在训练范围内）；默认空 = ckpt 的 --tgt_offs 里的远槽")
    a("--sh_sync", action="store_true", help="(mika MSFH-sync) train --self_hist 的 WM 不加 --roll 时走同步路径（--sampler 从纯噪声采全部槽，历史 token 经 fm.rollout demo= 传入）；默认关 = 原断言（只接 --roll）")
    a("--det_cfg", action="store_true", help="(ctrl) train --mode det 的 WM：每次决策一次前向出条件均值，并按 --cfg 做 x̂_∅ + cfg·(x̂_c − x̂_∅)；只走同步路径（不加 --roll）；默认关")
    a("--near_only", action="store_true", help="(faraux) 带远槽的 MS-far WM 推理时只用近槽（--tgt_offs 里时距 < 9 的前缀槽，n_fut 随之改），= MSN 的推理路径；"
                                               "配 train --far_drop 训的 WM；默认关")
    a("--wm_act", default="", choices=["", "clean", "pre"], help="(uwm) train --uwm 的 WM 用自己的动作 token 出动作（probwm/uwm.py）：clean = 画面槽给本次滚动去噪后的干净估计（τ = 1），"
                                                                "pre = 画面槽给滚动去噪前的窗口（联合去噪，画面这一步与动作第 1 步共用一次前向）；只接 --roll 的 MS-far msroll 路径；默认关")
    a("--wm_act_steps", type=int, default=4, help="(uwm) --wm_act：动作从纯噪声出发的 Euler 步数（第 1 步之后只算动作组）")
    a("--wm_act_mix", type=float, default=1.0, help="(uwm) --wm_act：WM 动作块的权重；1 = 只用 WM（不跑 IDM），< 1 = 与 IDM 的块（标准化空间）按权重平均")
    a = p.parse_args()
    if a.plan > 1 and a.exec == 2:
        a.exec = 2 * a.plan
    if a.merge:
        rs = [json.loads(Path(f).read_text()) for f in a.merge]
        return summarize([e for r in rs for e in r["episodes"]], {e["task"]: e["name"] for r in rs for e in r["episodes"]})
    torch.manual_seed(a.seed)
    goals = ()
    if a.para:
        from libero.libero import benchmark, get_libero_path
        P, bm = json.loads(Path(a.para).read_text()), benchmark.get_benchmark_dict()[a.suite]()
        a.text = a.text or P["emb"]
        bid = {bm.get_task(i).name: i for i in range(bm.n_tasks)}
        jobs = [(bid[nm], e, r, g) for nm, e, r, g in P["jobs"]][a.shard::a.nshard]
        goals = sorted(os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
                       for t in map(bm.get_task, range(bm.n_tasks)))  # 按名字排序 = tasks.json 顺序 = LIBERO-Para 的 eval_id
    elif a.plus:
        assert a.out and not a.perturb, "--plus 须给 --out；不与 --perturb 同用（--obs_sync 会重渲染、绕过 LIBERO-Plus 的传感器噪声，只用于无噪声的任务，如 LIBERO-PRO）"
        P = json.loads(Path(a.plus).read_text())
        a.text = a.text or P["emb"]
        jobs = [tuple(j) for j in P["jobs"]][a.shard::a.nshard]
        goals = tuple(P.get("goals", ()))  # job 清单带 goals（bddl 路径，如 probwm/goal_pro.py）时，同 --para 记录各目标首次成立的步数
    elif getattr(a, "pro2", ""):  # (pro) Pro2.0：job 只是占位（领哪个 episode 由服务端调度器定），总数 = run 的 episode 数
        jobs = [("pro2", i) for i in range(a.pro2_n)]
        assert a.wait == 0, "--pro2：服务端 reset 已空跑稳定化步，--wait 须为 0"
        a.max_steps = max(a.max_steps, 10 ** 6)  # 截断由服务端按 suite 上限判（truncated），主进程不另截
    else:
        jobs = [(t, e) for t in parse_ids(a.tasks) for e in range(a.ep0, a.ep0 + a.n_ep)][a.shard::a.nshard]
    tids = sorted({j[0] for j in jobs})
    if getattr(a, "pro2", ""):
        inits, names = {}, {"pro2": "pro2"}
    elif a.plus:  # 初始状态已由 libero_plus.py inits 安全加载进 init.npz；任务名即 key
        Z = np.load(P["init"])
        zc = {k: Z[k] for k in {P["init_key"][t] for t in tids}}
        inits, names = {t: zc[P["init_key"][t]] for t in tids}, {t: t for t in tids}
    else:
        inits, names = {t: init_states(a.suite, t) for t in tids}, {t: bench_task(a.suite, t).name for t in tids}
    n, ctx = min(a.n_env, len(jobs)), mp.get_context("spawn")
    pipes, pck = [], {"bc": a.bc, "wm+idm": a.idm}.get(a.policy)
    pargs = torch.load(pck, map_location="cpu")["args"] if pck else {}
    wrist = pargs.get("wrist", 0) > 0  # 策略要腕部相机时 env 才开这个相机
    if a.policy == "wm+idm" and int(torch.load(a.wm, map_location="cpu", mmap=True)["args"].get("tgt_tok", 0) or 0) > 0:  # (pro) 多视角 WM 的腕部上下文也要腕部相机
        assert not getattr(a, "cam_oracle", ""), "WM 上下文带腕部时不接 --cam_oracle"  # IDM 也可另读腕部（如 IW8，Policy.encode 另返回 IDM 的腕部流）
        wrist = True
    prop = tuple(pargs["proprio"].split(",")) if pargs.get("proprio") else ()  # 策略要本体感觉时 worker 才回传
    wobj = torch.load(a.wm, map_location="cpu")["args"].get("obj", "") if a.policy == "wm+idm" else ""
    wwhere = torch.load(a.wm, map_location="cpu")["args"].get("where", False) if a.policy == "wm+idm" else False
    seg = (bool(wobj) and a.obj_src == "oracle") or (bool(a.sguide) and a.sguide_src == "oracle") or (wwhere and a.where_src == "oracle")  # oracle 掩码：worker 另回传分割覆盖率
    for i_ in range(n):  # 先起子进程（spawn，不继承 CUDA 上下文），模型在主进程里同时加载
        pa, ch = ctx.Pipe()
        if getattr(a, "pro2", ""):  # (pro) Pro2.0：worker 是官方调度 run 的一个 runner（每个 worker 一个 runner_id）
            from .pro2_env import worker as pro2_worker
            ctx.Process(target=pro2_worker, args=(ch, prop, wrist, a.pro2_url, os.environ[a.pro2_key_env], a.pro2, f"{a.pro2_runner}-{i_}",
                                                  tuple(x for x in a.pro2_paths.split(",") if x)), daemon=True).start()
            pipes.append(pa)
            continue
        ctx.Process(target=worker, args=(ch, a.img, a.wait, a.perturb, goals, wrist, prop, a.obs_sync, bool(a.plus), seg,
                                         getattr(a, "soft_reset", False), getattr(a, "plus_reseed", False), bool(getattr(a, "trace", "")),
                                         bool(getattr(a, "cam_oracle", ""))),
                    daemon=True).start()
        pipes.append(pa)
    dev = torch.device("cuda")
    pol = Policy(a, dev)
    tt_ = int(getattr(pol.wa, "tgt_tok", 0) or 0) if a.policy == "wm+idm" else 0
    if tt_ > 0 and pol.dims[0] > tt_:  # (pro) train --tgt_tok 的多视角 WM：worker 另渲腕部（见上），Policy.encode 把 k×k 池化后的腕部 token 拼在 agentview 后
        pol.xw, pol.xwk = True, int(round((tt_ / (pol.dims[0] - tt_)) ** 0.5))
        assert pol.dims[0] - tt_ == tt_ // pol.xwk ** 2, ("--tgt_tok 的额外视角 token 数不是 agentview 的池化", pol.dims, tt_)
    pol.orc, pol.orc_d, pol.orc_e = {}, {}, {}  # --oracle：各 env 当前 episode 的专家特征 / 决策序号 / 专家末端轨迹
    pol.orc_ff = {}  # (pro) --oracle_far：各 env 当前 episode 只给远槽缓存用的专家特征
    tix = {} if a.plus or getattr(a, "pro2", "") else {t: pol.names.index(names[t]) for t in tids}
    prow, pmeta = [0] * n, [None] * n  # (pro) --pro2：各 env 当前指令在文本表里的行号、当前 episode 的 spec
    if getattr(a, "pro2", ""):
        from .pro2_env import TextCache
        tcache = TextCache(pol)
    if pol.nobj:  # 各任务的槽位类别：cat = 0..6；ground = 该任务 bddl 目标的 (操作物体, 目标)（LIBERO-PRO 的新目标也按它自己的 bddl）
        from libero.libero import get_libero_path
        bd = {t: os.path.join(get_libero_path("bddl_files"), a.suite, t + ".bddl") if a.plus else
              os.path.join(get_libero_path("bddl_files"), bench_task(a.suite, t).problem_folder, bench_task(a.suite, t).bddl_file) for t in tids}
        jsl = {t: list(range(pol.nobj)) if pol.wa.obj_mode == "cat" else list(pol.objtok.goal_cats(Path(bd[t]).read_text())) for t in tids}
        print("物体 token 槽位类别：", {str(t)[:50]: v for t, v in jsl.items()}, flush=True)
    if a.sguide and a.sguide_src == "oracle":  # oracle 引导目标：该任务 bddl 目标的 (操作物体, 放置目标) 类别
        from libero.libero import get_libero_path
        from .objtok import goal_cats
        jsg = {t: list(goal_cats(Path(os.path.join(get_libero_path("bddl_files"), a.suite, t + ".bddl") if a.plus else
                                      os.path.join(get_libero_path("bddl_files"), bench_task(a.suite, t).problem_folder, bench_task(a.suite, t).bddl_file)).read_text())) for t in tids}
    T, D = pol.dims
    hist = torch.zeros(n, (pol.K - 1) * pol.dl + 1, T, D, dtype=torch.float16, device=dev)  # 10 Hz 缓存，WM / IDM 看 hist[:, ::dl]
    whist = torch.zeros(n, pol.nw, getattr(pol, "wT", 0) or T, D, dtype=torch.float16, device=dev) if pol.nw else None  # 腕部相机最近 nw 帧（(pro) wT：IDM 腕部流的 token 数）
    rhist, chist = (torch.zeros_like(hist), torch.zeros_like(hist)) if pol.rawh else (None, None)  # 原始历史 / IDM 的规范化历史
    ohist = torch.zeros_like(hist) if getattr(a, "cam_oracle", "") else None  # (pro) --cam_oracle：IDM 的历史
    crW = getattr(a, "cam_route", 0)  # (cam) --cam_route：各 env 本集的路（0 未定 = 先走 canon，1 canon，2 原始）、窗口缓存、记录
    if crW:
        assert a.canon and a.policy == "wm+idm" and pol.npf and pol.dl == 1 and a.exec == 2 and not pol.rawh and not pol.nw and not getattr(a, "cam_oracle", "") \
            and not a.poe and not a.purify and not getattr(a, "ov", "") and crW >= pol.na.H + 2, "--cam_route 只接单帧 canon + 带本体的 IDM、--exec 2，不与 --wm_raw / 腕部 / --cam_oracle / --poe / --ov 同用"
        crt, crb, crl = [0] * n, [None] * n, [None] * n  # crb[w] = (原始帧列表, 规范化帧列表, 本体列表, 已执行动作列表)
    phist = torch.zeros(n, (pol.npf - 1) * pol.dl + 1, pol.na.pdim, device=dev) if pol.npf else None  # 本体感觉最近 npf 帧（原始单位）
    ens = [[] for _ in range(n)]  # --ens：各 env 之前决策的 (起始步, 动作块)
    pending, job, cur, t, t0 = list(jobs), [None] * n, [None] * n, [0] * n, [0.0] * n
    fresh, imgs, eps, tg = [True] * n, [None] * n, [], 0.0
    mag, flips, last_g = np.zeros(n), np.zeros(n), np.zeros(n)  # 诊断：平均平移动作幅度、夹爪切换次数
    hold = np.zeros(n)  # 夹爪滞回：新指令连续出现 --grip_hold 次才切换
    trj, gfirst, gend = [[] for _ in range(n)], [None] * n, [0] * n  # --para：末端轨迹、各任务目标首次成立的步数（-1 = 从未）
    rsv, rtg = [[] for _ in range(n)], [[] for _ in range(n)]  # (pro) --replan：各 env 本集逐决策的惊讶统计量（没有上次估计时为 None）与是否触发
    shc = None  # (ctrl) train --self_hist M,g 的 WM：每个 env 维护本 episode 已进 WM 历史的帧的池化缓存（同 mikasa_eval / robotwin_eval 的 HistCache），每次决策经 pol.sh_demo 进 WM；其余 WM 不变
    if getattr(getattr(pol, "wa", None), "self_hist", ""):
        assert a.exec == 2 and a.plan == 1 and pol.dl == 1, "self_hist 的 WM：每次决策须恰为一个 10 Hz 数据帧（--exec 2、--plan 1、train --delta 1），缓存帧号才与训练的选帧同口径"
        from .mikasa_eval import HistCache
        shc = HistCache(n, *(int(x) for x in pol.wa.self_hist.split(",")), T, D, pol.K, dev, wa=pol.wa)
        print(f"self_hist WM：历史缓存 M,g = {pol.wa.self_hist}（{n} 个 env）", flush=True)
    ovq, ove = [[] for _ in range(n)], [None] * n  # (pro) --ov_log：各 env 最近 12 个决策帧的图像 md5、当前这段图像不变开始时的末端位置
    or4 = [[] for _ in range(n)]  # (pro PC2) --ov 规则 4：本集逐决策 R4 是否触发
    ovf, odp, obk, ous, osv, osb = [[False, False]] * n, [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]  # (pro) --pro2：worker 的观测失效旗标、本集逐决策的 [陈旧]、[黑屏]；--ov：不可信、统计量、是否替换
    sv_on, rv = a.surp or bool(a.recover), pol.rv  # --surp / --recover：上一次计划对下一帧的预测、连续计数、恢复剩余决策数、日志
    pf, pok = (torch.zeros(n, T, D, dtype=torch.float16, device=dev) if sv_on else None), np.zeros(n, bool)
    cs, cm, rleft, bq, ahist = np.zeros(n), np.zeros(n), np.zeros(n, int), [[] for _ in range(n)], [[] for _ in range(n)]
    slog, mlog, tlog = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    pd = max(a.exec // 2, 1) - 1  # 执行 exec 个动作后的新观测对应计划里的第几帧

    def recv(w):
        r = pipes[w].recv()
        if isinstance(r, Exception):
            raise r
        return r

    h0 = [None] * n  # --hash0：各 env 本 episode 初始观测的 hash
    trc, TR = [None] * n, {}  # --trace：各 env 当前 episode 的轨迹 / 全部 episode 的轨迹（键 = (tag 或任务, 初态)）
    hsh = (lambda im: __import__("hashlib").sha1(np.ascontiguousarray(im[0] if isinstance(im, tuple) else im).tobytes()).hexdigest()[:16]) if getattr(a, "hash0", False) else (lambda im: None)
    waitr = set()  # --async_reset：在 reset（排队等建 env 的槽）还没回话的 worker；其余 env 照常推进，不整批停下来等

    def assign(ws):  # 给空闲 worker 分下一个 episode：优先同任务，免得重建 env
        for w in ws:
            job[w] = None
            if pending:
                j = next((j for j in pending if j[0] == cur[w]), pending[0])
                pending.remove(j)
                job[w], cur[w], t[w], t0[w], fresh[w], mag[w], flips[w], hold[w] = j, j[0], 0, time.time(), True, 0, 0, 0
                trj[w], gfirst[w], gend[w], ens[w] = [], [-1] * len(goals), 0, []
                rsv[w], rtg[w] = [], []
                ovf[w], odp[w], obk[w], ous[w], osv[w], osb[w], or4[w] = [False, False], [], [], [], [], [], []
                ovq[w], ove[w] = [], None
                pok[w], cs[w], cm[w], rleft[w], bq[w], ahist[w], slog[w], mlog[w], tlog[w] = False, 0, 0, 0, [], [], [], [], []
                ox = None
                if a.oracle:  # --oracle：plandiag --save_expert 存的专家（<任务名>__<初态>.npz）；没有专家的 episode 用 WM 的计划
                    fo_ = Path(a.oracle) / f"{j[0]}__{j[1]}.npz"
                    pol.orc[w] = pol.orc_e[w] = pol.orc_ff[w] = None
                    if fo_.exists():
                        X_ = np.load(fo_)
                        ox = (X_["body_pos"], X_["body_quat"], X_["S0"])
                        if not a.oracle_wm:
                            fe_ = torch.cat([pol.encode(list(X_["rgb"][i:i + 64])) for i in range(0, len(X_["rgb"]), 64)])
                            if getattr(a, "oracle_far", False):  # (pro) 部分 oracle：专家特征只给远槽缓存（见主循环），IDM 的计划不整段替换
                                pol.orc_ff[w] = fe_
                            else:
                                pol.orc[w] = fe_
                        pol.orc_e[w] = (X_["prop"][:, :3].astype(np.float64), X_["grs"])  # --oracle_exec：专家每帧的末端位置与夹爪指令
                pipes[w].send(("reset", None) if getattr(a, "pro2", "") else ("reset", ((a.suite, j[0]), inits[j[0]][j[1]], ox)))
        for w in ws:
            if job[w] is not None:
                if getattr(a, "async_reset", False):
                    waitr.add(w)
                else:
                    r_ = recv(w)
                    if not pro2_first(w, r_):
                        continue
                    imgs[w] = r_[0]
                    h0[w] = hsh(imgs[w])
                    if a.trace:
                        trc[w] = dict(names=r_[-1][3], eef=[r_[-1][0]], grip=[r_[-1][1]], obj=[r_[-1][2]], plan=[], g=[])

    def pro2_first(w, r_):  # (pro) --pro2 的 reset 回话：None = 调度器已没有可领的 episode（这个 env 停下）；否则记下指令行号与 spec
        if not getattr(a, "pro2", ""):
            return True
        if r_ is None:
            job[w] = None
            return False
        pmeta[w], prow[w] = r_[3], tcache(r_[3]["instr"])
        return True

    def collect(block):  # --async_reset：收已经 reset 完的 worker 的首帧；block 时至少等到一个
        from multiprocessing.connection import wait
        if block and waitr:
            wait([pipes[w] for w in waitr])
        for w in list(waitr):
            if pipes[w].poll():
                r_ = recv(w)
                if not pro2_first(w, r_):
                    waitr.discard(w)
                    continue
                imgs[w] = r_[0]
                h0[w] = hsh(imgs[w])
                if a.trace:
                    trc[w] = dict(names=r_[-1][3], eef=[r_[-1][0]], grip=[r_[-1][1]], obj=[r_[-1][2]], plan=[], g=[])
                waitr.discard(w)

    T0 = time.time()
    assign(range(n))
    while any(j is not None for j in job):
        collect(False)
        act_ = [w for w in range(n) if job[w] is not None and w not in waitr]
        if not act_:
            collect(True)
            continue
        g0 = time.time()
        ob = [imgs[w] for w in act_]
        if getattr(a, "ov_log", False) and not getattr(a, "pro2", ""):  # (pro) 决策帧上的陈旧 / 黑屏旗标（只记）
            import hashlib
            assert pol.npf, "--ov_log 要策略带本体感觉（末端位置）"
            for w in act_:
                im_, pv_ = imgs[w][0], np.asarray(imgs[w][1], np.float64)
                im_ = im_[0] if isinstance(im_, tuple) else im_
                h_, e_ = hashlib.md5(np.ascontiguousarray(im_).tobytes()).digest(), pv_[:3]
                if fresh[w]:
                    ovq[w], ove[w] = [], e_
                if ovq[w] and h_ == ovq[w][-1]:
                    dup_ = float(np.abs(e_ - ove[w]).max()) > 1e-3
                else:
                    dup_, ove[w] = h_ in ovq[w], e_
                ovq[w] = (ovq[w] + [h_])[-12:]
                ovf[w] = [bool(dup_) and not fresh[w], not np.asarray(im_).any()]
        if seg:
            ob, cv = zip(*ob)
            cv = torch.from_numpy(np.stack(cv)).to(dev)
        if pol.npf:
            ob, pv = zip(*ob)
            pv = torch.from_numpy(np.stack(pv)).to(dev)
        if a.cam_oracle:  # (pro) worker 回传的是 [扰动相机画面, 原始相机画面]
            ob, obc = [x[0] for x in ob], [x[1] for x in ob]
        idx = torch.tensor(act_, device=dev)
        fr = torch.tensor([fresh[w] for w in act_], device=dev)[:, None, None, None]
        if crW:  # (cam) --cam_route：先取原始特征，再照常规范化（与默认路径同一个 canonize，调用次序不变）
            cs_, pol.a.canon = pol.a.canon, ""
            zr_ = pol.encode(list(ob), idx, fr.flatten())
            pol.a.canon = cs_
            z = pol.canonize(zr_)
        else:
            z = pol.encode(list(ob), idx, fr.flatten())
        z, zw = z if pol.nw else (z, None)
        ch = None
        ti = torch.tensor([prow[w] if getattr(a, "pro2", "") else job[w][2] if a.para or a.plus else tix[job[w][0]] for w in act_], device=dev)
        if pol.rawh:  # 原始历史 → 规范化（时序规范化器用最近几帧）；--wm_raw 时 WM 仍读原始帧，规范化结果只进 IDM 的历史
            rh = torch.cat([rhist[idx, 1:], z[:, None]], 1)
            rhist[idx] = rh = torch.where(fr, z[:, None].expand_as(rh), rh)
            zc = pol.poe(rh[:, -1], hist[idx], ti, fr.flatten(), idx) if a.poe else pol.canonize(rh)
            if a.wm_raw:
                ch = torch.cat([chist[idx, 1:], zc[:, None]], 1)
                chist[idx] = ch = torch.where(fr, zc[:, None].expand_as(ch), ch)
            else:
                z = zc
        if a.cam_oracle:  # (pro) 原始相机画面不过规范化；wm / both：WM 的历史换成它；idm / both：IDM 的历史（hc）换成它
            assert not pol.rawh and not pol.nw, "--cam_oracle 只接单帧规范化（canon K=1）、不带腕部"
            cs_, pol.a.canon = pol.a.canon, ""
            zcl = pol.encode(list(obc), idx, fr.flatten())
            pol.a.canon = cs_
            zi_ = zcl if a.cam_oracle in ("idm", "both") else z
            ch = torch.cat([ohist[idx, 1:], zi_[:, None]], 1)
            ohist[idx] = ch = torch.where(fr, zi_[:, None].expand_as(ch), ch)
            if a.cam_oracle in ("wm", "both"):
                z = zcl
        if getattr(a, "pro2", "") or getattr(a, "ov_log", False):  # (pro) 观测失效旗标逐决策记下；--ov 时据此（及 R3）把不可信的观测换成伪观测，再进历史
            fl_ = [ovf[w] for w in act_]
            for i, w in enumerate(act_):
                odp[w].append(bool(fl_[i][0]))
                obk[w].append(bool(fl_[i][1]))
            if pol.ov is not None:
                z = pol.ov_sub(z, idx, fr.flatten(), torch.tensor(fl_, dtype=torch.bool, device=dev), pe=pv[:, :3] if pol.npf else None)
                sv_, va_, u_, sb_ = (x.cpu().tolist() for x in pol.ov_last)
                r4_ = pol.ov_r4last.cpu().tolist() if getattr(pol, "ov_r4", None) is not None else None
                if r4_ is not None:
                    for i, w in enumerate(act_):
                        or4[w].append(bool(r4_[i]))
                for i, w in enumerate(act_):
                    ous[w].append(bool(u_[i]) if pol.ov[0] in ("wm", "hold") else round(float(u_[i]), 4))  # soft：本次决策逐 patch 平均 w
                    osv[w].append(round(sv_[i], 4) if va_[i] else None)
                    osb[w].append(bool(sb_[i]) if pol.ov[0] in ("wm", "hold") else round(float(sb_[i]), 4))  # soft：w > 0.5 的 patch 占比
        if crW:  # (cam) --cam_route：窗口内两路都缓存；满 N 帧时按自监督逆动力学选路；选原始的 env 之后读原始特征
            for i, w in enumerate(act_):
                if fresh[w]:
                    crt[w], crb[w], crl[w] = 0, ([], [], [], []), None
                if crt[w] == 0:
                    crb[w][0].append(zr_[i].clone()), crb[w][1].append(z[i].clone()), crb[w][2].append(pv[i].clone())
                    if len(crb[w][0]) >= crW and len(crb[w][3]) < 2 * (crW - 1):  # 窗口内执行的动作不够（被截断）：不选，留在 canon
                        crt[w], crl[w], crb[w] = 1, {"pick": "canon", "s3": None, "at": len(crb[w][0])}, None
                    elif len(crb[w][0]) >= crW:
                        R_, C_, P_ = (torch.stack(x) for x in crb[w][:3])
                        A_ = torch.tensor(crb[w][3], dtype=torch.float32, device=dev)
                        an_ = torch.arange(0, crW - pol.na.H, device=dev)
                        fi_ = torch.clamp(an_[:, None] + torch.arange(-pol.na.ih, pol.na.H + 1, device=dev), min=0)
                        pi_ = torch.clamp(an_[:, None] + torch.arange(-pol.na.ih, 1, device=dev), min=0)
                        ai_ = torch.clamp(2 * an_[:, None] + torch.arange(2 * pol.na.H, device=dev), max=len(A_) - 1)
                        sc_ = []
                        for Z_ in (C_, R_):
                            with torch.random.fork_rng(devices=[dev]), torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                                torch.manual_seed(12345)
                                o_ = pol.net(Z_[fi_], zp=P_[pi_]).float()
                            sc_.append(pol.idm.metrics(o_, A_[ai_], pol.st)["nmse"])
                        crt[w] = 2 if sc_[1] < sc_[0] else 1
                        crl[w] = {"pick": "raw" if crt[w] == 2 else "canon", "s3": [round(sc_[0], 4), round(sc_[1], 4)], "at": len(crb[w][0])}
                        if crt[w] == 2:  # 历史换成原始特征（本帧之前的那几帧；本帧下面照常拼上），滚动缓存作废 → 下次 rolling 同新 episode 重采
                            L_ = hist.shape[1]
                            hist[w] = torch.stack([crb[w][0][max(len(crb[w][0]) - 1 - L_ + j, 0)] for j in range(L_)]).to(hist.dtype)
                            if isinstance(getattr(pol, "rv", None), torch.Tensor) and pol.rv.dim() == 2:
                                pol.rv[w] = False
                        crb[w] = None
            z = torch.where(torch.tensor([crt[w] == 2 for w in act_], device=dev)[:, None, None], zr_.to(z.dtype), z)
        hp, h = hist[idx], torch.cat([hist[idx, 1:], z[:, None]], 1)  # hp：上一时刻的历史（--wmpur 用）
        hist[idx] = h = torch.where(fr, z[:, None].expand_as(h), h)  # 新 episode：历史全用第一帧
        if a.wmpur:  # WM 条件净化，写回历史（之后的净化以已净化的历史为条件）；tb = 基础任务在 tasks.json 的下标（场景锚用）
            tb = torch.tensor([max((i for i, nm in enumerate(pol.names) if job[w][0].startswith(nm)), key=lambda i: len(pol.names[i]))
                               if a.plus else tix[job[w][0]] for w in act_], device=dev)
            hist[idx] = h = pol.wmpur(hp, h, ti, fr.flatten(), idx, tb)
        if a.pfilt:  # WM 条件后验净化（递归滤波）
            hist[idx] = h = pol.pfilt(hp, h, ti, fr.flatten())
        if shc is not None:  # (ctrl) 刚进 WM 历史的这一帧先进缓存，再按训练的 sh_pick 在 [开头, t − K] 里取本 episode 的历史帧
            pol.sh_demo = shc.step(idx, fr.flatten(), h[:, -1])
        wh = ph = None
        if pol.nw:  # 腕部历史同理
            wh = torch.cat([whist[idx, 1:], zw[:, None]], 1)
            whist[idx] = wh = torch.where(fr, zw[:, None].expand_as(wh), wh)
        if pol.prim is not None:  # 当前原语（上一次执行的夹爪指令、末端高度、抽屉检测器）
            ti = pol.prim.step(act_, fr.flatten(), ti, z.float() * pol.sd + pol.mu, last_g[act_] > 0, pv[:, 2].tolist())
        if pol.npf:  # 本体感觉历史同理
            ph = torch.cat([phist[idx, 1:], pv[:, None]], 1)
            phist[idx] = ph = torch.where(fr[..., 0], pv[:, None].expand_as(ph), ph)
        rec = None
        if sv_on:  # 惊讶 / 运动（标准化空间，按 WM 的逐通道帧差尺度 rs 归一，patch 和通道上平均）
            zf = z.float()
            sv = ((zf - pf[idx].float()) / pol.rs).pow(2).mean((1, 2)).tolist()
            mv = ((zf - h[:, -2].float()) / pol.rs).pow(2).mean((1, 2)).tolist()
            for i, w in enumerate(act_):
                okm = not fresh[w]
                slog[w].append(round(sv[i], 3) if pok[w] else None)
                mlog[w].append(round(mv[i], 3) if okm else None)
                if rv and rleft[w] == 0:
                    cs[w] = cs[w] + 1 if pok[w] and sv[i] > rv[2] else 0
                    cm[w] = cm[w] + 1 if okm and mv[i] < rv[3] else 0
                    if max(cs[w], cm[w]) >= rv[1]:
                        rleft[w], cs[w], cm[w] = rv[5], 0, 0
                        tlog[w].append(len(slog[w]) - 1)
                        g = last_g[w]
                        if rv[0] == "lift":
                            bq[w] = [[0, 0, rv[4], 0, 0, 0, g]] * (2 * rv[5])
                        elif rv[0] == "reverse":
                            bq[w] = [list(-np.asarray(u[:6])) + [g] for u in ahist[w][::-1][:2 * rv[5]]]
            if rv and rv[0] in ("cfg", "second"):
                rec = torch.tensor([rleft[w] > 0 for w in act_], device=dev)
        obj = pol.objin(cv if seg else None, h[:, -1], torch.tensor([jsl[job[w][0]] for w in act_], device=dev), ti) if pol.nobj else None
        gdf = pol.sguide(cv if seg else None, h[:, -1], torch.tensor([jsg[job[w][0]] for w in act_], device=dev) if seg and a.sguide_src == "oracle" else None,
                         ti, torch.tensor([bool(last_g[w] > 0 and not fresh[w]) for w in act_], device=dev)) if pol.sg else None
        if a.oracle:
            pol.orc_d = {w: t[w] // a.exec for w in act_}  # 专家每 2 个环境步一帧（= 一次决策）
            if getattr(a, "oracle_far", False):  # (pro) 部分 oracle：Policy.rolling 把远槽缓存换成专家第 d + 时距 帧（新 episode 的第一次决策不换）
                pol.orc_far = {w: (pol.orc_ff[w], pol.orc_d[w]) for w in act_ if pol.orc_ff.get(w) is not None}
        acts = pol.act(h[:, ::pol.dl], ti, idx, fr.flatten(), wh, ph if ph is None else ph[:, ::pol.dl], ch, rec, obj, gdf)
        if getattr(a, "replan", "") and getattr(pol, "rp_last", None) is not None:  # (pro) 记下这次决策的惊讶统计量与触发（只读，不改推理）
            sv_, va_, tr_ = (x.cpu().tolist() for x in pol.rp_last)
            for i, w in enumerate(act_):
                rsv[w].append(round(sv_[i][pol.rp[3]], 4) if va_[i] else None)
                rtg[w].append(bool(tr_[i]))
        if sv_on:
            pf[idx] = pol.plan[:, pd].to(pf.dtype)
        g1 = time.time()
        for i, w in enumerate(act_):
            x = acts[i, :min(a.exec, a.max_steps - t[w])]
            if a.wp_exec:  # 可部署版：同 --oracle_exec，但航点 = 探针从 WM 计划解出的 t+1 末端；夹爪始终用 IDM
                geo = np.clip((pol.wp[i] - pv[i, :3].double().cpu().numpy()) / np.array([0.0193, 0.0150, 0.0151]), -1, 1)
                x = x.copy()
                if a.wp_exec == "mix":
                    x[:, :3] = 0.5 * x[:, :3] + 0.5 * geo
                else:
                    x[:, :3] = geo
                    if a.wp_exec == "geo":
                        x[:, 3:6] = 0
            if a.oracle_exec and pol.orc_e.get(w) is not None:  # 执行侧上界：以专家下一帧的末端位置为航点（LIBERO 动作 = 末端增量；c = demo 上拟合的每帧位移 / 动作）
                pe_, gr_ = pol.orc_e[w]
                d_ = t[w] // a.exec
                geo = np.clip((pe_[min(d_ + 1, len(pe_) - 1)] - pv[i, :3].double().cpu().numpy()) / np.array([0.0193, 0.0150, 0.0151]), -1, 1)
                x = x.copy()
                if a.oracle_exec == "mix":  # IDM 动作 + 指向航点的残差（各一半）；转动、夹爪仍用 IDM
                    x[:, :3] = 0.5 * x[:, :3] + 0.5 * geo
                else:  # geo：纯几何（不转腕）/ geo_rot：几何平移 + IDM 的转动；夹爪都照专家
                    x[:, :3] = geo
                    if a.oracle_exec == "geo":
                        x[:, 3:6] = 0
                    x[:, 6] = 1.0 if gr_[min(d_, len(gr_) - 1)] > 0 else -1.0
            if sv_on:
                pok[w] = True
                if rleft[w] > 0:
                    rleft[w] -= 1
                    if rv[0] in ("lift", "reverse"):  # 退让：这次不执行计划（下一次决策不算惊讶）；倒放动作不够时停在原地
                        q = bq[w][:len(x)] + [[0, 0, 0, 0, 0, 0, last_g[w]]] * max(0, len(x) - len(bq[w]))
                        x, bq[w], pok[w] = np.asarray(q, np.float32), bq[w][len(x):], False
            if a.ens is not None:  # 时间集成：覆盖第 t+j 步的各个块按 exp(−m·i) 加权（i=0 最早）
                ens[w] = [(s0, c) for s0, c in ens[w] if s0 + len(c) > t[w]] + [(t[w], acts[i].copy())]
                for j in range(len(x)):
                    ck = np.stack([c[t[w] + j - s0] for s0, c in ens[w] if t[w] + j - s0 < len(c)])  # 别叫 cs：会覆盖 --surp 的逐 env 计数数组
                    wt = np.exp(-a.ens * np.arange(len(ck)))[:, None]
                    x[j] = (wt * ck).sum(0) / wt.sum()
                x[:, 6] = np.where(x[:, 6] > 0, 1.0, -1.0)
            if a.grip_hold > 1 and not fresh[w]:
                hold[w] = hold[w] + 1 if x[0, 6] != last_g[w] else 0
                x[:, 6] = x[0, 6] if hold[w] >= a.grip_hold else last_g[w]
                hold[w] = 0 if hold[w] >= a.grip_hold else hold[w]
            g = np.concatenate([[x[0, 6]] if fresh[w] else [last_g[w]], x[:, 6]])
            mag[w], flips[w], last_g[w] = mag[w] + np.abs(x[:, :3]).sum(), flips[w] + (np.diff(g) != 0).sum(), x[-1, 6]
            pipes[w].send(("step", x.tolist()))
            if crW and crt[w] == 0 and crb[w] is not None:  # (cam) --cam_route：窗口内实际执行的动作
                crb[w][3].extend(np.asarray(x, np.float32).tolist())
            if a.trace:
                trc[w]["plan"].append(pol.trx[i].astype(np.float16))
                trc[w]["g"].append(float(x[-1, 6]))
            fresh[w] = False
            if sv_on:
                ahist[w] = (ahist[w] + x.tolist())[-16:]
        fin = []
        for w in act_:
            imgs[w], ok, k, *dg = recv(w)
            pdone = False
            if getattr(a, "pro2", ""):  # (pro) 回话的最后一项是 meta：当前指令（D21 会换）、本集是否结束（成功或服务端截断）
                m_ = dg.pop()
                prow[w], pdone = tcache(m_["instr"]), bool(m_["done"])
                ovf[w] = list(m_.get("ov", [False, False]))
            if a.trace:
                tr_ = dg.pop()
                trc[w]["eef"].append(tr_[0]), trc[w]["grip"].append(tr_[1]), trc[w]["obj"].append(tr_[2])
            t[w] += k
            if dg:
                trj[w].append(dg[0])
                gend[w] = dg[1]
                gfirst[w] = [s if s >= 0 or not dg[1] >> i & 1 else t[w] for i, s in enumerate(gfirst[w])]
            if ok or t[w] >= a.max_steps or pdone:
                eps.append(dict(task=job[w][0], name=names[job[w][0]], ep=job[w][1], success=ok, steps=t[w], **({"img0": h0[w]} if h0[w] else {}),
                                **({"pro2": {k_: pmeta[w][k_] for k_ in ("spec", "job")}} if getattr(a, "pro2", "") else {}),
                                **({"rp_surp": rsv[w], "rp_trig": rtg[w]} if getattr(a, "replan", "") else {}),
                                **({"ov_dup": odp[w], "ov_blank": obk[w]} if getattr(a, "pro2", "") or getattr(a, "ov_log", False) else {}),
                                **({"ov_u": ous[w], "ov_s": osv[w], "ov_sub": osb[w]} if getattr(a, "ov", "") else {}),
                                **({"ov_r4": or4[w]} if getattr(pol, "ov_r4", None) is not None else {}),
                                **({"cam_route": crl[w]} if crW else {}),
                                time=time.time() - t0[w], act_xyz=mag[w] / (3 * t[w]), grip_flips=int(flips[w])))
                if a.oracle:
                    eps[-1]["oracle"] = pol.orc.get(w) is not None or pol.orc_ff.get(w) is not None
                if hasattr(pol, "pd"):
                    eps[-1].update(poe_d=(pol.pd[w, 0] / pol.pd[w, 2]).item(), poe_on=(pol.pd[w, 1] / pol.pd[w, 2]).item(),
                                   poe_u=(pol.pd[w, 3] / pol.pd[w, 2]).item(), **({"poe_tr": pol.ptr[w]} if a.poe_trace else {}))
                if a.para:
                    eps[-1].update(row=job[w][2], tag=job[w][3], eef=trj[w], goal_first=gfirst[w], goal_end=gend[w])
                elif a.plus:
                    eps[-1].update(row=job[w][2], tag=job[w][3], **({"goal_first": gfirst[w], "goal_end": gend[w]} if goals else {}))
                if sv_on:
                    eps[-1].update(surp=slog[w], mot=mlog[w], trig=tlog[w])
                if a.pcons and pol.pp is not None:
                    c = pol.pcs[w].tolist()
                    eps[-1].update(pcons=c[0] / max(c[2], 1), pcons1=c[1] / max(c[2], 1))
                if pol.kfp is not None:  # 两级：沿用（细化）关键帧的决策上相邻关键帧的变化、沿用 / 从头采的决策数
                    c = pol.kcs[w].tolist()
                    eps[-1].update(kcons=c[0] / max(c[1], 1), kf_keep=int(c[1]), kf_redo=int(c[2]))
                if pol.prim is not None:
                    eps[-1].update(prim=pol.prim.log(w))
                if a.trace:  # 末端 / 夹爪 / 物体每次决策后一帧（第 0 帧 = reset 后），计划各槽探针 xyz（第 0 列 = 当前帧），执行的夹爪指令
                    TR[(job[w][3] if a.para or a.plus else job[w][0], job[w][1])] = dict(
                        {k: np.stack(v) if k in ("eef", "grip", "obj", "plan") and len(v) else v for k, v in trc[w].items()}, success=bool(ok), steps=t[w])
                print(f"[{len(eps)}/{len(jobs)}] task {job[w][0]} ep {job[w][1]:>2} {'成功' if ok else '失败'} "
                      f"{t[w]:>3} 步 {eps[-1]['time']:.0f}s  |a_xyz| {eps[-1]['act_xyz']:.3f} 夹爪切换 {int(flips[w])}",
                      flush=True)
                fin.append(w)
        tg += g1 - g0
        assign(fin)
    wall = time.time() - T0
    for pa in pipes:
        pa.send(("close", None))
    print(f"{a.policy} {a.sampler if a.policy != 'bc' else ''}: {len(eps)} ep 用时 {wall:.0f}s，"
          f"平均每 ep {np.mean([e['time'] for e in eps]):.1f}s（{n} 个 env 并行，吞吐 {wall / len(eps):.1f}s/ep）；"
          f"推理 {tg:.0f}s / 环境+复位 {wall - tg:.0f}s", flush=True)
    rate = summarize(eps, names)
    out = Path(a.out or f"runs/goal/closed_loop/{a.policy.replace('+', '_')}"
               f"{'_' + a.sampler if a.policy != 'bc' else ''}_s{a.shard}of{a.nshard}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(a), "episodes": eps, "rate": rate, "wall": wall}, indent=1))
    if getattr(pol, "ov", None) is not None and pol.ov[0] == "softlog" and pol.ov_res:  # (pro) OV-soft 标定：逐 patch 残差存到 <out 去掉 .json>.ovres.npy
        np.save(str(out)[:-5] + ".ovres.npy", torch.cat(pol.ov_res).numpy())
    if a.trace:
        torch.save(TR, a.trace)


if __name__ == "__main__":
    main()
