"""RoboTwin 2.0 闭环评测，逐项照官方 XPolicyLab eval_remote_policy（scripts/eval_policy_xpolicylab.py）：
Easy = --cfg demo_clean（seen 指令），Hard = --cfg demo_randomized（unseen 纹理 + 杂物 + 灯光 + 桌高，unseen 指令）。
- 每任务 100 个 episode；种子从 100000·(1+seed) 起逐个试：先在该种子上跑专家（setup_demo(is_test=True) → play_once()），
  不稳定 / 出错 / 专家规划或成功判定失败的种子跳过。有效与否与策略无关 → 多个 worker 按种子升序并行试，凑够前 100 个有效种子
  就停（同一任务在途的种子数 ≤ 还缺的个数，所以集合与官方顺序执行完全相同）。已知无效的种子记在 runs/rt/seeds/<cfg>/<task>.json，下次直接跳过。
- 有效种子上重新 setup_demo；指令 = generate_episode_descriptions(任务, [专家 play_once 的 info], 100, seed) 的 seen / unseen 里
  random.Random(f"{seed}:select").choice（同官方 build_instruction）；任务的成功判定依赖专家设下的属性（如 open_laptop 的 arm_tag），
  所以有效种子上的专家一定要跑（同官方）。
- 每个路点 take_action(qpos) 算一步，每步后判 eval_success，到 _eval_step_limit.yml 的上限为止；第一步之前环境就出错的不计（换下一个种子，同官方）。
策略每 --exec 个路点决策一次：head（+ 腕部）to256 → closed_loop.Policy（wm+idm / bc）→ 14 维 qpos 路点。
观测（--obs cams，默认）：只给策略要用的相机拍照（官方 get_obs 每次给 head / front / 双腕 4 台相机都做 32 spp 光追，B200 没有 RT 核，
  每台约 130 ms，占评测时间的大半）；拍照的函数、相机参数、渲染设置与 get_obs 完全相同（_update_render → take_picture →
  get_picture("Color") ×255 截断 uint8），本体 = get_obs 的 joint_action/vector。--obs full = 原样调 get_obs（核对用）。
  光追渲染是 GPU 瓶颈（一张卡 4–8 个 env 就打满），所以 --sim_gpus 给多张卡；50 任务全量按任务分片到多个进程 / 机器（scripts/robotwin_eval.sh）。
worker 每做完 --recycle 个 episode 重起一次（SAPIEN worker 显存随时间上涨）；同一 worker 优先接着做上一个任务（省每个任务第一次 setup 的加载）。
IDM 带腕部（--wrist）时腕部流 = 左 / 右腕各 2×2 池化成 8×8 后按 token 拼接（feats/<encoder>_wrist，render_robotwin 之后另建）。
要在隔离环境里跑（scripts/robotwin.sh）：
  python -m probwm.robotwin_eval --wm runs/rt/wm/ckpt.pt --idm runs/rt/idm/idmF8P.pt --cfg demo_clean --out runs/rt/cl/x_easy.json
  快速诊断：--n_ep 20；合并分片：python -m probwm.robotwin_eval --merge runs/rt/cl/x_*.json
"""
import os

for _k, _v in [("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1")]:
    os.environ.setdefault(_k, _v)

import argparse
import json
import multiprocessing as mp
from multiprocessing.connection import wait as mp_wait
import random
import time
from pathlib import Path

import numpy as np
import torch

from .render_robotwin import OBS, PILOT, _close, _setup, jpeg, rt_hold, rt_path, task_args, to256  # jpeg：--jpeg；_setup / _close：建 / 关场景排队


def end_state(env):  # (rteasy) --end_state：episode 结束时的末态（任务对象位姿 / 关节、夹爪开合、接触对），失败归类用；出错的项跳过
    out = {}
    for k_, f_ in (("lg_open", "is_left_gripper_open"), ("rg_open", "is_right_gripper_open")):
        try:
            out[k_] = bool(getattr(env.robot, f_)())
        except Exception:
            pass
    try:
        out["grip"] = [float(env.robot.get_left_gripper_val()), float(env.robot.get_right_gripper_val())]
    except Exception:
        pass
    for k_, f_ in (("ee_l", "get_left_ee_pose"), ("ee_r", "get_right_ee_pose"), ("tcp_l", "get_left_tcp_pose"), ("tcp_r", "get_right_tcp_pose")):
        try:  # 末端 / TCP 位姿（按键类任务算末端到按键的距离）
            out[k_] = np.asarray(getattr(env.robot, f_)(), float).round(4).tolist()
        except Exception:
            pass
    obj = {}
    for k_, v_ in list(vars(env).items()):
        try:
            if type(v_).__name__ in ("Actor", "ArticulationActor"):
                p_ = v_.get_pose()
                d_ = {"name": v_.get_name(), "p": np.asarray(p_.p, float).round(4).tolist(), "q": np.asarray(p_.q, float).round(4).tolist()}
                if hasattr(v_, "get_qpos"):
                    d_["qpos"], d_["qlim"] = np.asarray(v_.get_qpos(), float).round(4).tolist(), np.asarray(v_.get_qlimits(), float).round(4).tolist()
                for i_ in range(4):  # 功能点 / 接触点（check_success 多用它们，不是物体原点）
                    try:
                        d_.setdefault("fp", []).append(np.asarray(v_.get_functional_point(i_, "pose").p, float).round(4).tolist())
                    except Exception:
                        break
                try:
                    d_["cp0"] = np.asarray(v_.get_contact_point(0), float)[:3].round(4).tolist()
                except Exception:
                    pass
                obj[k_] = d_
            elif type(v_).__name__ == "Pose" and any(w_ in k_ for w_ in ("target", "goal", "pose", "des")):  # sapien.Pose 型的目标位姿
                obj[k_] = {"p": np.asarray(v_.p, float).round(4).tolist(), "q": np.asarray(v_.q, float).round(4).tolist()}
            elif isinstance(v_, (np.ndarray, list, tuple)) and any(w_ in k_ for w_ in ("target", "goal", "pose", "des")) \
                    and 0 < np.asarray(v_, dtype=object).size <= 64 and np.asarray(v_).dtype.kind in "fi":  # 数组型的目标位置
                obj[k_] = {"arr": np.asarray(v_, float).round(4).tolist()}
            elif isinstance(v_, (list, tuple)) and v_ and all(type(x_).__name__ in ("Actor", "ArticulationActor") for x_ in v_):
                obj[k_] = [{"name": x_.get_name(), "p": np.asarray(x_.get_pose().p, float).round(4).tolist(),
                            "q": np.asarray(x_.get_pose().q, float).round(4).tolist()} for x_ in v_]
        except Exception:
            pass
    out["obj"] = obj
    out["attr"] = {k_: v_ for k_, v_ in vars(env).items() if not k_.startswith("_") and isinstance(v_, (bool, int, float, str)) and len(str(v_)) < 80}
    pat = [(o_, n_) for o_ in (env, env.robot) for n_ in ("is_left_gripper_open", "is_right_gripper_open") if hasattr(o_, n_)]
    old = [getattr(o_, n_) for o_, n_ in pat]
    try:  # 位置类条件是否已满足：把"夹爪张开"条件当成已满足再判一次 check_success（区分"到位但没松手"）
        for o_, n_ in pat:
            setattr(o_, n_, lambda: True)
        out["succ_if_open"] = bool(env.check_success())
    except Exception:
        pass
    finally:
        for (o_, n_), f_ in zip(pat, old):
            setattr(o_, n_, f_)
    try:
        cs = sorted({tuple(sorted((c_.bodies[0].entity.name, c_.bodies[1].entity.name))) for c_ in env.scene.get_contacts()})
        out["contacts"] = [list(c_) for c_ in cs if not ({"table", "ground", "wall"} & set(c_))][:300]
    except Exception:
        pass
    return out


def worker(conn, cfg, img, keys, obs_mode="cams", jp="", every=0, ee=0, orc=False, es=False):  # es：(rteasy) --end_state  # orc：(rteasy) --oracle_plan，专家检查那一遍按采集频率截头部帧  # keys：要回传的相机（render_robotwin.OBS 的键，第一个是 head）；jp：--jpeg；every：--obs_every；ee：(prec) 末端裁剪边长
    try:  # 主进程退出（被杀）时 worker 跟着退出，不留孤儿进程占卡
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass
    rt_path()
    keep = rt_hold()  # noqa: F841（本机 RoboTwin 渲染进程总数上限，持有到进程结束）
    import sys
    sys.stdout = open(os.devnull, "w")  # RoboTwin 每个 take_action 都 print 一行 step（\r），不进日志
    from envs import UnStableError
    from generate_episode_instructions import generate_episode_descriptions
    from scripts.collect_data import class_decorator
    envs, n_done = {}, 0

    def close(env, cc=False):
        try:
            _close(env, clear_cache=cc)
        except Exception:
            pass

    def view(env):
        if obs_mode == "full":  # 官方 get_obs（所有相机）
            o = env.get_obs()
            x = [to256(jpeg(np.ascontiguousarray(o["observation"][OBS[k]]["rgb"]), jp) if jp else o["observation"][OBS[k]]["rgb"], img) for k in keys]
            q = o["joint_action"]["vector"]
        else:  # 同 get_obs 的渲染与取图，只拍要用的相机
            env._update_render()
            c = env.cameras
            cams = {"rgb": c.static_camera_list[c.static_camera_name.index("head_camera")], "wl": c.left_camera, "wr": c.right_camera}
            for k in keys:
                cams[k].take_picture()
            x = [(cams[k].get_picture("Color") * 255).clip(0, 255).astype("uint8")[:, :, :3] for k in keys]
            x = [to256(jpeg(np.ascontiguousarray(v), jp) if jp else v, img) for v in x]
            q = env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState()
        if ee:  # (prec) IDM 的末端裁剪流：两只手的夹爪中心投到头部相机（活相机的 cam2world / 内参），在 256² 头部帧上各裁 ee×ee（probwm/eecrop.py，同训练数据）
            from .eecrop import crop
            hc = cams["rgb"]
            el, er, E_, K_ = (np.asarray(v, np.float64) for v in (env.robot.get_left_ee_pose(), env.robot.get_right_ee_pose(), hc.get_model_matrix(), hc.get_intrinsic_matrix()))
            x = [x[0], crop(x[0][None], el[None], er[None], ee, E_, K_)[0]]
            if os.environ.get("PREC_EEDBG"):  # 冒烟核对：活相机矩阵与末端位姿逐帧追加到这个文件
                with open(os.environ["PREC_EEDBG"], "a") as f_:
                    f_.write(json.dumps({"E": E_.tolist(), "K": K_.tolist(), "ee_l": el.tolist(), "ee_r": er.tolist()}) + "\n")
        return (tuple(x) if len(x) > 1 else x[0]), np.asarray(q, np.float32)

    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            task, seed, eid = arg
            if task not in envs:
                envs[task] = (class_decorator(task), task_args(task, cfg, eval_mode=True))
                if os.environ.get("RT_INSTR"):  # (rteasy 11-17) --instr：覆盖 cfg 的 eval_instruction（如 demo_clean + unseen = 干净场景 + 没见过的指令）
                    envs[task][1]["eval_instruction"] = os.environ["RT_INSTR"]
            env, a = envs[task]
            try:  # 专家检查
                _setup(env, now_ep_num=eid, seed=seed, is_test=True, **a)
                if orc:  # (rteasy) --oracle_plan：专家这一遍每到采集存帧时刻（take_dense_action 里按 save_freq 调 _take_picture，同训练 demo）截一帧头部（同评测渲染 / JPEG / 256²）与 qpos
                    efr = []
                    env._take_picture = lambda: efr.append(view(env))
                info = env.play_once()
                if orc:
                    env.__dict__.pop("_take_picture", None)
                close(env)
                ok, why = bool(env.plan_success and env.check_success()), "expert_failed"
            except UnStableError:
                close(env)
                ok, why = False, "unstable"
            except Exception as e:  # 异常可能是 worker 状态坏了（如之前 OOM 过）：报给主进程，换新 worker 重试后才定为无效种子
                close(env)
                conn.send(("error", f"expert_error {type(e).__name__}"))
                continue
            if ok:
                try:
                    _setup(env, now_ep_num=eid, seed=seed, is_test=True, **a)
                except UnStableError:
                    close(env)
                    ok, why = False, "unstable"
                except Exception as e:
                    close(env)
                    conn.send(("error", f"setup_error {type(e).__name__}"))
                    continue
            if not ok:
                conn.send(("skip", why))
                continue
            try:
                c = generate_episode_descriptions(task, [(info or {}).get("info", {})], 100, seed=seed)[0].get(a["eval_instruction"])
                lang = random.Random(f"{seed}:select").choice(c) if c else task
            except Exception:
                lang = task
            env.set_instruction(instruction=lang)
            conn.send(("ok", *view(env), lang, env.step_lim) + ((efr[::2],) if orc else ()))  # (rteasy) 专家帧隔一取一 = WM 帧率（每次决策 2 个路点）
        elif cmd == "step":
            env, a = envs[arg[0]]
            n, err, mids = 0, "", []
            try:
                for x in arg[1]:
                    env.take_action(np.asarray(x, np.float64), action_type="qpos")
                    n += 1
                    if env.eval_success or env.take_action_cnt >= env.step_lim:
                        break
                    if every and n % every == 0 and n < len(arg[1]):  # --obs_every：执行长动作段时，中间每 every 个路点也拍一帧，补进历史
                        mids.append(view(env))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            done = bool(err or env.eval_success or env.take_action_cnt >= env.step_lim)
            if done:
                n_done += 1
                es_ = end_state(env) if es else None  # (rteasy) --end_state：关环境前记末态
                close(env, n_done % a["clear_cache_freq"] == 0)
                conn.send((None, None, True, bool(env.eval_success), n, env.take_action_cnt, err) + ((es_,) if es else ()))
            else:
                conn.send((*view(env), False, False, n, env.take_action_cnt, "") + ((mids,) if mids else ()))  # 没有中途帧时仍是 7 项（与旧主进程兼容）
        else:
            break


def cl_args(**kw):
    """closed_loop 的全部默认参数（截获它 main() 里的 parse_args，空命令行；别的实验以后加的选项自动取默认）+ 覆盖。"""
    from . import closed_loop

    class Got(Exception):
        pass
    orig = argparse.ArgumentParser.parse_args

    def grab(self, *_, **__):
        raise Got(orig(self, []))
    argparse.ArgumentParser.parse_args = grab
    try:
        closed_loop.main()
    except Got as e:
        ns = e.args[0]
    finally:
        argparse.ArgumentParser.parse_args = orig
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def summarize(eps):
    per = {}
    for e in eps:
        per.setdefault((e["cfg"], e["task"]), []).append(e["success"])
    rate = {f"{c}/{t}": float(np.mean(v)) for (c, t), v in sorted(per.items())}
    for k, r in rate.items():
        print(f"  {k:<40} {r:.3f} ({len(per[tuple(k.split('/'))])} ep)")
    for c in sorted({c for c, _ in per}):
        v = [r for k, r in rate.items() if k.startswith(c + "/")]
        print(f"  mean {c}: {np.mean(v):.3f} over {len(v)} tasks")
    return rate


def main():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a("--policy", default="wm+idm", choices=["wm+idm", "bc"])
    a("--wm", default="", help="bc 也需要：norm、编码器、语言表"); a("--idm", default=""); a("--bc", default="")
    a("--sampler", default="scratch2"); a("--cfg_w", type=float, default=3.0, help="WM 的语言 CFG"); a("--cfg_far", type=float, default=-1.0, help="≥0：远槽（时距 ≥ 9）的 CFG（closed_loop --cfg_far），近槽仍用 --cfg_w"); a("--fsteps", type=int, default=2)
    a("--canon", default="", help="规范化器 ckpt（canon.py；用 clean / 我们的 DR 重渲染配对训练）"); a("--canon_steps", type=int, default=4)
    a("--poe", default="", help="w[,steps[,thr]]：门控 PoE（closed_loop --poe，需 --canon、--prior）"); a("--prior", default="runs/rt/prior/ckpt.pt")
    a("--cfg", default="demo_clean", help="demo_clean = Easy，demo_randomized = Hard")
    a("--tasks", default=",".join(PILOT)); a("--n_ep", type=int, default=100); a("--seed", type=int, default=0, help="官方 seed：种子从 100000·(1+seed) 起")
    a("--exec", type=int, default=2, help="每次决策执行的路点数（2 = 一个 WM 帧）")
    a("--roll", default="", help="滚动去噪（closed_loop --roll）；MSF / MSN 用 1,0,8")
    a("--obs", default="cams", choices=["cams", "full"], help="cams：只拍策略要用的相机（默认）；full：官方 get_obs（全部相机）")
    a("--jpeg", default="", choices=["", "legacy", "std"], help="观测先过一遍训练数据同款 JPEG：legacy = 官方数据集，std = 我们按官方脚本采的（XPolicyLab）")
    a("--ov", default="", help="(rteasy 11-16) 预测-校正 OV（closed_loop Policy.ov_sub，pro 的 R3-only wm 模式）：wm,3,thr_hi,thr_lo,cap——上一次决策对当前帧的干净估计与观测的 mean 统计量超阈值 → 用估计顶上；比较 / 替换在 WM 看到的那份特征（canon 之后）上做；RoboTwin 没有陈旧 / 黑屏旗标；默认关")
    a("--instr", default="", choices=["", "seen", "unseen"], help="(rteasy 11-17) 覆盖 cfg 的 eval_instruction：demo_clean + unseen = 语言隔离测（干净场景 + 没见过的指令）；默认按 cfg（Easy seen、Hard unseen）")
    a("--end_state", action="store_true", help="(rteasy) 每个 episode 结束时记末态进结果 json 的 end：任务对象（env 上的 Actor / 铰接体）位姿与关节、两夹爪是否张开、接触对（失败归类：物体–目标距离、是否仍夹着）")
    a("--seed_list", default="", help="(rteasy) JSON {任务: [种子, …]}：只跑这些种子（专家失败的直接跳过、不补），用于失败种子重跑；任务集取 JSON 的键")
    a("--recycle", type=int, default=20, help="worker 每做完这么多个 episode 重起（显存上涨）；0 = 不重起")
    a("--fixed_noise", action="store_true", help="flow 动作头每个 episode 固定一份初始噪声（相邻决策落在同一模态）")
    a("--act_roll", type=float, default=0.0, help=">0：动作滚动——上一个动作块平移 --exec 个路点（尾部补最后一个）加噪到 τ0 = 这个值作初值再去噪")
    a("--ensemble", type=float, default=-1, help="≥0：ACT 式时间集成的 m（权重 exp(−m·i)，i = 0 为最早的预测；ACT 用 0.01）；<0 关")
    a("--sim_delay", type=int, default=0, help="仿真推理延迟 d 个路点（异步执行的参考）：拿到观测、开始算新动作块时，机器人照上一块再执行 d 个路点（已发出，新块改不了），"
                                               "新块从第 d 位接上；每次决策间仍执行 --exec 个路点，新回合的第一块没有前缀（等它算完再动）；"
                                               "--ensemble 按绝对路点时刻对齐（只集成新发出的那几位）；只接 wm+idm；默认 0 = 同步执行（原行为）")
    a("--rtc", action="store_true", help="配 --sim_delay：把这 d 个已发出的路点作前缀喂给训练时 RTC 的 IDM（closed_loop Policy.rtc_prefix；--idm / --idm_ens 都须是 "
                                         "idm train --rtc_dmax 训的）；不开 = 过期路点直接跳过；d = 0 时等价于不开")
    a("--obs_every", type=int, default=0, help="--exec 大于它时，执行动作段的中途每这么多个路点也拍一帧补进历史（历史帧间隔与训练一致，2）；BC，或 (rteasy) wm+idm 的 MSF 滚动（缓存每次决策平移 exec / 2 帧）")
    a("--video", type=int, default=0, help="每任务前这么多个 episode 存下策略看到的头部帧和动作（<out 去掉 .json>_video/<任务>_<种子>.npz，看失败方式用）")
    a("--n_env", type=int, default=20); a("--sim_gpus", default="", help="worker 的 CUDA_VISIBLE_DEVICES（轮流分配），空 = 继承")
    a("--img", type=int, default=256); a("--text_model", default="google/flan-t5-large")
    a("--seed_cache", default="runs/rt/seeds"); a("--out", default=""); a("--merge", nargs="*")
    a("--timeout", type=float, default=1800, help="一次专家检查 / 一次执行超过这么多秒算卡死（SAPIEN 偶发），换 worker")
    a("--idm_tau", default="", help="(xden) closed_loop --idm_tau（pre：噪声感知 IDM 读 WM 去噪前的窗口）；默认关")
    a("--t0_ignore", action="store_true", help="(gawm X2F5CU 精度模式) closed_loop --t0_ignore：canon_slot WM 的 t+0 槽不替换观测；默认关")
    a("--t0_sde", type=float, default=-1.0, help="(rteasy 11-17) closed_loop --t0_sde τ0：canon_slot WM 的 t+0 槽从当前观测加噪到 τ0 起去噪（SDEdit 式投影）；默认关")
    a("--wm_data", default="", help="(gawm X2F5CU) closed_loop --wm_data：WM 训练数据根改用本机同内容的路径；默认空")
    a("--idm_stale", default="", help="(xden2) closed_loop --idm_stale pred|obs（IDM 读上一次决策的干净计划，WM 步在动作之后；可叠 canon_slot / canon_hist 的 WM）；默认关")
    a("--replan", default="", help="(hz2) closed_loop --replan thr[,mode[,cool[,stat]]]（惊讶触发重规划）；触发的 env 同时清空 --ensemble 的旧动作块；"
                                   "逐决策的三个统计量 [mean, top, cos] 与触发的决策序号写进 episode 的 surp / trig；默认关")
    a("--any_far", default="", help="(hz2) closed_loop --any_far：任意时距远槽 WM 查询的远槽时距（如 16,24,32）；默认关")
    a("--det_cfg", action="store_true", help="(gawm XF5D) closed_loop --det_cfg：--mode det 的 WM 同步路径也做 CFG；默认关")
    a("--near_only", action="store_true", help="(gawm) closed_loop --near_only：带远槽训练的 WM（train --far_aux / --far_drop）推理时只用近槽（远槽不去噪、不滚动）；默认关")
    a("--grip_idm", default="", help="(prec) 两个夹爪维（6 / 13）换成这个 IDM（只读头部历史 + 同一计划 + 本体，不带额外流）在同一次决策上的输出；"
                                     "如主 IDM 是末端裁剪头 idmPLee、夹爪仍用 idmPL；默认关")
    a("--eecrop", default="80,160", help="(prec) IDM 用 --wkeys eecrop 训练时：末端裁剪边长（256² 头部帧上）、编码分辨率；与 scripts/prec_crop.py 建特征时一致")
    a("--ens_gwin", type=int, default=0, help="(prec) >0：--ensemble 时两个夹爪维（6 / 13）只集成最新这么多个动作块（臂维照旧）；默认关")
    a("--ens_contact", default="", help="(prec) W[,thr[,ahead[,after]]]：--ensemble 时接触阶段全部维只集成最新 W 个动作块。接触 = 最新块前 ahead 个路点里"
                                        "任一夹爪开度与当前 qpos 差 > thr（默认 0.1、8），或夹爪在动（与上次决策的 qpos 差 > 0.02）之后 after 次决策内（默认 3）；"
                                        "episode 记 ct = 接触决策占比；默认关")
    a("--log_roll", default="", help="(flywheel) 每个计入结果的 episode 存 <DIR>/<cfg>/<任务>/<种子>.npz：逐决策的原始 DINO 特征（编码器输出，未标准化）、"
                                     "进历史的帧（开 canon 时 = 规范化后）、本体、发出 / 实际执行的路点、IDM 原始动作块（格式见 README）；只做记录、不改任何计算；默认关")
    a("--wm_act", default="", help="(uwm) closed_loop --wm_act clean|pre（train --uwm 的 WM 用自己的动作 token 出动作）；默认关")
    a("--wm_act_steps", type=int, default=4, help="(uwm) closed_loop --wm_act_steps"); a("--wm_act_mix", type=float, default=1.0, help="(uwm) closed_loop --wm_act_mix（< 1：与 IDM 平均）")
    a("--n_avg", type=int, default=1, help="(rteasy) closed_loop --n_avg：同一计划下同一个 IDM 采这么多个动作块取均值（--idm_ens 的零训练对照）；默认 1 = 关")
    a("--roll_k", type=int, default=0, help="(rteasy) closed_loop --roll_k：每次决策批量采 K 个计划取共识（--roll_sel medoid | cluster）；episode 另记分歧度 pdiv；默认关")
    a("--roll_sel", default="medoid", choices=["medoid", "cluster"])
    a("--roll_kc", type=int, default=0, help="(rteasy) closed_loop --roll_kc：K 个计划分块前向（每块几份；0 = 一次）")
    a("--goal_first", action="store_true", help="(rteasy 臂 G) closed_loop --goal_first：目标先行两阶段滚动（配 train --goal_first 的 WM）；默认关")
    a("--gf_fresh", type=int, default=4, help="(rteasy) closed_loop --gf_fresh")
    a("--lead_k", type=int, default=0, help="(rteasy 臂 G2) closed_loop --lead_k：远槽领先的多步去噪；默认 0 = 关")
    a("--wm_ens", default="", help="(rteasy) closed_loop --wm_ens：多 WM 计划按槽平均；默认关")
    a("--ph2", action="store_true", help="(rteasy) 配 --exec 1：每个路点都重新决策，用两套交错的相位状态（WM / 规范化原始 / 腕部 / 本体历史与滚动缓存各一套，"
      "按本 episode 已执行路点数的奇偶选），每套内帧间隔仍是 2 个路点（同训练）；ACT 集成仍按真实 env 跨两相位做；默认关")
    a("--wm_xtok_null", action="store_true", help="(rteasy) WM 带双腕上下文时把双腕 token 置 0（= 训练里的腕部缺省；Hard 上的不带腕部对照）")
    a("--oracle_plan", default="", choices=["", "q", "t", "tf", "ts"], help="(rteasy) 闭环上限诊断：IDM 读的计划换成同种子专家检查那一遍的头部帧（按采集频率截、隔一取一 = WM 帧率，"
                                                                 "同编码 / canon）。对齐：q = 按当前 qpos 找最近的专家帧（只往前、窗口 10 帧），t = 按已执行路点数 / 2；WM 照常滚动；"
                                                                 "tf = 部分 oracle：只把远槽的缓存估计换成专家帧（按步数对齐），近槽仍由 WM 联合去噪；"
                                                                 "ts = 影子：不换计划（行为同不开），只记每次决策 WM 计划各槽与对齐专家帧的均方差（及复制当前帧的参照）；默认关")
    a("--idm_ens", default="", help="(rteasy) closed_loop --idm_ens：逗号分隔的另几个 IDM（同配方不同训练种子），与 --idm 的动作块等权平均；默认关")
    a("--idm_graph", action="store_true", help="(rteasy 11-18) closed_loop --idm_graph：IDM 骨干捕获 CUDA graph；默认关")
    a("--idm_ens_par", action="store_true", help="(rteasy 11-18) closed_loop --idm_ens_par：集成并行（vmap 一次前向）；默认关")
    a("--idm_ms", action="store_true", help="(rteasy 11-19) closed_loop --idm_ms：集成成员多流并发（逐位同顺序版）；默认关")
    a("--compile", default="", help="closed_loop --compile（torch.compile 模式；配 --graphs 用 default 或 max-autotune-no-cudagraphs）；默认空 = 关")
    a("--compile_on", default="wm,canon,idm,enc", help="closed_loop --compile_on")
    a("--graphs", action="store_true", help="(11-19) closed_loop --graphs：编码器 / WM / IDM 速度场 / canon 的前向按输入形状捕获 CUDA graph（结果逐位不变）；默认关")
    a = p.parse_args()
    if a.instr:  # worker 由 spawn 起，继承环境变量
        os.environ["RT_INSTR"] = a.instr
    assert not a.oracle_plan or (a.policy == "wm+idm" and a.roll and not a.log_roll), "--oracle_plan 只接 wm+idm 的 --roll 路径"
    assert not (a.idm_ens or a.n_avg > 1) or not (a.fixed_noise or a.act_roll > 0 or a.grip_idm or a.wm_act), "--idm_ens 不与 --fixed_noise / --act_roll / --grip_idm / --wm_act 同用"
    assert a.sim_delay >= 0 and (not a.sim_delay or (a.policy == "wm+idm" and not (a.ph2 or a.act_roll > 0 or a.oracle_plan or a.log_roll or a.replan or a.ens_gwin or a.ens_contact
                                                                                    or a.grip_idm or a.exec > a.obs_every > 0))), \
        "--sim_delay 只接 wm+idm，不与 --ph2 / --act_roll / --oracle_plan / --log_roll / --replan / --ens_gwin / --ens_contact / --grip_idm / --obs_every 同用"
    assert not a.rtc or not a.idm_ens_par, "--rtc 不与 --idm_ens_par 同用"
    if a.merge:
        return summarize([e for f in a.merge for e in json.loads(Path(f).read_text())["episodes"]])
    torch.manual_seed(a.seed)
    tasks = a.tasks.split(",")
    pck = {"bc": a.bc, "wm+idm": a.idm}[a.policy]
    pargs = torch.load(pck, map_location="cpu")["args"]
    eec = [int(x) for x in a.eecrop.split(",")] if pargs.get("wkeys") == "eecrop" else []  # (prec) IDM 的额外流是末端裁剪（idm --wkeys eecrop）：裁剪边长、编码分辨率
    assert not eec or a.obs == "cams", "--eecrop 只接 --obs cams"
    xw = a.policy == "wm+idm" and bool(torch.load(a.wm, map_location="cpu", mmap=True)["args"].get("tgt_tok", 0))  # (rteasy) WM 上下文带双腕（train --tgt_tok 的多视角 WM，dinov3_vits16hw）
    assert not xw or not eec, "WM 带双腕上下文时 IDM 不能另读末端裁剪"  # (rteasy 11-16) IDM 另读双腕可以：见 encode 的 xw 分支
    keys = ["rgb"] + (["wl", "wr"] if (pargs.get("wrist", 0) > 0 and not eec) or xw else [])
    prop = bool(pargs.get("proprio"))
    assert not prop or pargs["proprio"] == "qpos", "RoboTwin 的本体感觉只有 qpos（14 维，同 get_obs 的 joint_action/vector）"

    n, ctx = a.n_env, mp.get_context("spawn")
    gl, env0 = [g for g in a.sim_gpus.split(",") if g], os.environ.get("CUDA_VISIBLE_DEVICES")
    pipes, procs = [None] * n, [None] * n

    def spawn(i):  # SAPIEN / CuRobo 用 worker 自己的卡；卡死的 worker（SAPIEN 偶发卡在 do_poll）杀掉后在同一槽位重起
        if gl:
            os.environ["CUDA_VISIBLE_DEVICES"] = gl[i % len(gl)]
        pa, ch = ctx.Pipe()
        procs[i] = ctx.Process(target=worker, args=(ch, a.cfg, a.img, keys, a.obs, a.jpeg, a.obs_every if a.exec > a.obs_every > 0 else 0, eec[0] if eec else 0)
                               + (bool(a.oracle_plan), bool(a.end_state)), daemon=True)
        procs[i].start()
        pipes[i] = pa
        if gl:
            os.environ.pop("CUDA_VISIBLE_DEVICES") if env0 is None else os.environ.__setitem__("CUDA_VISIBLE_DEVICES", env0)

    for i in range(n):  # 先起子进程（spawn，不继承 CUDA 上下文）
        spawn(i)

    def respawn(w):
        procs[w].kill()
        procs[w].join(5)
        spawn(w)

    from .closed_loop import Policy
    from .encode import pool, text_encoder
    dev = torch.device("cuda")

    pol = Policy(cl_args(policy=a.policy, wm=a.wm, idm=a.idm, bc=a.bc, text="", sampler=a.sampler, cfg=a.cfg_w, cfg_far=a.cfg_far, plan=1, exec=a.exec, canon=a.canon,
                         canon_steps=a.canon_steps, canon_tau0=0.0, poe=a.poe, prior=a.prior, fsteps=a.fsteps, roll=a.roll, seed=a.seed, n_env=n * (2 if a.ph2 else 1),
                         **({"idm_tau": a.idm_tau} if a.idm_tau else {}), **({"idm_stale": a.idm_stale} if a.idm_stale else {}), **({"replan": a.replan} if a.replan else {}),
                         **({"any_far": a.any_far} if a.any_far else {}), **({"near_only": True} if a.near_only else {}), **({"det_cfg": True} if a.det_cfg else {}),
                         **({"roll_shift": a.exec // 2} if a.exec > a.obs_every > 0 else {}), **({"idm_ens": a.idm_ens} if a.idm_ens else {}), **({"idm_ens_par": True} if a.idm_ens_par else {}), **({"idm_graph": True} if a.idm_graph else {}), **({"idm_ms": True} if a.idm_ms else {}), **({"graphs": True} if a.graphs else {}), **({"compile": a.compile, "compile_on": a.compile_on} if a.compile else {}),
                         **({"n_avg": a.n_avg} if a.n_avg > 1 else {}), **({"roll_k": a.roll_k, "roll_sel": a.roll_sel, "roll_kc": a.roll_kc} if a.roll_k > 1 else {}),
                         **({"goal_first": True, "gf_fresh": a.gf_fresh} if a.goal_first else {}), **({"lead_k": a.lead_k, "gf_fresh": a.gf_fresh} if a.lead_k else {}),
                         **({"wm_ens": a.wm_ens} if a.wm_ens else {}), **({"wm_data": a.wm_data} if a.wm_data else {}), **({"ov": a.ov, "ov_log": True} if a.ov else {}), **({"t0_ignore": True} if a.t0_ignore else {}), **({"t0_sde": a.t0_sde} if a.t0_sde >= 0 else {})), dev)  # (rteasy) --exec 4 --obs_every 2：中途帧补进历史，MSF 滚动缓存每次决策平移 exec / 2 帧
    assert not (getattr(pol, "clip", None) and (a.ph2 or a.exec > a.obs_every > 0 or eec)), \
        "(rteasy) 视频编码器（V-JEPA 2.1 短窗）只接每次决策编码一次帧：不与 --ph2 / --obs_every 中途帧 / 末端裁剪同用（11-17 起可配双腕 WM：两个腕部视角各一份帧缓存）"
    if a.wm_act:  # (uwm) WM 动作 token：建好 Policy 后设参数再初始化（不改上面那次 cl_args 调用）
        pol.a.wm_act, pol.a.wm_act_steps, pol.a.wm_act_mix = a.wm_act, a.wm_act_steps, a.wm_act_mix
        pol.uwm_init()
    if a.grip_idm:  # (prec) 夹爪维用另一个 IDM
        from . import idm as idm_
        gnet, gna, gst = idm_.make(torch.load(a.grip_idm, map_location="cpu"), dev)
        gnet.steps = a.fsteps
        assert not gna.wrist and gna.H == pol.na.H and gna.ih == pol.na.ih and not idm_.fextra(gna) and not idm_.fars(gna), "--grip_idm 只支持与主 IDM 同 H / ih 的普通头"
    tap = [None]  # (flywheel) --log_roll：编码器这一次的原始输出（只截获返回值，计算不变）
    if a.log_roll:
        assert not pol.nw and not eec and not (a.exec > a.obs_every > 0) and pol.proj is None and pol.ad is None and not pol.clip, \
            "--log_roll 只接只读头部相机的 DINO 特征策略（无腕部 / 末端裁剪 / 中途帧 / PCA / 适配器 / 视频编码器）"
        enc0 = pol.enc

        def enc_tap(*x, **k):
            tap[0] = enc0(*x, **k)
            return tap[0]
        pol.enc = enc_tap
    tm = a.text_model.split("/")[-1]
    assert pol.wa.text == tm or pol.wa.text.startswith(tm + "_"), f"--text_model {a.text_model} 与 WM 的语言表 {pol.wa.text} 不一致"
    embed, rows, ntrunc = text_encoder(a.text_model, dev, maxlen=pol.temb.shape[-2]), {}, [0]  # 与训练表同长度（32 / 64）

    def row(lang):  # 指令 → 语言表行号（评测指令在线编码后追加）
        if lang not in rows:
            e, m, k = embed([lang])
            ntrunc[0] += k
            rows[lang] = len(pol.temb)
            pol.temb, pol.tmask = torch.cat([pol.temb, e.to(pol.temb.dtype)]), torch.cat([pol.tmask, m])
        return rows[lang]

    T, D = pol.dims
    wT = pargs.get("wT") or T

    if eec:  # (prec) 末端裁剪的编码器：同一 DINOv3（层同头部特征，Policy 也从这个 info.json 建编码器），分辨率 = 训练特征的（scripts/prec_crop.py）
        from .encode import ENCODERS, load_dinov3
        einf = json.loads((Path(pol.wa.data) / f"feats/{pol.wa.encoder}/info.json").read_text())
        efwd = load_dinov3(ENCODERS[einf.get("base", pol.wa.encoder)][0], einf["layers"], eec[1], dev)
        assert wT == 2 * (eec[1] // 16) ** 2, f"--eecrop 的分辨率与 IDM 的末端流 token 数不一致（{wT}）"

    @torch.no_grad()
    def encode(ob, *sf):  # head 同 Policy.encode；腕部：左 / 右各池化到 wT/2 个 token 后拼接，用 IDM ckpt 的 wnorm 标准化
        if xw:  # (rteasy) ob 每项 = (头部, 左腕, 右腕)：头部同 Policy.encode（含 canon）；双腕过同一编码器、2×2 池化到 8×8、用 WM 的 norm 标准化，
            nw_, pol.nw = pol.nw, False  # (rteasy 11-17) 头部单独编码：IDM 也读腕部（pol.nw）时 Policy.encode 会把每项当 (头部, 腕部) 对，这里只给头部，先关掉再恢复
            try:
                z = pol.encode([o[0] for o in ob], *sf)  # 拼在头部 token 后（左 64 + 右 64，同 scripts/rteasy_wenc.py）；--wm_xtok_null 时双腕全 0（= 训练里的"腕部缺省"）；sf = 视频编码器的 (slots, fresh)
            finally:
                pol.nw = nw_
            nx = T - z.shape[1]
            if a.wm_xtok_null:
                return torch.cat([z, torch.zeros(len(ob), nx, z.shape[2], dtype=z.dtype, device=dev)], 1), None
            x = torch.from_numpy(np.stack([o[j] for j in (1, 2) for o in ob])).to(dev)
            if getattr(pol, "clip", None):  # (rteasy 11-17) 视频编码器（V-JEPA 短窗）：两个腕部视角各自按 env 维护帧缓存，各取 clip 帧窗（新 episode 用第一帧补满），同头部的 fbuf
                slots, fresh = sf
                L = 1 - pol.clip[0]
                if getattr(pol, "wxbuf", None) is None:
                    pol.wxbuf = torch.zeros(2, pol.a.n_env, L, *x.shape[1:], dtype=x.dtype, device=dev)
                xs = []
                for v, xv in enumerate(x.chunk(2)):
                    b = torch.cat([pol.wxbuf[v, slots, 1:], xv[:, None]], 1)
                    pol.wxbuf[v, slots] = b = torch.where(fresh.view(-1, 1, 1, 1, 1), xv[:, None].expand_as(b), b)
                    xs.append(b[:, [L - 1 + o for o in pol.clip]])
                x = torch.cat(xs)
            with torch.autocast("cuda", torch.bfloat16):
                f = pol.enc(x)
            wl_, wr_ = pool(f, int(round((f.shape[1] / (nx // 2)) ** 0.5))).chunk(2)
            zw_ = None
            if pol.nw:  # (rteasy 11-16) IDM 也读双腕（idm --wrist）：同一次编码，按 IDM 的腕部 token 数池化、用 IDM ckpt 的 wnorm
                wl0, wr0 = f.chunk(2)
                k_ = int(round((f.shape[1] / (wT // 2)) ** 0.5))
                zw_ = ((torch.cat([pool(wl0, k_), pool(wr0, k_)], 1).float() - pol.wmu) / pol.wsd).half()
            return torch.cat([z, ((torch.cat([wl_, wr_], 1).float() - pol.mu) / pol.sd).half()], 1), zw_
        if not pol.nw:
            return pol.encode(ob, *sf), None  # (rteasy) sf = (slots, fresh)：视频编码器按 env 维护帧缓存（同 closed_loop 的 LIBERO 循环）
        if eec:  # (prec) ob 每项 = (头部 256², 两只手的裁剪 (2,C,C,3))；头部同下面腕部分支（Policy.encode 的无腕部路径）
            with torch.autocast("cuda", torch.bfloat16):
                h = pol.enc(torch.from_numpy(np.stack([o[0] for o in ob])).to(dev))
                w = efwd(torch.from_numpy(np.stack([o[1] for o in ob])).to(dev).flatten(0, 1)).float().view(len(ob), -1, D)
            z = ((h.float() - pol.mu) / pol.sd).half()
            return (pol.canonize(z) if a.canon and not pol.rawh else z), ((w - pol.wmu) / pol.wsd).half()
        with torch.autocast("cuda", torch.bfloat16):
            f = pol.enc(torch.from_numpy(np.stack([o[j] for j in range(3) for o in ob])).to(dev))
        h, wl, wr = f.chunk(3)
        k = int(round((T / (wT // 2)) ** 0.5))
        w = torch.cat([pool(wl, k), pool(wr, k)], 1)
        z = ((h.float() - pol.mu) / pol.sd).half()
        return (pol.canonize(z) if a.canon and not pol.rawh else z), ((w.float() - pol.wmu) / pol.wsd).half()

    assert not a.ph2 or (a.exec == 1 and not a.obs_every and not a.oracle_plan and a.roll_k <= 1 and not a.replan and not a.log_roll), \
        "--ph2 只配 --exec 1，不与 --obs_every / --oracle_plan / --roll_k / --replan / --log_roll 同用"
    nv = n * (2 if a.ph2 else 1)  # (rteasy) --ph2：env w 的第 p 相位状态在槽 w + n·p
    fv = [True] * nv  # (rteasy) --ph2：各相位槽是否还没决策过（新 episode 两个相位都从头）
    hist = torch.zeros(nv, pol.K, T, D, dtype=torch.float16, device=dev)
    rhist = torch.zeros_like(hist) if pol.rawh else None  # 原始（未规范化）历史：--poe 时规范化器读当前原始帧
    whist = torch.zeros(nv, pol.nw, wT, D, dtype=torch.float16, device=dev) if pol.nw else None
    phist = torch.zeros(nv, pol.npf, pol.na.pdim, device=dev) if pol.npf else None
    hc = None  # (gawm XF5M) train --self_hist M,g 的 WM：每个 env 维护本 episode 已观测帧的池化缓存（同 mikasa_eval.HistCache），每次决策经 pol.sh_demo 进 WM
    if getattr(pol.wa, "self_hist", ""):
        from .mikasa_eval import HistCache
        hc = HistCache(nv, *(int(x) for x in pol.wa.self_hist.split(",")), T, D, pol.K, dev, wa=pol.wa)
        print(f"self_hist WM：历史缓存 M,g = {pol.wa.self_hist}（{nv} 个状态槽）", flush=True)

    # 种子调度：每任务按种子升序派发；在途数 ≤ 还缺的有效种子数 → 有效种子集合 = 前 n_ep 个（同官方顺序执行）
    st0 = 100000 * (1 + a.seed)
    sl = {t: list(v) for t, v in json.loads(Path(a.seed_list).read_text()).items() if t in tasks} if a.seed_list else None  # (rteasy) --seed_list
    cdir = Path(a.seed_cache) / a.cfg
    cdir.mkdir(parents=True, exist_ok=True)
    cache = {t: json.loads((cdir / f"{t}.json").read_text()) if (cdir / f"{t}.json").exists() else {} for t in tasks}
    nxt, valid, inflight, eid = {t: st0 for t in tasks}, {t: 0 for t in tasks}, {t: 0 for t in tasks}, {t: 0 for t in tasks}
    job, state, fresh, obs, prv, lang, t0_, tr = [None] * n, [""] * n, [True] * n, [None] * n, [None] * n, [""] * n, [0.0] * n, [0.0] * n
    nact, last, nep = [0] * n, [None] * n, [0] * n  # 本 episode 已成功执行的路点数；worker 上一个任务、已做的 episode 数（--recycle）
    vid, nvid = [None] * n, {}  # --video：在录的 episode 帧、各任务已存的个数
    pend = [[] for _ in range(n)]  # --obs_every：待补进历史的中途帧
    ens = [[] for _ in range(n)]  # --ensemble：各 env 历次预测里还没执行到的部分（动作块, 年龄）
    dq = [None] * n  # --sim_delay：各 env 已发出、下一次决策推理期间会执行的 d 个路点（原始单位）
    ctv = [float(x) for x in a.ens_contact.split(",")] if a.ens_contact else []  # (prec) --ens_contact：W, thr, ahead, after
    ctv += [0.1, 8, 3][len(ctv) - 1:] if ctv else []
    cdn, cq, cnt_ct = [0] * n, [None] * n, [[0, 0] for _ in range(n)]  # (prec) 夹爪在动后的倒计数、上次决策的 qpos、本 episode [接触决策数, 决策数]
    fz, prevn = [None] * n, [None] * n  # --fixed_noise：各 env 本 episode 的固定噪声；--act_roll：上一次的标准化动作块
    eps, skipped, tg = [], [], 0.0
    rsv, rtg = [[] for _ in range(n)], [[] for _ in range(n)]  # (hz2) --replan：各 env 本 episode 逐决策的统计量、触发的决策序号
    lg = [None] * n  # (flywheel) --log_roll：各 env 本 episode 的逐决策记录
    orq, orcF, orcQ, orcJ = [None] * n, {}, {}, {}  # (rteasy) --oracle_plan：待编码的专家帧、专家特征 / qpos、当前对齐到的专家 WM 帧
    orcE, orcC = {}, {}  # (rteasy) --oracle_plan ts：各 env 本 episode 逐决策的各槽计划误差 / 复制当前帧误差
    pdv = [[] for _ in range(n)]  # (rteasy) --roll_k：各 env 本 episode 逐决策的 K 计划分歧度
    ovs = [[0, 0, 0] for _ in range(n)]  # (rteasy) --ov：各 env 本集 [决策数, 有估计, 被替换]

    redo, tries = [], {}  # 基础设施故障（worker 崩溃 / 卡死）的 (任务, 种子)：同一种子从头重做（最多 2 次），不算策略失败

    def infra(w, why):  # worker 崩溃或卡死：重起 worker，这个种子重做
        t, s = job[w]
        respawn(w)
        inflight[t] -= 1
        tries[(t, s)] = tries.get((t, s), 0) + 1
        if tries[(t, s)] <= 2:
            redo.append((t, s))
        else:
            skipped.append((t, s, why))  # 反复出故障的种子跳过（不记进缓存，与种子无关）
        print(f"worker {w} {why}（{t} 种子 {s}），重起；{'重做该种子' if tries[(t, s)] <= 2 else '跳过该种子'}", flush=True)
        nep[w], last[w] = 0, None

    def take(w):  # 下一个 (任务, 种子)：先重做出过基础设施故障的；同一 worker 优先接着做上一个任务；跳过缓存里已知无效的种子
        if redo:
            t, s = redo.pop(0)
            inflight[t] += 1
            return t, s
        if sl is not None:  # (rteasy) --seed_list：按清单派发
            for t in ([last[w]] if last[w] in tasks and sl.get(last[w]) else []) + [t for t in tasks if sl.get(t)]:
                inflight[t] += 1
                return t, sl[t].pop(0)
            return None
        for t in ([last[w]] if last[w] in tasks else []) + sorted(tasks, key=lambda t: valid[t] + inflight[t]):
            if valid[t] + inflight[t] < a.n_ep:
                while cache[t].get(str(nxt[t]), "ok") != "ok":
                    nxt[t] += 1
                s, nxt[t] = nxt[t], nxt[t] + 1
                inflight[t] += 1
                return t, s
        return None

    def start(w):  # 异步：发出 reset（专家检查要几十秒），其他 env 照常推进
        if a.recycle and nep[w] >= a.recycle:
            respawn(w)
            nep[w], last[w] = 0, None
        job[w] = take(w)
        state[w] = "" if job[w] is None else "reset"
        if job[w] is not None:
            tr[w] = time.time()
            pipes[w].send(("reset", (job[w][0], job[w][1], eid[job[w][0]])))

    def recv(w):
        r = pipes[w].recv()
        if isinstance(r, Exception):
            raise r
        return r

    def poll_resets(block):
        ws = [w for w in range(n) if state[w] == "reset"]
        if block and ws:
            mp_wait([pipes[w] for w in ws], 60)
        for w in ws:
            if not pipes[w].poll():
                dead = not procs[w].is_alive()
                if dead or time.time() - tr[w] > a.timeout:  # 专家检查卡死 / worker 崩溃（如别人占满显存时 OOM）：换 worker，重做
                    infra(w, "worker 崩溃（reset）" if dead else "专家检查超时")
                    start(w)
                continue
            try:
                r = recv(w)
            except (EOFError, OSError):
                infra(w, "worker 崩溃（reset）")
                start(w)
                continue
            t, s = job[w]
            if r[0] == "error" and (tries.get((t, s), 0) == 0 or any(x in r[1] for x in ("Memory", "Accelerator", "CUDA", "Cuda"))):
                infra(w, f"专家检查异常 {r[1]}")  # 第一次报异常 / 显存类异常：换新 worker 重做，不进缓存（11-06 9022 上 OOM 后 worker 状态坏掉，连着 8000 个种子报 AttributeError 被误记成无效）
                start(w)
            elif r[0] in ("skip", "error"):  # 专家失败 / 不稳定 / 新 worker 上仍报异常：无效种子（同官方跳过），记进缓存
                cache[t][str(s)] = r[1]
                (cdir / f"{t}.json").write_text(json.dumps(cache[t]))
                inflight[t] -= 1
                skipped.append((t, s, r[1]))
                if r[0] == "error":
                    respawn(w)
                start(w)
            else:
                _, obs[w], prv[w], lang[w], _ = r[:5]
                if a.oracle_plan:  # (rteasy) 这个种子的专家帧，下一次决策前编码
                    orcF.pop(w, None), orcE.pop(w, None), orcC.pop(w, None)
                    orq[w] = r[5]
                state[w], fresh[w], t0_[w], nact[w], last[w] = "run", True, time.time(), 0, t
                ovs[w] = [0, 0, 0]  # (rteasy) --ov：本集 [决策数, 有估计的决策数, 被替换的决策数]
                fv[w] = fv[w + n * (nv > n)] = True
                if a.log_roll:  # (flywheel) 新 episode（含基础设施故障后同种子重做）：记录从头开始
                    lg[w] = {"f": [], "z": [], "q": [], "a": [], "k": [], "c": [], "lim": r[4]}
                vid[w] = [] if nvid.get(t, 0) < a.video else None
                pend[w] = []

    T0 = time.time()
    for w in range(n):
        start(w)
    while any(j is not None for j in job):
        poll_resets(block=not any(x == "run" for x in state))
        act_ = [w for w in range(n) if state[w] == "run"]
        if not act_:
            continue
        g0 = time.time()
        for w in act_:  # --obs_every：上一个动作段中途拍的帧按顺序补进历史（新 episode 不会有）
            for o_, pv_ in pend[w]:
                zm, zwm = encode([o_])
                hist[w] = torch.cat([hist[w, 1:], zm[0][None]], 0)
                if pol.nw:
                    whist[w] = torch.cat([whist[w, 1:], zwm[0][None]], 0)
                if pol.npf:
                    phist[w] = torch.cat([phist[w, 1:], torch.from_numpy(pv_).to(dev)[None]], 0)
            pend[w] = []
        idx = torch.tensor(act_, device=dev)
        (z, zw) = encode([obs[w] for w in act_], *((idx, torch.tensor([fresh[w] for w in act_], device=dev)) if getattr(pol, "clip", None) else ()))
        if getattr(pol, "ov", None):  # (rteasy 11-16) 预测-校正 OV：RoboTwin 没有陈旧 / 黑屏旗标（置 0），只靠 R3
            z = pol.ov_sub(z, idx, torch.tensor([fresh[w] for w in act_], device=dev), torch.zeros(len(act_), 2, dtype=torch.bool, device=dev))
            for i_, w_ in enumerate(act_):
                ovs[w_][0] += 1; ovs[w_][1] += int(pol.ov_last[1][i_]); ovs[w_][2] += int(pol.ov_last[3][i_])
        fr = torch.tensor([fresh[w] for w in act_], device=dev)[:, None, None, None]
        if a.ph2:  # (rteasy) 本次决策用这个相位的状态槽；该槽第一次决策时按新 episode 处理（历史全用当前帧、滚动从头采）
            vidx = [w + n * (nact[w] % 2) for w in act_]
            idx, fr = torch.tensor(vidx, device=dev), torch.tensor([fv[v] for v in vidx], device=dev)[:, None, None, None]
        ti = torch.tensor([row(lang[w]) for w in act_], device=dev)
        if pol.rawh:  # 同 closed_loop：原始历史 → 门控 PoE（--poe）或时序规范化器
            rh = torch.cat([rhist[idx, 1:], z[:, None]], 1)
            rhist[idx] = rh = torch.where(fr, z[:, None].expand_as(rh), rh)
            z = pol.poe(rh[:, -1], hist[idx], ti, fr.flatten(), idx) if a.poe else pol.canonize(rh)
        h = torch.cat([hist[idx, 1:], z[:, None]], 1)
        hist[idx] = h = torch.where(fr, z[:, None].expand_as(h), h)  # 新 episode：历史全用第一帧
        if hc is not None:  # (gawm XF5M) 当前帧（进 WM 历史的那一帧，开 canon 时是规范化后的）先进缓存，再按 sh_pick 取历史帧
            pol.sh_demo = hc.step(idx, fr.flatten(), z)
        if a.log_roll:  # (flywheel) 这次决策的原始特征、进历史的帧（标准化；开 canon 时是规范化后的）、本体
            fr_, zr_ = tap[0].cpu().numpy(), (z.cpu().numpy() if a.canon else None)
            for i, w in enumerate(act_):
                lg[w]["f"].append(fr_[i])
                lg[w]["q"].append(np.asarray(prv[w], np.float32))
                if a.canon:
                    lg[w]["z"].append(zr_[i])
        wh = ph = None
        if pol.nw:
            wh = torch.cat([whist[idx, 1:], zw[:, None]], 1)
            whist[idx] = wh = torch.where(fr, zw[:, None].expand_as(wh), wh)
        if pol.npf:
            pv = torch.from_numpy(np.stack([prv[w] for w in act_])).to(dev)
            ph = torch.cat([phist[idx, 1:], pv[:, None]], 1)
            phist[idx] = ph = torch.where(fr[..., 0], pv[:, None].expand_as(ph), ph)
        if a.fixed_noise or a.act_roll > 0:  # 只改 flow 动作头的采样（idm.Net 的推理属性）：固定噪声 / 动作滚动
            net = pol.net
            for i, w in enumerate(act_):
                if fresh[w] or fz[w] is None:
                    fz[w], prevn[w] = torch.randn(net.n_act, net.ad, device=dev), None
            if a.fixed_noise:
                net.fixed_eps = torch.stack([fz[w] for w in act_])
            if a.act_roll > 0:
                z0 = torch.zeros(net.n_act, net.ad, device=dev)
                sh = [z0 if prevn[w] is None else torch.cat([prevn[w][a.exec:], prevn[w][-1:].expand(a.exec, -1)]) for w in act_]
                net.warm_init = (torch.stack(sh), torch.tensor([prevn[w] is not None for w in act_], device=dev), a.act_roll)
        with torch.no_grad():
            if a.oracle_plan:  # (rteasy) oracle 计划：IDM 读的计划换成专家帧（closed_loop Policy.orc / orc_d，WM 照常滚动）
                for w in act_:
                    if orq[w] is not None:  # 新 episode：专家帧过同一编码器 / 标准化 / canon
                        # 专家帧与观测同一格式（末端裁剪 / 腕部时是元组），走同一个 encode(ob) 取头部特征（含 canon），只用第 0 项
                        orcF[w] = torch.cat([encode([x[0] for x in orq[w][k:k + 64]])[0] for k in range(0, len(orq[w]), 64)])
                        orcQ[w], orcJ[w], orq[w] = np.stack([x[1] for x in orq[w]]), 0, None
                for w in act_:
                    if w not in orcF:
                        continue
                    if a.oracle_plan in ("t", "tf", "ts"):  # 按步数：已执行的路点数 / 2 = 专家的第几个 WM 帧
                        orcJ[w] = min(nact[w] // 2, len(orcQ[w]) - 1)
                    else:  # q：按当前本体找最近的专家时刻（只往前、窗口 10 帧），再取其后 t+k
                        j0 = orcJ[w]
                        orcJ[w] = j0 + int(np.argmin(np.abs(orcQ[w][j0:j0 + 11] - np.asarray(prv[w], np.float32)[None]).sum(1)))
                if a.oracle_plan in ("q", "t"):
                    pol.orc = {w: orcF[w] for w in act_ if w in orcF}
                    pol.orc_d = {w: orcJ[w] for w in act_ if w in orcF}
                elif a.oracle_plan == "tf":
                    pol.orc_far = {w: (orcF[w], orcJ[w]) for w in act_ if w in orcF}
            if a.rtc and a.sim_delay:  # 推理期间照上一块执行的 d 个路点（已发出）作 IDM 的动作前缀；新回合的第一次决策没有前缀
                Pp, dv_ = np.zeros((len(act_), a.sim_delay, pol.net.ad), np.float32), np.zeros(len(act_), np.int64)
                for i, w in enumerate(act_):
                    if not fresh[w] and dq[w] is not None:
                        Pp[i, :len(dq[w])], dv_[i] = dq[w], len(dq[w])
                pol.rtc_prefix(Pp, dv_)
            acts = pol.act(h, ti, idx, fr.flatten(), wh, ph)
            if a.rtc and a.sim_delay:
                pol.rtc_prefix(None)
            if a.roll_k > 1 and getattr(pol, "plan_div", None) is not None:  # (rteasy) K 计划分歧度（新 episode 的第一次决策是同步采样，不记）
                dv_ = pol.plan_div.float().cpu().numpy()
                for i, w in enumerate(act_):
                    if fresh[w]:
                        pdv[w] = []
                    else:
                        pdv[w].append(round(float(dv_[i]), 4))
            if a.oracle_plan == "ts":  # (rteasy) 影子诊断：这次给 IDM 的计划（= WM 计划）各槽对齐到专家帧的均方差，与"拿当前帧冒充"的同口径参照
                ofs = torch.tensor([int(x) for x in pol.wa.tgt_offs.split(",")], device=dev)
                for i, w in enumerate(act_):
                    if w in orcF and not fresh[w]:
                        ex = orcF[w][torch.clamp(orcJ[w] + ofs, max=len(orcF[w]) - 1)].float()
                        pl = pol.plan[i, :len(ofs), :ex.shape[1]].float()
                        orcE.setdefault(w, []).append(((pl - ex) ** 2).mean((1, 2)).cpu().numpy())
                        orcC.setdefault(w, []).append(((h[i, -1, :ex.shape[1]].float()[None] - ex) ** 2).mean((1, 2)).cpu().numpy())
            if a.grip_idm:  # (prec) 同一计划（Policy.act 存的 pol.plan）、同一历史与本体 → 夹爪维
                with torch.autocast("cuda", torch.bfloat16):
                    og = gnet(torch.cat([h[:, pol.K - 1 - gna.ih:], pol.plan[:, :gna.H].to(h.dtype)], 1), zw=None, zp=ph)
                acts = np.array(acts, copy=True)
                acts[:, :, [6, 13]] = idm_.clip(idm_.decode(og.float(), gst), gst).cpu().numpy()[:, :, [6, 13]]
        chunk_ = np.array(acts, dtype=np.float32) if a.log_roll else None  # (flywheel) IDM 的原始动作块（时间集成之前的拷贝）
        tg += time.time() - g0
        if a.replan and pol.rp_last is not None:  # (hz2) 记录统计量；触发的 env 旧动作块作废（时间集成只留这次的新计划）
            sv_, va_, tr_ = (x.cpu().tolist() for x in pol.rp_last)
            for i, w in enumerate(act_):
                if fresh[w]:
                    rsv[w], rtg[w] = [], []
                rsv[w].append([round(x, 4) for x in sv_[i]] if va_[i] else None)
                if tr_[i]:
                    rtg[w].append(len(rsv[w]) - 1)
                    ens[w] = []
        if a.act_roll > 0:
            lo = pol.net.last_out
            assert len(lo) == len(act_), (lo.shape, len(act_))
            for i, w in enumerate(act_):
                prevn[w] = lo[i].float()
        if a.ensemble >= 0:  # ACT 式时间集成：每个路点取所有覆盖它的历次预测的加权平均，权重 exp(−m·i)，i = 0 是最早的那次
            acts = np.array(acts, copy=True)
            for i, w in enumerate(act_):
                if fresh[w]:
                    ens[w] = []
                ens[w] = [(c[a.exec:], g + 1) for c, g in ens[w] if len(c) > a.exec] + [(acts[i].copy() if a.sim_delay else acts[i], 0)]  # --sim_delay：存原块（下一行起会就地改写前 exec + d 位）
                L = min(a.exec + a.sim_delay, len(acts[i]))  # --sim_delay：要发出的是第 d..d+exec−1 位（新回合第 0..exec+d−1 位），都按绝对时刻集成
                num, den = np.zeros_like(acts[i, :L]), np.zeros((L, 1), np.float32)
                for c, g in ens[w]:  # g = 这次预测的"年龄"（决策数），最老的权重最大（同 ACT）
                    k = min(L, len(c))
                    wt = np.exp(-a.ensemble * (max(x for _, x in ens[w]) - g))
                    num[:k] += wt * c[:k]
                    den[:k] += wt
                rawi = acts[i].copy() if (a.ens_gwin or ctv) else None  # (prec) 最新块的原值（下一行会就地改写 acts[i]，而 ens[w][-1] 是它的视图）
                acts[i, :L] = num / den
                if a.ens_gwin or ctv:  # (prec) 只用最新若干块的集成（权重规则同上，年龄上限按所选的块重算）
                    def sub(win):
                        cs = (ens[w][:-1] + [(rawi, 0)])[-win:]
                        nu, de, mx = np.zeros_like(acts[i, :L]), np.zeros((L, 1), np.float32), max(x for _, x in cs)
                        for c, g in cs:
                            k = min(L, len(c))
                            wt = np.exp(-a.ensemble * (mx - g))
                            nu[:k] += wt * c[:k]
                            de[:k] += wt
                        return nu / de
                    if a.ens_gwin:
                        acts[i][:L, [6, 13]] = sub(a.ens_gwin)[:, [6, 13]]
                    if ctv:
                        q_ = np.asarray(prv[w], np.float32)
                        if fresh[w]:
                            cdn[w], cq[w], cnt_ct[w] = 0, None, [0, 0]
                        mv = cq[w] is not None and float(np.abs(q_[[6, 13]] - cq[w][[6, 13]]).max()) > 0.02
                        cdn[w] = int(ctv[3]) if mv else max(cdn[w] - 1, 0)
                        cq[w] = q_
                        ct_ = cdn[w] > 0 or float(np.abs(rawi[:int(ctv[2])][:, [6, 13]] - q_[[6, 13]]).max()) > ctv[1]
                        cnt_ct[w][0] += int(ct_)
                        cnt_ct[w][1] += 1
                        if ct_:
                            acts[i, :L] = sub(int(ctv[0]))
        for i, w in enumerate(act_):
            if a.sim_delay:  # 这一段执行 = 已发出的 d 个路点 + 新块第 d 位起；剩下 d 个留给下一次推理期间执行。acts[i, :exec] 改成实际发出的（录像照记）
                if fresh[w] or dq[w] is None:
                    dq[w] = acts[i, :0]
                cmb = np.concatenate([dq[w], acts[i, len(dq[w]):a.exec + a.sim_delay]])
                acts[i, :a.exec], dq[w] = cmb[:a.exec], cmb[a.exec:].copy()
            pipes[w].send(("step", (job[w][0], acts[i, :a.exec])))
            fresh[w] = False
            if a.ph2:
                fv[vidx[i]] = False
        for w in act_:
            try:
                r = recv(w) if pipes[w].poll(a.timeout) else None
            except (EOFError, OSError):
                r = None
            if r is None:  # 执行几个路点卡死 / worker 崩溃：换 worker，同一种子从头重做（不算策略失败）
                infra(w, "执行超时或 worker 崩溃")
                start(w)
                continue
            o, pv, done, ok, k, cnt, err = r[:7]
            if not done and len(r) > 7 and r[7]:
                pend[w] = r[7]  # 中途帧：下一次决策前按顺序补进历史
            nact[w] += k
            if lg[w] is not None:  # (flywheel) 发出的路点、实际执行了前几个、IDM 原始块
                i_ = act_.index(w)
                lg[w]["a"].append(np.array(acts[i_, :a.exec], dtype=np.float32))
                lg[w]["k"].append(k)
                lg[w]["c"].append(chunk_[i_])
            if vid[w] is not None:  # 这一步策略看到的头部帧、本体与给出的动作
                vid[w].append((obs[w][0] if isinstance(obs[w], tuple) else obs[w], prv[w], acts[act_.index(w), :a.exec]) + ((obs[w][1],) if eec else ()))
            if not done:
                obs[w], prv[w] = o, pv
                continue
            t, s = job[w]
            inflight[t] -= 1
            nep[w] += 1
            if err and nact[w] == 0:  # 第一个路点之前 / 之中就出错：不计，换下一个种子（同官方 rollout_steps == 0）
                skipped.append((t, s, "env_error " + err))
            else:
                valid[t] += 1
                eps.append(dict(task=t, cfg=a.cfg, seed=s, ep=eid[t], lang=lang[w], success=ok, steps=cnt, err=err,
                                time=time.time() - t0_[w], **({"poe": pol.pd[w].tolist()} if a.poe and hasattr(pol, "pd") else {}),
                                **({"surp": rsv[w], "trig": rtg[w]} if a.replan else {}),
                                **({"ct": round(cnt_ct[w][0] / max(cnt_ct[w][1], 1), 3)} if ctv else {}),
                                **({"orc": [int(orcJ.get(w, -1)), len(orcQ.get(w, ()))]} if a.oracle_plan else {}),
                                **({"orc_err": np.mean(orcE[w], 0).round(5).tolist(), "orc_cp": np.mean(orcC[w], 0).round(5).tolist(),
                                    "orc_nf": [[round(float(x[:8].mean()), 4), round(float(x[8:].mean()), 4)] for x in orcE[w]]}
                                   if a.oracle_plan == "ts" and orcE.get(w) else {}),
                                **({"pdiv": pdv[w]} if a.roll_k > 1 else {}),
                                **({"end": r[7]} if a.end_state and len(r) > 7 else {}),
                                **({"ov": ovs[w]} if a.ov else {})))  # (rteasy) --roll_k：逐决策 K 计划分歧度  # (rteasy) ts：各槽平均误差、复制参照、逐决策 [近槽, 远槽] 误差  # (rteasy) 结束时对齐到的专家帧 / 专家帧数
                eid[t] += 1
                if vid[w]:
                    vd = Path(str(a.out or "runs/rt/cl/x.json")[:-5] + "_video")
                    vd.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(vd / f"{t}_{s}.npz", rgb=np.stack([x[0] for x in vid[w]]), qpos=np.stack([x[1] for x in vid[w]]),
                                        **({"crop": np.stack([x[3] for x in vid[w]])} if eec else {}),  # (prec) 末端裁剪 (n,2,C,C,3)
                                        act=np.stack([np.asarray(x[2]) for x in vid[w]]), success=ok, lang=lang[w])
                    nvid[t] = nvid.get(t, 0) + 1
                vid[w] = None
                if lg[w] is not None:  # (flywheel) 先写临时名再改名（被杀时不留半截文件）
                    L_, p_ = lg[w], Path(a.log_roll) / a.cfg / t / f"{s}.npz"
                    p_.parent.mkdir(parents=True, exist_ok=True)
                    m_ = dict(task=t, cfg=a.cfg, seed=s, ep=eps[-1]["ep"], lang=lang[w], success=bool(ok), steps=cnt, err=err, step_lim=L_["lim"],
                              exec=a.exec, ensemble=a.ensemble, policy=a.policy, wm=a.wm, idm=a.idm, bc=a.bc, canon=a.canon, seed_base=a.seed, jpeg=a.jpeg)
                    np.savez(p_.with_suffix(".tmp.npz"), feat=np.stack(L_["f"]), qpos=np.stack(L_["q"]), act=np.stack(L_["a"]), nexec=np.array(L_["k"], np.int8),
                             chunk=np.stack(L_["c"]), meta=json.dumps(m_, ensure_ascii=False), **({"zc": np.stack(L_["z"])} if a.canon else {}))
                    os.replace(p_.with_suffix(".tmp.npz"), p_)
                lg[w] = None
                print(f"[{len(eps)}/{len(tasks) * a.n_ep}] {a.cfg} {t} seed {s} {'成功' if ok else '失败'} {cnt} 步 "
                      f"{eps[-1]['time']:.0f}s  {lang[w]}", flush=True)
            start(w)
    wall = time.time() - T0
    for pa in pipes:
        pa.send(("close", None))
    print(f"{a.policy}: {len(eps)} ep 用时 {wall:.0f}s（{n} 个 env）；推理 {tg:.0f}s；跳过的种子 {len(skipped)}；指令被截断 {ntrunc[0]} 条", flush=True)
    rate = summarize(eps)
    out = Path(a.out or f"runs/rt/cl/{a.policy.replace('+', '_')}_{a.cfg}_s{a.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(a), "episodes": eps, "skipped": skipped, "rate": rate, "wall": wall, "text_truncated": ntrunc[0]},
                              indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
