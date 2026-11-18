"""RoboTwin 2.0（aloha-agilex，Protocol B = 只用 clean demo 训练）的数据管线。要在隔离环境里跑（scripts/robotwin.sh；RoboTwin 源码 ~/rtwin/RoboTwin）。

  官方采集（种子从 0 起，专家成功的前 50 条；RoboTwin 自己的 collect_data.sh）：bash scripts/robotwin.sh <GPU> collect <task> demo_clean
  同一采集的并行版（长任务；接着已有的 seed.txt，结果同串行）：python -m probwm.render_robotwin collect --tasks put_bottles_dustbin --workers 10
  我们自己的程序化 DR 重渲染（同一批 demo：同种子、回放同一关节路径，只换外观，物理不变）：
      python -m probwm.render_robotwin dr --tasks ... --variants 1,2
  转成我们的格式：python -m probwm.render_robotwin convert --src demo_clean ours_dr1 ours_dr2 [--tasks ...]
  python -m probwm.render_robotwin --build-meta [--out data_robotwin]

对齐：RoboTwin 每 15 个仿真步（250 Hz）记一帧，16.7 Hz；HDF5 里 state[t] = 第 t 帧的 qpos（14 = 左臂 6 + 左夹爪 + 右臂 6 + 右夹爪，
同 take_action 的顺序），action[t] = state[t+1]（下一个路点；评测时 take_action 逐个执行，每个算一步）。
stride 2 → 8.3 Hz：我们的帧 k = 原始帧 2k，帧 k→k+1 之间执行 a_{2k}, a_{2k+1}（同 LIBERO 的 20 Hz 控制 / 10 Hz 帧，idm.STRIDE）。
图像：head / 腕部 320×240 → 256²（cv2 INTER_AREA 非等比拉伸，保留整个视野）；闭环 robotwin_eval 用同一个 to256。
DR（我们自己的，不用 RoboTwin 的纹理库 / 杂物模型 / 随机化代码）：桌面与墙面换 domrand.texture 的程序纹理（或纯色）、
灯光颜色强度与环境光、桌面上 1–8 个只有外观没有碰撞的几何体干扰物；每条 demo 一组参数（DR 目录的 dr/<k>.json）。只用全局 np.random 以外的
独立随机数，不改 setup_demo 的随机序列（物体布局与 clean 完全相同），所以回放官方关节路径必然复现同一条轨迹（check_success 复核）。
"""
import os

for _k, _v in [("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("HDF5_USE_FILE_LOCKING", "FALSE")]:
    os.environ.setdefault(_k, _v)

import argparse
import glob
import json
import multiprocessing as mp
import shutil
import sys
import time
import zlib

import numpy as np

RT = os.path.expanduser(os.environ.get("ROBOTWIN_ROOT", "~/rtwin/RoboTwin"))
SUITE, EMB = "robotwin", "aloha_agilex"
PILOT = ("stack_blocks_three", "put_bottles_dustbin", "handover_block", "lift_pot", "open_laptop")
CAMS = {"rgb": "cam_head", "wl": "cam_left_wrist", "wr": "cam_right_wrist"}  # npz 键 → HDF5 vision 组；特征库 feats/<encoder>[_<键>]
OBS = {"rgb": "head_camera", "wl": "left_camera", "wr": "right_camera"}  # npz 键 → get_obs()["observation"] 的相机名
QPOS = ("left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states")
VAL_FROM = 45  # 每任务 50 条：demo 45–49 作 val（同 LIBERO）；最终跑 --trainval


def to256(img, n=256):  # (h,w,3) uint8 → n×n；训练数据与闭环同一函数
    import cv2
    return cv2.resize(np.ascontiguousarray(img), (n, n), interpolation=cv2.INTER_AREA)


def src_dir(cfg, task):  # RoboTwin 采集输出：<RT>/data/<cfg>/<task>/aloha_agilex/{data/episode_*.hdf5, seed.txt, _traj_data/, instruction/}
    return os.path.join(RT, "data", cfg, task, EMB)


def rt_path():  # RoboTwin 代码用相对路径（./assets、./env_cfg）：进程内切到源码根目录
    os.chdir(RT)
    for p in (RT, os.path.join(RT, "description/utils")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _slot_call(fn, **kw):  # 在"在建"槽里调 fn；最多持槽 $RT_SLOT_MAX 秒（默认 120）：第一次建上下文若卡在驱动里，到时由后台线程放槽，不堵别的线
    import threading
    from . import envlock
    sl = envlock.slot().acquire()
    t = threading.Timer(float(os.environ.get("RT_SLOT_MAX", 120)), sl.release)
    t.daemon = True
    t.start()
    try:
        return fn(**kw)
    finally:
        t.cancel()
        sl.release()


def _setup(env, **kw):  # 建场景 / 渲染上下文：本机所有线共用的"在建"槽排队（probwm.envlock；几百个进程同时建会堵死驱动的 nvkms 锁）
    return _slot_call(env.setup_demo, **kw)


def _close(env, **kw):
    return _slot_call(env.close_env, **kw)


def rt_hold():  # RoboTwin 渲染进程（DR worker / 评测 env）在本机的总数上限：$RT_MAX_ENVS（默认 24），满了就等；父进程没了就退出
    from . import envlock

    def check():
        if os.getppid() == 1:
            raise SystemExit
    return envlock.hold("robotwin", int(os.environ.get("RT_MAX_ENVS", 24)), check)


def task_args(task, cfg, eval_mode=False):
    """同 RoboTwin collect_data.main / eval_policy_xpolicylab.load_task_args 的参数构造（需先 rt_path()）。"""
    import yaml
    from envs import CONFIGS_PATH
    with open(os.path.join(CONFIGS_PATH, f"{cfg}.yml"), encoding="utf-8") as f:
        a = yaml.safe_load(f)
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as f:
        emb = yaml.safe_load(f)
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as f:
        cam = yaml.safe_load(f)
    e = a["embodiment"]
    assert len(e) == 1, "只支持单一本体（aloha-agilex）"
    rf = emb[e[0]]["file_path"]
    with open(os.path.join(rf, "config.yml"), encoding="utf-8") as f:
        ec = yaml.safe_load(f)
    a.update(task_name=task, task_config=cfg, left_robot_file=rf, right_robot_file=rf, dual_arm_embodied=True, left_embodiment_config=ec,
             right_embodiment_config=ec, embodiment_name=str(e[0]), head_camera_h=cam[a["camera"]["head_camera_type"]]["h"],
             head_camera_w=cam[a["camera"]["head_camera_type"]]["w"], save_path=src_dir(cfg, task))
    if eval_mode:
        a.update(eval_mode=True, render_freq=0)
        a["data_type"].update(rgb=True, qpos=True)
    return a


# ---------------- 我们自己的程序化 DR ----------------

def _ent(x):  # RoboTwin 的 Actor 包装 → sapien Entity
    return getattr(x, "actor", x)


def _tex(seed, d):  # 程序纹理 PNG（domrand.texture，同 canon v2）缓存到 d/<seed>.png
    p = os.path.join(d, f"{seed}.png")
    if not os.path.exists(p):
        import imageio
        from .domrand import texture
        os.makedirs(d, exist_ok=True)
        imageio.imwrite(p + ".tmp.png", texture(np.random.default_rng(seed), 512, 512))
        os.replace(p + ".tmp.png", p)
    return p


def apply_dr(env, r, p=0.5, texdir=None):
    """在 setup_demo 之后、play_once 之前调用；只用独立随机数 r（不碰全局 np.random，物体布局与回放轨迹不变）→ 参数 dict。
    tex：桌面 / 墙面各自 70% 程序纹理、30% 纯色；light：平行光与点光源颜色（含近单色）、强度、环境光；
    dist：桌面上 1–8 个只有外观（无碰撞）的方块 / 球 / 圆柱 / 胶囊，避开任务物体（12 cm）与任务登记的 prohibited_area。
    三类各以概率 p 出现，至少一类。"""
    import sapien
    texdir = texdir or os.path.join(RT, "ours_tex")
    on = r.random(3) < p
    if not on.any():
        on[r.integers(3)] = True
    s = {}
    if on[0]:
        s["tex"] = {}
        for name, x in (("table", env.table), ("wall", env.wall)):
            if r.random() < 0.7:
                seed = int(r.integers(2 ** 31))
                s["tex"][name] = {"seed": seed}
                t = sapien.render.RenderTexture2D(_tex(seed, texdir))
            else:
                s["tex"][name] = {"rgb": r.random(3).round(3).tolist()}
            for sh in _ent(x).find_component_by_type(sapien.render.RenderBodyComponent).render_shapes:
                m = sh.get_material()
                if "seed" in s["tex"][name]:
                    m.set_base_color_texture(t)
                    m.base_color = [1, 1, 1, 1]
                else:
                    m.base_color = [*s["tex"][name]["rgb"], 1]
                m.roughness = float(r.uniform(0.2, 0.9))
    if on[1]:
        def col():
            c = r.random(3) if r.random() < 0.7 else np.eye(3)[r.integers(3)] * 0.8 + 0.2 * r.random(3)  # 30% 近单色
            return (c * r.uniform(0.3, 1.5)).round(3).tolist()
        s["light"] = {"dir": [col() for _ in env.direction_light_lst], "point": [col() for _ in env.point_light_lst],
                      "amb": (r.uniform(0.1, 0.7) * (0.7 + 0.3 * r.random(3))).round(3).tolist()}
        for L, c in zip(env.direction_light_lst, s["light"]["dir"]):
            L.set_color(c)
        for L, c in zip(env.point_light_lst, s["light"]["point"]):
            L.set_color(c)
        env.scene.set_ambient_light(s["light"]["amb"])
    if on[2]:
        bx, by, z0 = env.table_xy_bias[0], env.table_xy_bias[1], 0.74 + env.table_z_bias  # 桌面高度（create_table：顶面 = table_height）
        objs = [e.get_pose().p[:2] for e in env.scene.get_all_actors() if e.get_name() not in ("", "table", "wall", "ground")]
        areas = [a for a in (env.prohibited_area or []) if len(a) == 4]
        s["dist"] = []
        for _ in range(int(r.integers(1, 9))):
            for _try in range(30):
                x, y = bx + r.uniform(-0.55, 0.55), by + r.uniform(-0.3, 0.3)
                if all(np.hypot(x - o[0], y - o[1]) > 0.12 for o in objs) and \
                        not any(a[0] - 0.03 < x < a[2] + 0.03 and a[1] - 0.03 < y < a[3] + 0.03 for a in areas):
                    break
            else:
                continue
            kind, h = ["box", "sphere", "cylinder", "capsule"][r.integers(4)], float(r.uniform(0.015, 0.05))
            w = float(r.uniform(0.015, 0.05))
            d = {"kind": kind, "xy": [round(x, 3), round(y, 3)], "h": round(h, 3), "w": round(w, 3), "yaw": float(r.uniform(0, np.pi))}
            if r.random() < 0.3:
                d["tex"] = int(r.integers(2 ** 31))
                m = sapien.render.RenderMaterial()
                m.set_base_color_texture(sapien.render.RenderTexture2D(_tex(d["tex"], texdir)))
                m.base_color = [1, 1, 1, 1]
            else:
                d["rgb"] = r.random(3).round(3).tolist()
                m = sapien.render.RenderMaterial(base_color=[*d["rgb"], 1])
            b = env.scene.create_actor_builder()
            up = sapien.Pose(q=[0.7071068, 0, 0.7071068, 0])  # 圆柱 / 胶囊的轴（x）转成竖直
            if kind == "box":
                b.add_box_visual(half_size=[w, w * float(r.uniform(0.5, 1.5)), h], material=m)
            elif kind == "sphere":
                b.add_sphere_visual(radius=h, material=m)
            elif kind == "cylinder":
                b.add_cylinder_visual(pose=up, radius=w, half_length=h, material=m)
            else:
                b.add_capsule_visual(pose=up, radius=min(w, h), half_length=h, material=m)
            g = b.build_static(name=f"ours_dr_{len(s['dist'])}")  # 只有 RenderBody，没有碰撞形状：不影响物理
            g.set_pose(sapien.Pose([x, y, z0 + h + (min(w, h) if kind == "capsule" else 0)],
                                   [np.cos(d["yaw"] / 2), 0, 0, np.sin(d["yaw"] / 2)]))
            s["dist"].append(d)
            objs.append(np.array([x, y]))
    return s


# ---------------- 带超时的 worker 进程（SAPIEN 在 GPU 争用下偶发卡死在 do_poll，进程池里一个 worker 卡住整个池就卡住）----------------

_ENV = {}


def _env(task, cfg):  # 每个 worker 按（任务, 配置）缓存环境对象与参数
    if (task, cfg) not in _ENV:
        from scripts.collect_data import class_decorator
        _ENV[(task, cfg)] = (class_decorator(task), task_args(task, cfg))
    return _ENV[(task, cfg)]


def _wmain(conn):
    rt_path()
    keep = rt_hold()  # noqa: F841（持有到进程结束）
    while True:
        m = conn.recv()
        if m is None:
            break
        fn, arg = m
        try:
            conn.send(("ok", globals()[fn](*arg)))
        except Exception as e:
            conn.send(("err", f"{type(e).__name__}: {e}"))


class Workers:
    """n 个持久 worker（spawn），submit(key, 函数名, 参数) → poll() 得到 [(key, "ok" | "err" | "timeout", 结果)]。
    一次调用超过 timeout 秒就杀掉该进程、换一个新的（记为 timeout，由调用方决定重试或算失败）。gpus：轮流分给各 worker 的卡。"""

    def __init__(s, n, timeout, gpus=None, recycle=0):  # recycle：每个 worker 做完这么多次调用就换新进程（SAPIEN worker 显存随时间上涨）
        s.ctx, s.timeout, s.gpus, s.ws, s.recycle = mp.get_context("spawn"), timeout, gpus or [], [None] * n, recycle
        for i in range(n):
            s._start(i)

    def _start(s, i):
        env0 = os.environ.get("CUDA_VISIBLE_DEVICES")
        if s.gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(s.gpus[i % len(s.gpus)])
        a, b = s.ctx.Pipe()
        pr = s.ctx.Process(target=_wmain, args=(b,), daemon=True)
        pr.start()
        if s.gpus:
            os.environ.pop("CUDA_VISIBLE_DEVICES") if env0 is None else os.environ.__setitem__("CUDA_VISIBLE_DEVICES", env0)
        s.ws[i] = [pr, a, None, 0.0, 0]  # 进程、管道、当前 key、开始时间、已完成的调用数

    def free(s):
        return sum(w[2] is None for w in s.ws)

    def busy(s):
        return len(s.ws) - s.free()

    def submit(s, key, fn, arg):
        i = next(i for i, w in enumerate(s.ws) if w[2] is None)
        s.ws[i][1].send((fn, arg))
        s.ws[i][2], s.ws[i][3] = key, time.time()

    def poll(s, wait=2.0):
        from multiprocessing.connection import wait as mwait
        live = [w[1] for w in s.ws if w[2] is not None]
        if live:
            mwait(live, wait)
        out = []
        for i, w in enumerate(s.ws):
            if w[2] is None:
                continue
            if w[1].poll():
                try:
                    st, r = w[1].recv()
                except (EOFError, OSError):  # worker 崩溃（如 OOM）：记为 dead，换新进程
                    st, r = "dead", None
                out.append((w[2], st, r))
                w[2] = None
                w[4] += 1
                if st in ("dead", "err") or (s.recycle and w[4] >= s.recycle):  # 出过异常的 worker 状态可能坏了（如 OOM 后），也换新
                    try:
                        w[1].send(None)
                        w[0].join(30)
                    except Exception:
                        pass
                    if w[0].is_alive():
                        w[0].kill()
                        w[0].join(5)
                    s._start(i)
            elif time.time() - w[3] > s.timeout:
                out.append((w[2], "timeout", None))
                w[0].kill()
                w[0].join(5)
                s._start(i)
        return out

    def close(s):
        for w in s.ws:
            try:
                w[1].send(None)
            except Exception:
                pass
        for w in s.ws:
            w[0].join(10)
            if w[0].is_alive():
                w[0].kill()


# ---------------- 并行版官方采集（collect_data.py 两个阶段逐项照做，按种子序取前 n_ep 个成功的）----------------

def _cseed(task, cfg, seed):  # 第一阶段（同 collect_data.run 的种子搜索）：专家规划 + 成功判定 + 关节限位 → 成功时把关节路径存成 _traj_data/episodeseed<种子>.pkl
    from envs import UnStableError
    env, a = _env(task, cfg)
    try:
        _setup(env, now_ep_num=0, seed=seed, **dict(a, need_plan=True))
        env.play_once()
        ok = bool(env.plan_success and env.check_success())
        if ok:
            ok = bool(env.planned_joints_legal()[0])
            if ok:
                env.save_traj_data(f"seed{seed}")
        _close(env)
        return ok
    except (UnStableError, Exception):
        try:
            _close(env)
        except Exception:
            pass
        return False


def _creplay(task, cfg, k, seed):  # 第二阶段（同 collect_data.run 的采集）：回放关节路径、渲染存帧、写 instruction/ 与 HDF5
    from scripts.collect_data import build_episode_instructions
    env, a = _env(task, cfg)
    a = dict(a, need_plan=False, render_freq=0, save_data=True)
    _setup(env, now_ep_num=k, seed=seed, **a)
    tr = env.load_tran_data(k)
    env.set_path_lst(dict(a, left_joint_path=tr["left_joint_path"], right_joint_path=tr["right_joint_path"]))
    info = env.play_once()
    ed = build_episode_instructions(a, info, k, seed)
    _close(env, clear_cache=(k + 1) % a["clear_cache_freq"] == 0)
    env.merge_pkl_to_hdf5_video(instructions=ed["seen"])
    env.remove_data_cache()
    return info, bool(env.check_success())


def collect(task, cfg, n_ep, W, t0):
    """collect_data.sh 的并行版（长任务的种子搜索要几小时）：接着已有的 seed.txt（它之前的种子都已试过）按种子升序派发，
    在途数 ≤ 还缺的个数 → 得到的就是"从 0 起前 n_ep 个成功的种子"（同串行；超时的种子算失败）；再并行回放缺的 episode（超时重试 2 次）。
    产物与官方采集逐文件同格式。"""
    d = src_dir(cfg, task)
    os.makedirs(os.path.join(d, "_traj_data"), exist_ok=True)
    sp = os.path.join(d, "seed.txt")
    seeds = [int(x) for x in open(sp).read().split()] if os.path.exists(sp) else []
    nxt, ok = (max(seeds) + 1 if seeds else 0), {}
    while len(seeds) + sum(ok.values()) < n_ep or W.busy():
        while len(seeds) + sum(ok.values()) + W.busy() < n_ep and W.free():
            W.submit(nxt, "_cseed", (task, cfg, nxt))
            nxt += 1
        for s_, st, r in W.poll():
            ok[s_] = st == "ok" and bool(r)
            print(f"{task} 种子 {s_} {'成功' if ok[s_] else '失败' if st == 'ok' else st}（已有 {len(seeds) + sum(ok.values())} / {n_ep}，"
                  f"{time.time() - t0:.0f}s）", flush=True)
    for s_ in sorted(x for x, v in ok.items() if v)[:n_ep - len(seeds)]:  # 按种子序编号（同串行的 episode 号）
        os.replace(os.path.join(d, "_traj_data", f"episodeseed{s_}.pkl"), os.path.join(d, "_traj_data", f"episode{len(seeds)}.pkl"))
        seeds.append(s_)
    with open(sp, "w") as f:
        f.write("".join(f"{x} " for x in seeds))
    todo = [k for k in range(n_ep) if not os.path.exists(os.path.join(d, "data", f"episode_{k:07d}.hdf5"))]
    tries, ip = {}, os.path.join(d, "scene_info.json")
    info = json.load(open(ip)) if os.path.exists(ip) else {}
    while todo or W.busy():
        while todo and W.free():
            k = todo.pop(0)
            shutil.rmtree(os.path.join(d, ".cache", f"episode{k}"), ignore_errors=True)
            W.submit(k, "_creplay", (task, cfg, k, seeds[k]))
        for k, st, r in W.poll():
            if st == "ok":
                info[f"episode_{k}"] = r[0]
                with open(ip, "w") as f:
                    json.dump(info, f, ensure_ascii=False, indent=4)
            elif tries.get(k, 0) < 2:
                tries[k] = tries.get(k, 0) + 1
                todo.append(k)
            print(f"{task} episode {k} 回放 {('ok' if r[1] else '失败') if st == 'ok' else st}（{time.time() - t0:.0f}s）", flush=True)
    for f in glob.glob(os.path.join(d, "_traj_data", "episodeseed*.pkl")):
        os.remove(f)
    return task, len(seeds), sum(os.path.exists(os.path.join(d, "data", f"episode_{k:07d}.hdf5")) for k in range(n_ep)), time.time() - t0


def read_dr(ddir):  # DR 参数：{k: {"seed", "ok", tex / light / dist}}；每条一个 dr/<k>.json，兼容旧的单文件 dr.json
    d = json.load(open(os.path.join(ddir, "dr.json"))) if os.path.exists(os.path.join(ddir, "dr.json")) else {}
    for f in glob.glob(os.path.join(ddir, "dr", "*.json")):
        d[os.path.basename(f)[:-5]] = json.load(open(f))
    return d


def dr_one(task, src, v, k, p):
    """我们的 DR 重渲染一条：按官方 seed.txt / _traj_data 回放 clean demo k（collect_data 的第二阶段），setup 之后施加我们的 DR，
    输出到 <RT>/data/ours_dr<v>/<task>/aloha_agilex（同官方 HDF5 格式 + dr/<k>.json）。"""
    env, args = _env(task, src)
    sdir, ddir = src_dir(src, task), src_dir(f"ours_dr{v}", task)
    seed = [int(x) for x in open(os.path.join(sdir, "seed.txt")).read().split()][k]
    a = dict(args, need_plan=False, render_freq=0, save_data=True, save_path=sdir)
    _setup(env, now_ep_num=k, seed=seed, **a)
    tr = env.load_tran_data(k)
    env.set_path_lst(dict(need_plan=False, left_joint_path=tr["left_joint_path"], right_joint_path=tr["right_joint_path"]))
    env.save_dir = ddir  # 帧缓存与 HDF5 写到 DR 目录
    prm = apply_dr(env, np.random.default_rng([v, k, zlib.crc32(task.encode())]), p)
    env.play_once()
    ok = bool(env.check_success())
    _close(env, clear_cache=k % args["clear_cache_freq"] == 0)
    ins = json.load(open(os.path.join(sdir, "instruction", f"episode_{k:07d}.json"), encoding="utf-8"))
    env.merge_pkl_to_hdf5_video(instructions=ins["seen"])
    env.remove_data_cache()
    with open(os.path.join(ddir, "dr", f"{k}.json.tmp"), "w") as f:
        json.dump({"seed": seed, "ok": ok, **prm}, f)
    os.replace(os.path.join(ddir, "dr", f"{k}.json.tmp"), os.path.join(ddir, "dr", f"{k}.json"))
    return ok


def jpeg(x, mode="std"):  # 与采集管线相同的 JPEG（cv2 默认质量 95）：legacy = 官方数据（RGB 数组直接 imencode）；std = XPolicyLab（先转 BGR）
    import cv2
    if mode == "legacy":
        return cv2.imdecode(cv2.imencode(".jpg", x)[1], cv2.IMREAD_COLOR)
    return cv2.cvtColor(cv2.imdecode(cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR))[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def fast_path(out, task, d, keys):  # dr_fast 的输出：<out>/<task>/demo_<d>[_<键组>].npz
    return os.path.join(out, task, f"demo_{d}{'' if keys == ('rgb',) else '_' + '+'.join(keys)}.npz")


def dr_fast(task, src, v, k, p, keys, stride, img, out):
    """快速 DR 重渲染：回放与施加 DR 同 dr_one（同一随机数种子 → 同一组 DR 参数），但把 _take_picture 换掉——官方每个存帧点都用 get_obs
    给 head / front / 双腕 4 台相机光追（B200 上每台约 130 ms），这里只在第 0, stride, 2·stride… 个存帧点给 keys 里的相机拍照，
    过一遍与采集同款的 JPEG（std），每个存帧点都记 qpos。直接写 npz（fast_path）：keys 各 (T',img,img,3) uint8、qpos (T,14) 全部存帧点、
    meta（seed / ok / DR 参数 / seen 指令）。存帧点与 clean HDF5 的帧一一对应（同一回放、同一 save_freq）。"""
    env, args = _env(task, src)
    sdir = src_dir(src, task)
    seed = [int(x) for x in open(os.path.join(sdir, "seed.txt")).read().split()][k]
    a = dict(args, need_plan=False, render_freq=0, save_data=True, save_path=sdir)
    _setup(env, now_ep_num=k, seed=seed, **a)
    tr = env.load_tran_data(k)
    env.set_path_lst(dict(need_plan=False, left_joint_path=tr["left_joint_path"], right_joint_path=tr["right_joint_path"]))
    env.save_dir = os.path.join(out, ".scratch", task)  # 不往 clean 源目录写任何东西（save_data 只为让 play_once 调 _take_picture）
    prm = apply_dr(env, np.random.default_rng([v, k, zlib.crc32(task.encode())]), p)
    c = env.cameras
    cams = {"rgb": c.static_camera_list[c.static_camera_name.index("head_camera")], "wl": c.left_camera, "wr": c.right_camera}
    fr, qs, n = {x: [] for x in keys}, [], [0]

    def take():
        if n[0] % stride == 0:
            env._update_render()
            for x in keys:
                cams[x].take_picture()
            for x in keys:
                fr[x].append(to256(jpeg(np.ascontiguousarray((cams[x].get_picture("Color") * 255).clip(0, 255).astype(np.uint8)[:, :, :3])), img))
        qs.append(np.asarray(env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(), np.float32))
        n[0] += 1
    env._take_picture = take
    env.play_once()
    ok = bool(env.check_success())
    _close(env, clear_cache=k % args["clear_cache_freq"] == 0)
    ins = json.load(open(os.path.join(sdir, "instruction", f"episode_{k:07d}.json"), encoding="utf-8"))
    op = fast_path(out, task, 1000 * v + k, tuple(keys))
    os.makedirs(os.path.dirname(op), exist_ok=True)
    meta = {"task": task, "k": k, "v": v, "seed": seed, "ok": ok, "stride": stride, "n": n[0], "seen": ins["seen"], "unseen": ins.get("unseen", []), **prm}
    with open(op + ".tmp", "wb") as fh:
        np.savez(fh, qpos=np.stack(qs), meta=np.array(json.dumps(meta, ensure_ascii=False)), **{x: np.stack(y) for x, y in fr.items()})
    os.replace(op + ".tmp", op)
    return ok


def drfast_all(tasks, src, variants, n_ep, p, keys, stride, img, out, W, t0, reverse=False):
    """缺的（任务, 变体, k）逐条派发；已有的（含 dr_one 的 HDF5 版）跳过。派发前再查一次输出是否已存在：几台机器 / 几个进程可以
    从两头（--reverse）同时做同一份清单，不会重复。"""
    todo = []
    for v in variants:
        for t in tasks:
            done = read_dr(src_dir(f"ours_dr{v}", t))  # dr_one 已采的（HDF5 版，全部相机）不重做
            todo += [(t, v, k) for k in range(n_ep) if str(k) not in done and not os.path.exists(fast_path(out, t, 1000 * v + k, tuple(keys)))]
    todo = todo[::-1] if reverse else todo
    print(f"快速 DR 待做 {len(todo)} 条（相机 {keys}，stride {stride}{'，倒序' if reverse else ''}）", flush=True)
    tries = {}
    while todo or W.busy():
        while todo and W.free():
            j = todo.pop(0)
            if os.path.exists(fast_path(out, j[0], 1000 * j[1] + j[2], tuple(keys))) and not tries.get(j):
                continue  # 别的进程已做完
            W.submit(j, "dr_fast", (j[0], src, j[1], j[2], p, tuple(keys), stride, img, out))
        for (t, v, k), st, r in W.poll():
            if st != "ok" and tries.get((t, v, k), 0) < 2:
                tries[(t, v, k)] = tries.get((t, v, k), 0) + 1
                todo.insert(0, (t, v, k))
            print(f"{t} dr{v} ep {k} {('ok' if r else '回放失败') if st == 'ok' else st + ' ' + str(r or '')}（{time.time() - t0:.0f}s）", flush=True)


def dr_all(tasks, src, variants, n_ep, p, W, t0):  # 缺的（任务, 变体, k）逐条派发；超时（卡死）重试 2 次
    todo = []
    for v in variants:
        for t in tasks:
            sdir, ddir = src_dir(src, t), src_dir(f"ours_dr{v}", t)
            os.makedirs(os.path.join(ddir, "dr"), exist_ok=True)
            shutil.copy(os.path.join(sdir, "seed.txt"), ddir)
            done = read_dr(ddir)
            for k in range(n_ep):  # 有 HDF5 但没有参数文件的（写到一半被杀）重做
                if str(k) not in done:
                    for f in (os.path.join(ddir, "data", f"episode_{k:07d}.hdf5"),):
                        if os.path.exists(f):
                            os.remove(f)
                    shutil.rmtree(os.path.join(ddir, ".cache", f"episode{k}"), ignore_errors=True)
                    todo.append((t, v, k))
    print(f"DR 待做 {len(todo)} 条", flush=True)
    tries = {}
    while todo or W.busy():
        while todo and W.free():
            j = todo.pop(0)
            W.submit(j, "dr_one", (j[0], src, j[1], j[2], p))
        for (t, v, k), st, r in W.poll():
            if st != "ok" and tries.get((t, v, k), 0) < 2:
                tries[(t, v, k)] = tries.get((t, v, k), 0) + 1
                todo.append((t, v, k))
            print(f"{t} dr{v} ep {k} {('ok' if r else '回放失败') if st == 'ok' else st + ' ' + str(r or '')}（{time.time() - t0:.0f}s）", flush=True)


# ---------------- 官方 RoboTwin 2.0 数据集（协议 C：每任务 aloha-agilex clean 50 + randomized 500）----------------
# HF TianxingChen/RoboTwin2.0 dataset/<task>/aloha-agilex_{clean_50,randomized_500}.zip（2025 版采集，"legacy" HDF5）解开后：
#   <task>/data/episode<k>.hdf5：joint_action/vector (T,14)（第 t 帧的 qpos，同 take_action 顺序）、endpose/{left,right}_endpose (T,7)、
#   observation/{head,left,right,front}_camera/rgb (T,) = cv2.imencode 的 JPEG（编码的是 RGB 数组 → imdecode 原样即 RGB）；
#   <task>/instructions/episode<k>.json = {"seen": [...], "unseen": [...]}；<task>/seed.txt。
# 不落 npz：legacy 只建 meta / h5 / 指令表 + meta/src_h5.json（每条 demo 的 HDF5），encode.py --src_h5 直接从 HDF5 解帧编码。
LCAM = {"rgb": "head_camera", "wl": "left_camera", "wr": "right_camera", "front": "front_camera"}
OFF = {"clean": 0, "ours": 0, "rand": 5000}  # demo 号 = OFF[来源] + k（我们自己的 DR 变体是 1000·v + k，v ≤ 4）


def legacy_frames(p, key="rgb", stride=2, img=256):  # HDF5 → (ceil(T/stride), img, img, 3) uint8，与闭环同一个 to256；官方 legacy 与 XPolicyLab 两种格式
    import cv2
    import h5py
    if ".npz" in p:  # dr_fast 的输出：已按 stride 抽好、已是 img²（双腕在 demo_<d>_wl+wr.npz）；"…npz#n" = 只取前 n 帧（截到 clean 的长度）
        p, n = (p.split("#") + [""])[:2]
        q = p if key == "rgb" else p[:-4] + "_wl+wr.npz"
        with np.load(q) as z:
            return z[key][:int(n)] if n else z[key]
    with h5py.File(p, "r") as f:
        if "vision" in f:  # XPolicyLab v1.0（我们自己用官方脚本采的）：vision/cam_*/colors，只能用 data/decode_image_bit 解码
            if RT not in sys.path:
                sys.path.insert(0, RT)
            from data.decode_image_bit import decode_image_bit
            b = f["vision"][CAMS[key]]["colors"]
            return np.stack([to256(decode_image_bit(b[i]), img) for i in range(0, len(b), stride)])
        b = f["observation"][LCAM[key]]["rgb"]
        b = b[np.arange(0, len(b), stride)]
    return np.stack([to256(cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR), img) for x in b])


def _xpl_demo(p):  # XPolicyLab HDF5 的小数组：qpos (T,14)、action (T,14)（= 下一帧 qpos）、两臂末端位姿、seen 指令
    import h5py
    with h5py.File(p, "r") as f:
        q = np.concatenate([f["state"][x][()] for x in QPOS], 1).astype(np.float32)
        a = np.concatenate([f["action"][x][()] for x in QPOS], 1).astype(np.float32)
        ee = [f[f"state/{s}_ee_poses"][()].astype(np.float32) for s in ("left", "right")]
        nimg = len(f["vision/cam_head/colors"])
    assert nimg == len(q) and q.shape[1] == 14, (p, q.shape, nimg)
    return q, a, ee


def _legacy_task(job):  # 一个任务的全部来源：小数组 + 指令（不读图像）
    import h5py
    task, srcs, stride = job
    out = []
    for src, d in srcs:
        if src == "dr":  # 我们的 DR 重渲染（协议 B）：d = "<RoboTwin data 根>|<drfast 根>"；每个 (v, k) 先找 dr_one 的 HDF5 版，再找 dr_fast 的 npz 版
            rt, fd = d.split("|")
            for v in [int(x) for x in os.environ.get("RT_DR_VARIANTS", "1,2").split(",")]:  # rthard v3：RT_DR_VARIANTS=1,2,3（默认同第二版）
                for k in range(50):
                    dd = os.path.join(rt, f"ours_dr{v}", task, "aloha_agilex")
                    jp, hp = os.path.join(dd, "dr", f"{k}.json"), os.path.join(dd, "data", f"episode_{k:07d}.hdf5")
                    np_ = fast_path(fd, task, 1000 * v + k, ("rgb",))
                    if os.path.exists(jp) and os.path.exists(hp):
                        m = json.load(open(jp))
                        if not m["ok"]:
                            continue
                        q, a, ee = _xpl_demo(hp)
                        p = hp
                    elif os.path.exists(np_):
                        with np.load(np_) as z:
                            m, qq = json.loads(str(z["meta"])), z["qpos"]
                        if not m["ok"]:
                            continue
                        nc = next((len(e["a_full"]) for e in out if e["src"] != "dr" and e["k"] == k), None)  # clean 的存帧数
                        if nc is not None and len(qq) == nc + 1:  # drfast 比官方 HDF5 多记了回放结束后的最后一个存帧点：截掉
                            qq = qq[:nc]
                        q, a, ee = qq, np.concatenate([qq[1:], qq[-1:]]), None
                        p = f"{np_}#{len(np.arange(0, len(qq), stride))}"
                    else:
                        continue
                    ins = json.load(open(os.path.join(rt, "demo_clean", task, "aloha_agilex", "instruction", f"episode_{k:07d}.json")))
                    idx = np.arange(0, len(q), stride)
                    bits = sum(b for c, b in (("tex", 2), ("light", 4), ("dist", 16)) if c in m)  # DR 类别位掩码（同 views）
                    out.append(dict(task=task, src=src, k=k, demo=1000 * v + k, path=p, q=q[idx], a=a[idx], a_full=a, drbits=bits,
                                    ee=[e[idx] for e in ee] if ee else [np.zeros((len(idx), 7), np.float32)] * 2,
                                    seen=ins["seen"], unseen=ins.get("unseen", []), split=int(k >= 45)))
            continue
        xpl = src == "ours"  # 我们自己用官方采集脚本采的 clean（XPolicyLab 格式：<d>/<task>/aloha_agilex/data/episode_XXXXXXX.hdf5）
        n = len(glob.glob(os.path.join(d, task, *(["aloha_agilex"] if xpl else []), "data", "episode*.hdf5")))
        for k in range(n):
            if xpl:
                p = os.path.join(d, task, "aloha_agilex", "data", f"episode_{k:07d}.hdf5")
                q, a, ee = _xpl_demo(p)
                ins = json.load(open(os.path.join(d, task, "aloha_agilex", "instruction", f"episode_{k:07d}.json")))
            else:
                p = os.path.join(d, task, "data", f"episode{k}.hdf5")
                with h5py.File(p, "r") as f:
                    q = f["joint_action/vector"][()].astype(np.float32)
                    ee = [f[f"endpose/{s}_endpose"][()].astype(np.float32) for s in ("left", "right")]
                    nimg = len(f["observation/head_camera/rgb"])
                assert nimg == len(q) and q.shape[1] == 14, (task, src, k, q.shape, nimg)
                ins = json.load(open(os.path.join(d, task, "instructions", f"episode{k}.json")))
                a = np.concatenate([q[1:], q[-1:]])  # action[t] = 下一个路点的 qpos（末帧保持）
            idx = np.arange(0, len(q), stride)
            out.append(dict(task=task, src=src, k=k, demo=OFF[src] + k, path=p,
                            q=q[idx], a=a[idx], a_full=a, ee=[e[idx] for e in ee], seen=ins["seen"], unseen=ins.get("unseen", []),
                            split=int(k >= (450 if src == "rand" else 45))))
    return task, out


def legacy(out, srcs, tasks, stride=2, img=256, P=8, workers=32):
    """官方 legacy 数据 → <out>/{meta/(index.npz, tasks.json, actions.npy, success.npy, states/{qpos,left_ee,right_ee}.npy, langs.json,
    langs_aug.json, src_h5.json), h5/robotwin/<task>_demo.hdf5}。srcs：[(来源名 clean | rand, 解开后的根目录)]。帧序：任务名 → 来源（clean 先）→ k。
    split：clean k ≥ 45、rand k ≥ 450 为 val（最终 --trainval）。指令表：每条 demo 一行 = 它自己的前 P 条 seen 指令。"""
    import h5py
    mdir, sdir, hdir = os.path.join(out, "meta"), os.path.join(out, "meta", "states"), os.path.join(out, "h5", SUITE)
    for x in (mdir, sdir, hdir):
        os.makedirs(x, exist_ok=True)
    with mp.get_context("spawn").Pool(min(workers, len(tasks))) as pool:
        res = dict(pool.imap_unordered(_legacy_task, [(t, srcs, stride) for t in tasks]))
    tasks = sorted(tasks)
    bad = 0
    for t in tasks:  # DR 版与同 k 的 clean 帧数必须一致（canon 逐帧配对；回放偏离的去掉）
        cl = {e["k"]: len(e["q"]) for e in res[t] if e["src"] != "dr"}
        keep = [e for e in res[t] if e["src"] != "dr" or cl.get(e["k"]) == len(e["q"])]
        bad += len(res[t]) - len(keep)
        res[t] = keep
    if bad:
        print(f"去掉 {bad} 条与 clean 帧数不一致的 DR demo", flush=True)
    demos = [e for t in tasks for e in res[t]]
    ln = np.array([len(e["q"]) for e in demos], np.int64)
    st = np.concatenate([[0], np.cumsum(ln)[:-1]]).astype(np.int64)
    tj = [{"suite": SUITE, "task": t, "language": res[t][0]["seen"][0]} for t in tasks]
    tid = np.array([tasks.index(e["task"]) for e in demos], np.int32)
    np.savez(os.path.join(mdir, "index.npz"), demo_start=st, demo_len=ln, task_id=tid, demo=np.array([e["demo"] for e in demos], np.int32),
             split=np.array([e["split"] for e in demos], np.int8))
    json.dump(tj, open(os.path.join(mdir, "tasks.json"), "w"), ensure_ascii=False, indent=1)
    np.save(os.path.join(mdir, "actions.npy"), np.concatenate([e["a"] for e in demos]))
    np.save(os.path.join(mdir, "success.npy"), np.concatenate([np.arange(len(e["q"])) == len(e["q"]) - 1 for e in demos]))
    np.save(os.path.join(sdir, "qpos.npy"), np.concatenate([e["q"] for e in demos]))
    for i, s_ in enumerate(("left_ee", "right_ee")):
        np.save(os.path.join(sdir, f"{s_}.npy"), np.concatenate([e["ee"][i] for e in demos]))
    json.dump([e["seen"][0] for e in demos], open(os.path.join(mdir, "langs.json"), "w"), ensure_ascii=False, indent=0)
    json.dump({"rows": [e["seen"][:P] for e in demos], "demo_row": list(range(len(demos)))}, open(os.path.join(mdir, "langs_aug.json"), "w"),
              ensure_ascii=False)
    json.dump({"paths": [e["path"] for e in demos], "stride": stride, "img": img,
               "src": [[e["src"], e["k"]] for e in demos]}, open(os.path.join(mdir, "src_h5.json"), "w"))
    json.dump({e["path"]: e["unseen"] for e in demos}, open(os.path.join(mdir, "unseen.json"), "w"), ensure_ascii=False)
    np.save(os.path.join(mdir, "dr_bits.npy"), np.array([e.get("drbits", 0) for e in demos], np.int8))  # views() 的 canon 配对用
    for t in tasks:  # IDM 的 16.7 Hz 全部路点（idm.load_actions）
        with h5py.File(os.path.join(hdir, f"{t}_demo.hdf5"), "w") as h:
            for e in res[t]:
                h[f"data/demo_{e['demo']}/actions"] = e["a_full"]
    print(f"legacy → {out}：{len(tasks)} 个任务，{len(demos)} 条 demo（{dict((s_, sum(e['src'] == s_ for e in demos)) for s_, _ in srcs)}），"
          f"{int(ln.sum())} 帧", flush=True)


# ---------------- 转成我们的格式 ----------------

def convert_task(job):  # 一个任务的全部来源（同一个 h5 动作文件只由一个进程写）
    task, srcs, out, stride, img, keys = job
    rt_path()
    return [convert_one(task, c, int(c[len("ours_dr"):]) if c.startswith("ours_dr") else 0, out, stride, img, keys) for c in srcs]


def convert_one(task, cfg, v, out, stride, img, keys):
    """<RT>/data/<cfg>/<task>/aloha_agilex 的 HDF5 → <out>/render/robotwin/<task>/demo_<d:03d>.npz（d = 1000·v + k，clean 的 v = 0）
    + <out>/h5/robotwin/<task>_demo.hdf5 的 data/demo_<d>/actions（16.7 Hz 全部路点，idm.load_actions）。"""
    import h5py
    from data.decode_image_bit import decode_image_bit  # README：图像只能经它解码（两种字节格式）
    d0 = src_dir(cfg, task)
    seeds = [int(x) for x in open(os.path.join(d0, "seed.txt")).read().split()]
    dr = read_dr(d0) if v else {}
    bad = {int(k) for k, x in dr.items() if not x["ok"]}
    odir, hdir = os.path.join(out, "render", SUITE, task), os.path.join(out, "h5", SUITE)
    os.makedirs(odir, exist_ok=True), os.makedirs(hdir, exist_ok=True)
    acts, n, fr = {}, 0, 0
    for p in sorted(glob.glob(os.path.join(d0, "data", "episode_*.hdf5"))):
        k = int(os.path.basename(p)[8:15])
        dno, op = 1000 * v + k, os.path.join(odir, f"demo_{1000 * v + k:03d}.npz")
        if k in bad:
            continue
        with h5py.File(p, "r") as f:
            q = np.concatenate([f["state"][x][()] for x in QPOS], 1).astype(np.float32)
            acts[dno] = np.concatenate([f["action"][x][()] for x in QPOS], 1).astype(np.float32)
            if os.path.exists(op):
                continue
            idx = np.arange(0, len(q), stride)
            d = {key: np.stack([to256(decode_image_bit(f["vision"][CAMS[key]]["colors"][i]), img) for i in idx]) for key in keys}
            ins = json.loads(f["instructions"].asstr()[()])
            d.update(actions=acts[dno][idx], success=np.arange(len(idx)) == len(idx) - 1, s__qpos=q[idx],
                     s__left_ee=f["state/left_ee_poses"][()][idx].astype(np.float32),
                     s__right_ee=f["state/right_ee_poses"][()][idx].astype(np.float32))
        d["meta"] = np.array(json.dumps({"suite": SUITE, "task": task, "language": ins[0], "instructions": ins, "demo": dno,
                                         "stride": stride, "split": int(k >= VAL_FROM), "seed": seeds[k], "src": cfg, "dr": v,
                                         **({"dr_prm": dr[str(k)]} if v else {})}, ensure_ascii=False))
        with open(op + ".tmp", "wb") as fh:
            np.savez_compressed(fh, **d)
        os.replace(op + ".tmp", op)
        n, fr = n + 1, fr + len(idx)
    hp = os.path.join(hdir, f"{task}_demo.hdf5")
    with h5py.File(hp, "a") as h:  # 各来源（clean / DR 变体）追加到同一个文件
        for dno, x in acts.items():
            if f"data/demo_{dno}" in h:
                del h[f"data/demo_{dno}"]
            h[f"data/demo_{dno}/actions"] = x
    return task, cfg, n, fr


def aug_langs(out, P=16):
    """meta/langs_aug.json：每条 clean demo（任务, k）一行 = 该 episode 的前 P 条官方 seen 指令；DR 版本（demo 号 1000·v + k）指向同一行。
    → encode.py --text --per_demo_aug P --tag seen（逐 demo 增广表，训练时 --txt_aug 随机取一条）。"""
    ix, tasks = np.load(os.path.join(out, "meta/index.npz")), json.load(open(os.path.join(out, "meta/tasks.json")))
    rows, at, dr = [], {}, []
    for t, d in zip(ix["task_id"].tolist(), ix["demo"].tolist()):
        key = (tasks[t]["task"], d % 1000)
        if key not in at:
            with np.load(os.path.join(out, "render", SUITE, key[0], f"demo_{d:03d}.npz")) as z:
                at[key] = len(rows)
                rows.append(json.loads(str(z["meta"]))["instructions"][:P])
        dr.append(at[key])
    with open(os.path.join(out, "meta/langs_aug.json"), "w") as f:
        json.dump({"rows": rows, "demo_row": dr}, f, ensure_ascii=False)
    print(f"langs_aug：{len(rows)} 行 × ≤{P} 条，{len(dr)} 条 demo")


def wrist(out, enc):  # feats/<enc>_wl + feats/<enc>_wr（encode --key wl / wr --pool 2，各 (N,64,D)）→ feats/<enc>_wrist (N,128,D)：IDM --wrist 的腕部流
    fl, fr = (np.load(os.path.join(out, f"feats/{enc}_{k}/feats.npy"), mmap_mode="r") for k in ("wl", "wr"))
    od = os.path.join(out, f"feats/{enc}_wrist")
    os.makedirs(od, exist_ok=True)
    W = np.lib.format.open_memmap(os.path.join(od, "feats.npy"), "w+", fl.dtype, (len(fl), fl.shape[1] + fr.shape[1], fl.shape[2]))
    for i in range(0, len(fl), 8192):
        W[i:i + 8192] = np.concatenate([fl[i:i + 8192], fr[i:i + 8192]], 1)
    W.flush()
    info = json.load(open(os.path.join(out, f"feats/{enc}_wl/info.json")))
    json.dump({**info, "views": "wl+wr", "grid": None, "tokens": int(W.shape[1])}, open(os.path.join(od, "info.json"), "w"))
    print(f"{od}：{W.shape}")


def views(out):
    """data_robotwin 的两个视图根（feats / text / meta 的其余文件都软链，只重写 meta/index.npz；Feats / canon.Pairs 按 demo_start 从同一 feats.npy 取帧）：
    <out>_cv     canon.Pairs 的配对格式：aug = (demo ≥ 1000)，DR 版的 demo 号改回 k（与同任务 clean 第 k 条逐帧配对），cam_v = light_v = 0，
                 dr = 类别位掩码（2 纹理、4 灯光、16 干扰物）→ train --canon <out>_cv --canon_norm <WM ckpt>
    <out>_clean  只留 clean demo → 单帧先验 train --prior --data <out>_clean --norm_from <WM ckpt>（PoE 的 v_prior 是干净分布）"""
    ix = dict(np.load(os.path.join(out, "meta/index.npz")))
    bits = {"tex": 2, "light": 4, "dist": 16}
    drm = []
    if os.path.exists(os.path.join(out, "meta/dr_bits.npy")):  # legacy 建的根（协议 B）：位掩码建根时已存好
        drm = np.load(os.path.join(out, "meta/dr_bits.npy")).tolist()
    for t, d in zip(ix["task_id"].tolist(), ix["demo"].tolist()) if not drm else []:
        m = 0
        if d >= 1000:
            task = json.load(open(os.path.join(out, "meta/tasks.json")))[t]["task"]
            prm = read_dr(src_dir(f"ours_dr{d // 1000}", task))[str(d % 1000)]
            m = sum(b for k, b in bits.items() if k in prm)
        drm.append(m)
    for name, keep in (("_cv", np.ones(len(ix["demo"]), bool)), ("_clean", ix["demo"] < 1000)):
        v = out + name
        os.makedirs(os.path.join(v, "meta"), exist_ok=True)
        for x in ("feats", "text", "h5"):
            if not os.path.lexists(os.path.join(v, x)):
                os.symlink(os.path.join(out, x), os.path.join(v, x))
        for f in os.listdir(os.path.join(out, "meta")):
            if f != "index.npz" and not os.path.lexists(os.path.join(v, "meta", f)):
                os.symlink(os.path.join(out, "meta", f), os.path.join(v, "meta", f))
        z = {k: x[keep] for k, x in ix.items()}
        if name == "_cv":
            z.update(demo=(ix["demo"] % 1000).astype(ix["demo"].dtype), aug=(ix["demo"] >= 1000).astype(np.int8),
                     cam_v=np.zeros(len(keep), np.float32), light_v=np.zeros(len(keep), np.float32), dr=np.array(drm, np.int8))
        np.savez(os.path.join(v, "meta/index.npz"), **z)
        print(f"{v}：{int(keep.sum())} 条 demo")


def run(fn, jobs, workers):
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
        for j, r in enumerate(pool.imap_unordered(fn, jobs), 1):
            print(f"[{j}/{len(jobs)}] {r}，已用 {time.time() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description="RoboTwin 2.0：我们的 DR 重渲染 + 转成 probwm 格式 + 构建 meta")
    p.add_argument("cmd", nargs="?", default="", choices=["", "collect", "dr", "drfast", "convert", "wrist", "views", "legacy"])
    p.add_argument("--fast_out", default="runs/drfast", help="drfast：npz 输出根（<根>/<task>/demo_<1000·v+k>.npz）")
    p.add_argument("--reverse", action="store_true", help="drfast：清单倒着做（另一个进程正着做同一份清单时用）")
    p.add_argument("--clean_root", default="", help="legacy：官方 aloha-agilex_clean_50 解开后的根（<根>/<task>/data/…）")
    p.add_argument("--rand_root", default="", help="legacy：官方 aloha-agilex_randomized_500 解开后的根；空 = 只用 clean")
    p.add_argument("--ours_root", default="", help="legacy：改用我们自己按官方脚本采的 clean（XPolicyLab 格式，RoboTwin data/demo_clean），代替 --clean_root")
    p.add_argument("--dr_root", default="", help="legacy：加我们的 DR 重渲染（协议 B）：<RoboTwin data 根>|<drfast 根>")
    p.add_argument("--P", type=int, default=8, help="legacy：每条 demo 的指令表取前 P 条 seen 指令")
    p.add_argument("--out", default="data_robotwin")
    p.add_argument("--tasks", nargs="+", default=list(PILOT))
    p.add_argument("--src", nargs="+", default=["demo_clean"], help="convert：来源配置（demo_clean、ours_dr1 …）；dr：被回放的 clean 配置取第一个")
    p.add_argument("--variants", default="1,2", help="dr：DR 变体编号（随机数种子的一部分；demo 号 = 1000·v + k）")
    p.add_argument("--p", type=float, default=0.5, help="dr：每类（纹理 / 灯光 / 干扰物）出现的概率，至少一类")
    p.add_argument("--timeout", type=float, default=3600, help="collect / dr：一次种子尝试或回放超过这么多秒就杀掉 worker 重来（SAPIEN 偶发卡死）")
    p.add_argument("--n_ep", type=int, default=50)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--img", type=int, default=256)
    p.add_argument("--keys", default="rgb,wl,wr", help="convert：存哪些相机（rgb = head，wl / wr = 左 / 右腕）")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--recycle", type=int, default=0, help="collect / dr：每个 worker 做完这么多条就换新进程（显存上涨）")
    p.add_argument("--gpus", default="", help="collect / dr：worker 轮流用的卡（逗号分隔），空 = 继承 CUDA_VISIBLE_DEVICES")
    p.add_argument("--build-meta", action="store_true", help="build_meta + meta/langs_aug.json")
    p.add_argument("--encoder", default="dinov3_vits16", help="wrist：拼接哪个编码器的 _wl / _wr 特征库")
    a = p.parse_args()
    a.out = os.path.abspath(a.out)
    if a.build_meta:
        from .demo_index import build_meta
        build_meta(a.out, [SUITE])
        return aug_langs(a.out)
    if a.cmd == "wrist":
        return wrist(a.out, a.encoder)
    if a.cmd == "views":
        return views(a.out)
    if a.cmd == "legacy":
        srcs = ([("ours", a.ours_root)] if a.ours_root else [("clean", a.clean_root)]) + ([("rand", a.rand_root)] if a.rand_root else []) \
            + ([("dr", a.dr_root)] if a.dr_root else [])
        r0 = srcs[0][1]
        tasks = sorted(x for x in os.listdir(r0) if os.path.isdir(os.path.join(r0, x))) if a.tasks == ["all"] else a.tasks
        return legacy(a.out, srcs, tasks, a.stride, a.img, a.P, a.workers)
    if a.cmd in ("collect", "dr", "drfast"):  # 带超时的 worker（--timeout 秒，卡死的进程杀掉换新）
        rt_path()
        W, t0 = Workers(a.workers, a.timeout, [g for g in a.gpus.split(",") if g], a.recycle), time.time()
        try:
            if a.cmd == "collect":  # 并行版官方采集：--src 取第一个作配置（demo_clean）
                for t in a.tasks:
                    print(collect(t, a.src[0], a.n_ep, W, t0), flush=True)
            elif a.cmd == "drfast":
                drfast_all(a.tasks, a.src[0], [int(v) for v in a.variants.split(",")], a.n_ep, a.p, a.keys.split(","), a.stride, a.img,
                           a.fast_out, W, t0, a.reverse)
                print(f"DR_ALL_DONE {time.time() - t0:.0f}s", flush=True)
            else:
                dr_all(a.tasks, a.src[0], [int(v) for v in a.variants.split(",")], a.n_ep, a.p, W, t0)
                print(f"DR_ALL_DONE {time.time() - t0:.0f}s", flush=True)
        finally:
            W.close()
    elif a.cmd == "convert":
        run(convert_task, [(t, a.src, a.out, a.stride, a.img, tuple(a.keys.split(","))) for t in a.tasks], a.workers)


if __name__ == "__main__":
    main()
