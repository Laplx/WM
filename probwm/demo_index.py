"""Build the episode index for rendered demonstrations."""
import glob
import json
import os
import re
import resource

import numpy as np

SUITES = ["libero_90", "libero_10", "libero_goal", "libero_spatial", "libero_object"]

VAL_FROM = 45

def build_meta(out, suites):
    # 全局帧序：suite 固定顺序（不在 SUITES 里的如 robocasa 排在后面）→ task 名 → demo 编号
    tasks, demos = [], []
    for s in [x for x in SUITES if x in suites] + [x for x in suites if x not in SUITES]:
        sdir = os.path.join(out, "render", s)
        for t in sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []:
            fs = sorted((int(m[1]), os.path.join(sdir, t, n)) for n in os.listdir(os.path.join(sdir, t))
                        if (m := re.fullmatch(r"demo_(\d+)\.npz", n)))
            if fs:
                with np.load(fs[0][1]) as z:
                    tasks.append({"suite": s, "task": t, "language": json.loads(str(z["meta"]))["language"]})
                demos += [(len(tasks) - 1, k, p) for k, p in fs]
    if not demos:
        return print("data/render 下没有 demo")

    # 第一遍：只读小数组，得到每条长度和每个 key 的维度
    lens, dims, strides, metas = [], {}, set(), []
    for _, _, p in demos:
        with np.load(p) as z:
            lens.append(len(z["success"]))
            metas.append(json.loads(str(z["meta"])))
            strides.add(metas[-1]["stride"])
            for k in z.files:
                if k.startswith("s__"):
                    dims[k[3:]] = max(dims.get(k[3:], 0), z[k].shape[1])
    if len(strides) > 1:
        print(f"警告：混用了不同 stride {sorted(strides)}")
    N, lens = int(sum(lens)), np.array(lens, np.int64)
    start = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)

    mdir, sdir = os.path.join(out, "meta"), os.path.join(out, "meta", "states")
    os.makedirs(sdir, exist_ok=True)
    for f in glob.glob(os.path.join(sdir, "*.npy")):  # 清掉旧 key
        os.remove(f)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)  # 每个 memmap 占一个 fd
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(hard, 65536)), hard))
    except (ValueError, OSError):
        pass
    mm = {k: np.lib.format.open_memmap(os.path.join(sdir, f"{k}.npy"), "w+", np.float32, (N, d)) for k, d in dims.items()}
    for m in mm.values():
        m[:] = np.nan
    with np.load(demos[0][2]) as z:
        ad = z["actions"].shape[1]  # 动作维数：LIBERO / RoboCasa 7，RoboTwin 14（关节 qpos）
    acts, succ = np.zeros((N, ad), np.float32), np.zeros(N, bool)

    # 第二遍：逐条写入预分配的数组
    for (_, _, p), s, n in zip(demos, start, lens):
        with np.load(p) as z:
            acts[s:s + n], succ[s:s + n] = z["actions"], z["success"]
            for k in z.files:
                if k.startswith("s__"):
                    a = z[k]
                    mm[k[3:]][s:s + n, :a.shape[1]] = a
    for m in mm.values():
        m.flush()

    tid, dno = np.array([d[0] for d in demos], np.int32), np.array([d[1] for d in demos], np.int32)
    # 划分：npz 的 meta 自带 split 时用它（RoboCasa：官方 50_demos 之外的 demo 作 val），否则 demo 编号 ≥ VAL_FROM 为 val
    split = np.array([m["split"] if "split" in m else int(k >= VAL_FROM) for m, k in zip(metas, dno)], np.int8)
    np.save(os.path.join(mdir, "actions.npy"), acts)
    np.save(os.path.join(mdir, "success.npy"), succ)
    np.savez(os.path.join(mdir, "index.npz"), demo_start=start, demo_len=lens, task_id=tid, demo=dno, split=split)
    with open(os.path.join(mdir, "tasks.json"), "w") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=1)
    langs = [m["language"] for m in metas]
    if len(set(langs)) > len(tasks):  # 同一任务的指令逐 episode 不同（RoboCasa）：另存逐 demo 的指令（encode.py --text --per_demo）
        with open(os.path.join(mdir, "langs.json"), "w") as f:
            json.dump(langs, f, ensure_ascii=False, indent=0)
    print(f"meta：{len(tasks)} 个任务，{len(demos)} 条 demo，{N} 帧，{len(dims)} 个状态 key → {mdir}")
