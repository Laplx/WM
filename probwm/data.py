import json
import os
from pathlib import Path

import numpy as np
import torch


def load_parts(d, info, n, out=None):
    """分片特征库（encode --alloc --parts）→ CPU 张量 (n, T, D) fp16：本地有 part_XXX.npy 就读本地，否则用 ssh 从 info["remote"]（user@host:dir）
    流式读进内存（如 RoboCasa 300 条 MimicGen 的 185 GB 特征放在 9025，训练在 9022）。ssh 命令可用 PROBWM_SSH 覆盖。out：直接写进这块 (n, T, D) 张量（省一份内存）。"""
    import subprocess
    T, D = info["grid"][0] * info["grid"][1], info["dim"]
    F = torch.empty((n, T, D), dtype=torch.float16) if out is None else out
    buf = memoryview(F.numpy().reshape(-1).view(np.uint8))
    for i, (_, _, f0, f1) in enumerate(info["parts"]):
        p = Path(d) / f"part_{i:03d}.npy"
        if p.exists():
            m = np.load(p, mmap_mode="r")
            for j in range(0, len(m), 4096):
                F[f0 + j:f0 + min(j + 4096, len(m))] = torch.from_numpy(np.array(m[j:j + 4096]))
            continue
        host, path = ((info.get("remotes") or [None] * len(info["parts"]))[i] or info["remote"]).split(":", 1)  # remotes：逐片的位置（分散在几台机器上时）
        ssh = [os.path.expanduser(x) for x in os.environ.get("PROBWM_SSH", "ssh -o BatchMode=yes").split()]
        pr = subprocess.Popen(ssh + [host, f"cat {path}/part_{i:03d}.npy"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, bufsize=1 << 24)
        v = np.lib.format.read_magic(pr.stdout)
        shape, _, dt = (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(pr.stdout)
        assert shape == (f1 - f0, T, D) and dt == np.float16, (i, shape, dt)
        o, e = f0 * T * D * 2, f1 * T * D * 2
        while o < e:
            k = pr.stdout.readinto(buf[o:min(e, o + (1 << 26))])
            assert k, f"part {i} 的流提前结束"
            o += k
        assert pr.wait() == 0, f"part {i}：ssh 退出码 {pr.returncode}"
        print(f"  读入 {host}:{path}/part_{i:03d}.npy（{(e - f0 * T * D * 2) / 1e9:.1f} GB）", flush=True)
    return F


class MmapParts:
    """分片特征库按需读取（PROBWM_FEATS_MMAP=1，如 MG_1000 的 679 GB）：不整份读进进程内存，按下标用多线程 pread 从本地 part_XXX.npy 读行
    （连续帧合并成一次读；走页缓存，内存可回收）。只读本地分片（/data），返回未标准化的 fp16，标准化在 Feats.get 里做。
    9022 /data（4 × NVMe RAID0）冷读：一个 batch 64 × 5 帧（63 MB）约 13–18 ms。"""

    def __init__(s, d, info, n):
        from concurrent.futures import ThreadPoolExecutor
        T, D = info["grid"][0] * info["grid"][1], info["dim"]
        s.fd, s.off, s.f0 = [], [], []
        for i, (_, _, f0, f1) in enumerate(info["parts"]):
            p = Path(d) / f"part_{i:03d}.npy"
            m = np.load(p, mmap_mode="r")
            assert m.shape == (f1 - f0, T, D) and m.dtype == np.float16, (p, m.shape, m.dtype)
            s.fd.append(os.open(p, os.O_RDONLY)), s.off.append(m.offset), s.f0.append(f0)
        assert f1 == n, (f1, n)
        s.f0, s.rb, s.shape = np.array(s.f0), T * D * 2, (n, T, D)
        s.dtype, s.device, s.ex = torch.float16, torch.device("cpu"), ThreadPoolExecutor(16)

    def __len__(s):
        return s.shape[0]

    def __getitem__(s, ix):  # ix：任意形状的帧下标（CPU LongTensor）→ (*ix.shape, T, D) fp16；(ix, 切片…) 先取行再切
        if isinstance(ix, tuple):
            return s[ix[0]][(slice(None),) * torch.as_tensor(ix[0]).dim() + ix[1:]]
        return s.read(ix)

    def to_dev(s, ix, dev):  # 读进一块复用的锁页内存再同步拷上卡（Feats.get 用；拷完缓冲区即可复用）；缓冲按线程分开（train / idm --prefetch 的后台线程）
        import threading
        if torch.device(dev).type != "cuda":  # (pro) CPU 上用（如 scripts/rteasy_stub.py 在 CPU 上现算 norm）：锁页内存要 CUDA，直接读
            return s.read(ix)
        n, k = ix.numel() * s.rb, threading.get_ident()
        if not hasattr(s, "pins"):
            s.pins = {}
        if k not in s.pins or s.pins[k].numel() < n:
            s.pins[k] = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        return s.read(ix, s.pins[k][:n].numpy()).to(dev)

    def read(s, ix, buf=None):
        ix = torch.as_tensor(ix)
        fl = ix.reshape(-1).numpy()
        out = (np.empty(len(fl) * s.rb, np.uint8) if buf is None else buf).reshape(len(fl), s.rb)
        pt = np.searchsorted(s.f0, fl, "right") - 1
        a = np.flatnonzero(np.r_[True, (np.diff(fl) != 1) | (np.diff(pt) != 0)])  # 连续帧（同一分片）合并成一次读
        b = np.r_[a[1:], len(fl)]

        def rd(k):
            i, j, q = a[k], b[k], pt[a[k]]
            buf, o = out[i:j].reshape(-1), s.off[q] + int(fl[i] - s.f0[q]) * s.rb
            while len(buf):
                r = os.preadv(s.fd[q], [buf], o)
                assert r > 0, "读到文件末尾"
                buf, o = buf[r:], o + r
        list(s.ex.map(rd, range(len(a))))
        return torch.from_numpy(out.view(np.float16).reshape(*ix.shape, *s.shape[1:]))


def bal_sampler(data, fr, p_clean, suites=None, p_x=0.0, p_ao=0.0, p_dr=0.0):
    """train / idm --bal（四 suite 联合训练，fullbench 线）：候选帧 fr（GPU 下标）→ sample(B) 返回 B 个帧下标。
    每个 batch 各 suite 等量（B 须是 suite 数的倍数）、suite 内任务均匀；任务内干净帧（aug == 0）占 p_clean，
    其余按该 suite 里 DR（aug、demo 编号 < 1000）与生成 demo（编号 ≥ 1000）的候选帧量分；任务缺 DR / 生成帧时那一份并回干净。
    (xtask) p_x > 0（train --bal_x）：index 的 xtask 列 > 0 的 demo（同一场景配别的训练原句，task_id = 指令的任务）单独成第 4 组；
    有这组帧的任务里它占 p_x，干净 / DR / 生成三组按原比例乘 (1 − p_x)（三组之间的相对比例不变）；没有的任务不变。默认 0 = 原行为。
    (xtask3) p_ao > 0（train --bal_ao）：xtask 列 ≥ 500 的 demo（dscale 的 LIBERO-90 场景加物体配对，task_id 只作记录）不进上面的分组；
    抽完一个 batch 后每个样本以概率 p_ao 换成这些 demo 的候选帧（帧均匀）。默认 0 = 原行为。"""
    root, dev = Path(data.root), fr.device
    ix, tasks = np.load(root / "meta/index.npz"), json.loads((root / "meta/tasks.json").read_text())
    keep = np.isin([tasks[t]["suite"] for t in ix["task_id"]], suites) if suites else np.ones(len(ix["demo"]), bool)  # 同 Feats.load
    fao = None
    if p_ao > 0:  # (xtask3) 加物体帧单独成池，其余照旧
        isao = torch.from_numpy(ix["xtask"][keep] >= 500).repeat_interleave(data.length).to(dev)
        fao, fr = fr[isao[fr]], fr[~isao[fr]]
        print(f"bal ao：加物体候选帧 {len(fao)}，每个样本以 {p_ao} 的概率换成它们", flush=True)
    aug = ix["aug"][keep] > 0 if "aug" in ix.files else np.zeros(int(keep.sum()), bool)
    G = 4 if p_x > 0 else 3
    gn = np.where(~aug, 0, np.where(ix["demo"][keep] >= 1000, 2, 1))
    if p_x > 0:  # (xtask)
        gn = np.where(ix["xtask"][keep] > 0, 3, gn)
    g = torch.from_numpy(gn).repeat_interleave(data.length).to(dev)
    nt = len(data.task_suite)
    key = data.task_d.to(dev).repeat_interleave(data.length.to(dev))[fr] * G + g[fr]
    o = torch.argsort(key)
    pool, cnt = fr[o], torch.bincount(key, minlength=G * nt).view(nt, G).double().cpu()
    off = (torch.cumsum(cnt.flatten(), 0) - cnt.flatten()).long().to(dev)
    cg = cnt.flatten().to(dev)
    ss = sorted({data.task_suite[t] for t in range(nt) if cnt[t, 0] > 0})
    tab = [[t for t in range(nt) if data.task_suite[t] == s and cnt[t, 0] > 0] for s in ss]
    mix = torch.zeros(nt, G, dtype=torch.float64)
    for s, ts in zip(ss, tab):
        m = cnt[ts, 1:3].sum(0)
        w = torch.cat([torch.tensor([p_clean], dtype=torch.float64), (1 - p_clean) * m / m.sum()]) if m.sum() > 0 else torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        if p_dr > 0 and m[0] > 0:  # (pro 11-17 B3，train --bal_dr) DR 组在三组里固定占 p_dr，生成组拿剩下的（不够时为 0）；默认 0 = 按帧量分（原行为）
            w = torch.tensor([p_clean, p_dr, max(0.0, 1 - p_clean - p_dr)], dtype=torch.float64)
        w = torch.cat([w, torch.zeros(G - 3, dtype=torch.float64)])
        for t in ts:
            mix[t] = w * (cnt[t] > 0)
            mix[t, 0] += w[1:][cnt[t, 1:] == 0].sum()
            if G == 4 and cnt[t, 3] > 0:  # (xtask)
                mix[t, :3] *= 1 - p_x
                mix[t, 3] = p_x
    print("bal：" + "，".join(f"{s} {len(ts)} 个任务（候选帧 干净 / DR / 生成{' / xtask' if G == 4 else ''} {cnt[ts].sum(0).long().tolist()}，首任务比例 "
                              f"{[round(float(v), 3) for v in mix[ts[0]]]}）" for s, ts in zip(ss, tab)), flush=True)
    if G == 4:
        print("bal xtask：" + "，".join(f"{t}:{[round(float(v), 3) for v in mix[t]]}" for t in range(nt) if cnt[t, 3] > 0), flush=True)
    mix, nts = mix.float().to(dev), torch.tensor([len(x) for x in tab], device=dev)
    tab = torch.tensor([x + [x[0]] * (max(map(len, tab)) - len(x)) for x in tab], device=dev)

    def sample(B):
        assert B % len(ss) == 0, f"--bal：batch {B} 不是 suite 数 {len(ss)} 的倍数"
        si = torch.arange(len(ss), device=dev).repeat_interleave(B // len(ss))
        t = tab[si, (torch.rand(B, device=dev) * nts[si]).long()]
        k = t * G + torch.multinomial(mix[t], 1).squeeze(1)
        out = pool[off[k] + (torch.rand(B, device=dev, dtype=torch.float64) * cg[k]).long()]
        if fao is not None:  # (xtask3)
            m = torch.rand(B, device=dev) < p_ao
            out = torch.where(m, fao[(torch.rand(B, device=dev, dtype=torch.float64) * len(fao)).long()], out)
        return out
    return sample


class CatViews:
    """几个同帧序的分片库（MmapParts）按 token 拼接，按需读取（IDM / BC 的额外相机流，如 RoboCasa 的右视角 + 腕部）。
    to_dev 返回标准化后的 fp16：mu / sd 为 (T, D)（各视角各自的逐通道统计量按 token 展开）。"""

    def __init__(s, stores, mu=None, sd=None):
        s.st, s.mu, s.sd = stores, mu, sd
        s.shape = (len(stores[0]), sum(x.shape[1] for x in stores), stores[0].shape[2])
        s.dtype, s.device = torch.float16, torch.device("cpu")

    def __len__(s):
        return s.shape[0]

    def raw(s, ix, dev):  # 未标准化
        return torch.cat([x.to_dev(ix, dev) for x in s.st], -2)

    def to_dev(s, ix, dev):  # mu 为 None（(pro) Feats 的多视角分片库，标准化在 Feats.get 里做）时返回未标准化的
        return s.raw(ix, dev) if s.mu is None else ((s.raw(ix, dev).float() - s.mu) / s.sd).to(torch.float16)

    def __getitem__(s, ix):  # (pro) 同 MmapParts：CPU 下标 → 未标准化 fp16（Feats 算 norm 时用）
        if isinstance(ix, tuple):
            return s[ix[0]][(slice(None),) * torch.as_tensor(ix[0]).dim() + ix[1:]]
        return torch.cat([x[ix] for x in s.st], -2)


class CatChan:
    """(rteasy 11-16) 几个同帧序、同 token 网格的分片库（MmapParts）按通道拼接，按需读取（如 DINOv3-S 384 + V-JEPA 2.1 末层 1024 → 1408）。
    特征库 info.json 写 {"cat": [子目录名, …]}；Feats 当作 MmapParts 用：返回未标准化 fp16，标准化在 Feats.get 里做。"""

    def __init__(s, stores):
        assert all(x.shape[:2] == stores[0].shape[:2] for x in stores), [x.shape for x in stores]
        s.st, s.shape = stores, (stores[0].shape[0], stores[0].shape[1], sum(x.shape[2] for x in stores))
        s.dtype, s.device = torch.float16, torch.device("cpu")

    def __len__(s):
        return s.shape[0]

    def to_dev(s, ix, dev):
        return torch.cat([x.to_dev(ix, dev) for x in s.st], -1)

    def __getitem__(s, ix):  # CPU 下标 → 未标准化 fp16（Feats 算 norm 时用）
        if isinstance(ix, tuple):
            return s[ix[0]][(slice(None),) * torch.as_tensor(ix[0]).dim() + ix[1:]]
        return torch.cat([x[ix] for x in s.st], -1)


class Feats:
    """特征整份常驻 GPU（已标准化，fp16）。batch 只做 GPU 下标操作，不用 DataLoader。"""

    def __init__(s, F, start, length, task, split, temb, tmask, K, delta, device,
                 states=None, src=None, root=None, task_names=None, task_suite=None, norm=None, views=1):
        s.K, s.delta, s.device = K, delta, device
        s.start, s.length = torch.as_tensor(start), torch.as_tensor(length)
        s.split_d, s.task_d = torch.as_tensor(split), torch.as_tensor(task).long()
        s.ftask = s.task_d.repeat_interleave(s.length).to(device)  # 每帧取语言嵌入的行号（逐 episode 指令表时见 load）
        s.fend = (s.start + s.length).repeat_interleave(s.length).to(device)  # 每帧所在 demo 的结束下标（开区间）
        s.fstart = s.start.repeat_interleave(s.length).to(device)
        s.fsplit = fsplit = s.split_d.repeat_interleave(s.length).to(device)
        s.temb, s.tmask = temb.to(device), tmask.to(device)
        s.task_names, s.task_suite, s._states, s.src, s.root = task_names, task_suite, states or {}, src, root
        # 逐通道标准化（统计量只用训练帧；评测时用 ckpt 里存的 norm）
        tr = torch.nonzero(fsplit == 0).squeeze(1)
        g = torch.Generator(device=device).manual_seed(0)  # 固定抽样：不同 run / rank 的 norm 一致，探针可以通用
        s.mm = isinstance(F, (MmapParts, CatViews, CatChan))  # 按需读取：不原地标准化，Feats.get 取出后再标准化（CatViews：(pro) 多视角分片库）
        if norm is None:  # 多视角（views > 1，token 按视角拼接）：每个视角各自的逐通道统计量 → mu / sd 为 (T, D)
            idx, Tv = tr[torch.randperm(len(tr), device=device, generator=g)[:20000]], F.shape[1] // views
            if isinstance(F, CatViews):  # (pro) 多视角分片库：各视角 token 数可以不同（agentview 256 + 腕部 64）；正式训练用 pro_wristassemble 写的 norm.pt（--norm_from）
                o = np.cumsum([0] + [x.shape[1] for x in F.st])
                ms = [(x.mean(0), x.std(0).clamp_min(1e-6)) for x in (F[idx.cpu(), o[v]:o[v + 1]].to(device).float().flatten(0, 1) for v in range(views))]
                s.mu, s.sd = (torch.cat([q.expand(int(o[v + 1] - o[v]), -1) for v, q in enumerate(m)]) for m in zip(*ms))
            else:
                ms = [(x.mean(0), x.std(0).clamp_min(1e-6)) for x in (F[idx.to(F.device), v * Tv:(v + 1) * Tv].to(device).float().flatten(0, 1) for v in range(views))]
                s.mu, s.sd = ms[0] if views == 1 else (torch.stack(m).repeat_interleave(Tv, 0) for m in zip(*ms))
            del ms
        else:
            s.mu, s.sd = norm["mu"].to(device), norm["sd"].to(device)
        for i in range(0, 0 if s.mm else len(F), 4096):  # 小块原地标准化，避免大块临时张量被缓存分配器长期占着
            F[i:i + 4096] = ((F[i:i + 4096].to(device).float() - s.mu) / s.sd).to(F.dtype).to(F.device)
        s.f = F
        s.offs = torch.arange(-(K - 1), 1, device=device) * delta  # 历史帧相对 anchor 的偏移
        fi = torch.arange(len(F), device=device)
        ok = (fi - (K - 1) * delta >= s.start.to(device).repeat_interleave(s.length.to(device))) & (fi + delta < s.fend)
        s.anchors = {sp: torch.nonzero(ok & (fsplit == sp)).squeeze(1) for sp in (0, 1)}
        a = s.anchors[0][torch.randperm(len(s.anchors[0]), device=device, generator=g)[:20000]]
        s.rs = ((s.get(a + delta) - s.get(a)).float().flatten(0, 1).std(0).clamp_min(1e-6)
                if norm is None else norm["rs"].to(device))  # 残差尺度（标准化空间）
        if device.type == "cuda":
            torch.cuda.empty_cache()

    @property
    def norm(s):
        return {"mu": s.mu.cpu(), "sd": s.sd.cpu(), "rs": s.rs.cpu()}

    def anchors_h(s, split, H):
        a = s.anchors[split]
        return a[a + H * s.delta < s.fend[a]]

    def batch(s, B, split=0, idx=None, n_fut=1, off=1, stride=1):  # 目标 z_{t+off+stride·j}，j < n_fut（off>1：远点 / 关键帧 WM，如 off 4 stride 4 → t+4, t+8）
        if idx is None:
            a = s.anchors[split] if n_fut == 1 and off == 1 else s.anchors_h(split, off + stride * (n_fut - 1))
            idx = a[torch.randint(len(a), (B,), device=s.device)]
        tgt = s.get(idx[:, None] + s.delta * (off + stride * torch.arange(n_fut, device=s.device))).flatten(1, 2)  # (B, n_fut*T, D)
        return s.get(idx[:, None] + s.offs), tgt, s.ftask[idx], idx

    def get(s, ix):  # 按帧下标取特征；特征放在 CPU 内存（PROBWM_FEATS_CPU=1，三视角等大特征库训练 WM）时在 CPU 上取再拷上卡
        if s.mm:  # PROBWM_FEATS_MMAP=1：从本地分片按需读，再标准化
            return ((s.f.to_dev(ix.cpu(), s.device).float() - s.mu) / s.sd).to(torch.float16)
        return s.f[ix] if s.f.device == ix.device else s.f[ix.cpu()].to(s.device, non_blocking=True)

    def text(s, task, rand=False, p_orig=0.0):
        if s.temb.dim() == 3:
            return s.temb[task], s.tmask[task]
        i = torch.randint(s.temb.shape[1], task.shape, device=task.device) if rand else torch.zeros_like(task)
        if rand and p_orig > 0:
            i = torch.where(torch.rand(task.shape, device=task.device) < p_orig, torch.zeros_like(i), i)
        return s.temb[task, i], s.tmask[task, i]   # 增广表 (n_task, P, L, D)：第 0 条是原指令

    def state(s, key):
        if key not in s._states:
            p = Path(s.root) / "meta/states" / f"{key}.npy"
            arr = np.load(p if p.exists() else Path(s.root) / f"meta/{key}.npy", mmap_mode="r")
            s._states[key] = torch.from_numpy(np.ascontiguousarray(arr[s.src])).to(s.device).float().reshape(len(s.src), -1)
        return s._states[key]

    def state_keys(s):
        if s.root is None:
            return list(s._states)
        return sorted(p.stem for p in (Path(s.root) / "meta/states").glob("*.npy")) + ["success"]

    @classmethod
    def load(cls, root, encoder, text, K, delta, device, suites=None, norm=None):
        root = Path(root)
        tasks = json.loads((root / "meta/tasks.json").read_text())
        ix = np.load(root / "meta/index.npz")
        st, ln, tid, sp = ix["demo_start"], ix["demo_len"], ix["task_id"], ix["split"]
        keep = np.ones(len(st), bool) if not suites else np.isin([tasks[t]["suite"] for t in tid], suites)
        st, ln, tid, sp = st[keep], ln[keep], tid[keep], sp[keep]
        info = root / f"feats/{encoder}/info.json"  # info 带 views（如 RoboCasa 的 agentview + 腕部）：各视角特征库按 token 拼接，WM / IDM 的 T = 各视角之和
        inf = json.loads(info.read_text()) if info.exists() else {}
        vs = inf.get("views", [encoder])
        vinf = [json.loads(q.read_text()) if (q := root / f"feats/{v}/info.json").exists() else {} for v in vs] if len(vs) > 1 else []
        if "cat" in inf:  # (rteasy) 通道拼接的特征库：各子库须是本地分片库，只支持按需读
            assert keep.all() and os.environ.get("PROBWM_FEATS_MMAP"), "通道拼接的特征库只支持 PROBWM_FEATS_MMAP=1、不筛 suite"
            F = CatChan([MmapParts(root / f"feats/{c}", json.loads((root / f"feats/{c}/info.json").read_text()), int(ln.sum())) for c in inf["cat"]])
        elif vinf and all("parts" in x for x in vinf):  # (pro) 多视角分片库（如 agentview + 腕部 8×8，scripts/pro_wristassemble.py）：各视角分片按 token 拼接，只支持按需读
            assert keep.all() and os.environ.get("PROBWM_FEATS_MMAP"), "多视角分片库只支持 PROBWM_FEATS_MMAP=1、不筛 suite"
            assert all(x["parts"] == vinf[0]["parts"] and x["dim"] == vinf[0]["dim"] for x in vinf), "各视角的分片边界 / 维数须相同"
            F = CatViews([MmapParts(root / f"feats/{v}", x, int(ln.sum())) for v, x in zip(vs, vinf)])
        elif "parts" in inf:  # 分片特征库（可在别的机器上）：整份读进 CPU 内存；只支持不筛 suite 的单视角库
            assert keep.all() and len(vs) == 1, "分片特征库不支持 --suites 筛选 / 多视角"
            if os.environ.get("PROBWM_FEATS_MMAP"):  # 按需读取（只读本地分片）：不整份进内存
                F = MmapParts(root / f"feats/{encoder}", inf, int(ln.sum()))
            else:
                F = load_parts(root / f"feats/{encoder}", inf, int(ln.sum()))
                F = F if os.environ.get("PROBWM_FEATS_CPU") else F.to(device)
        else:
            fs = [np.load(root / f"feats/{v}/feats.npy", mmap_mode="r") for v in vs]
            F = torch.empty((int(ln.sum()), sum(f.shape[1] for f in fs), fs[0].shape[2]), dtype=torch.float16,
                            device="cpu" if os.environ.get("PROBWM_FEATS_CPU") else device)  # CPU：只有 Feats.batch 取数，idm.zin 等直接下标的代码不支持
            o = 0
            for a, l in zip(st, ln):  # 逐 demo 拷贝上 GPU，避免占满主机内存
                F[o:o + l] = torch.cat([torch.from_numpy(np.array(f[a:a + l])) for f in fs], 1).to(F.device)
                o += l
        t = torch.load(root / f"text/{text}.pt")
        start = np.concatenate([[0], np.cumsum(ln)[:-1]])
        src = np.concatenate([np.arange(a, a + l) for a, l in zip(st, ln)])
        d = cls(F, start, ln, tid, sp, t["emb"], t["mask"], K, delta, device, src=src, root=root,
                task_names=[x["task"] for x in tasks], task_suite=[x["suite"] for x in tasks], norm=norm, views=len(vs))
        if "frame_row" in t:  # 逐帧文本行号（probwm/prim.py：每帧的当前原语），按全局帧号取
            d.ftask = t["frame_row"][torch.from_numpy(src)].long().to(device)
        elif "demo_row" in t:  # 逐 episode 指令表（encode.py --text --per_demo，如 RoboCasa）：ftask 改为每帧的文本行号，task_d 仍是任务号
            d.ftask = t["demo_row"][torch.from_numpy(keep)].long().repeat_interleave(d.length).to(device)
        return d

    @classmethod
    def synthetic(cls, K, delta, device, n_demo=400, L=40, T=16, D=32, seed=0, norm=None):
        """合成分叉数据：点匀速右移，到中点后按任务（=语言）向上或向下分叉。"""
        g = torch.Generator().manual_seed(seed)
        task = torch.arange(n_demo) % 2
        t = torch.arange(L).float()
        x = (t / L * 2 - 1).expand(n_demo, L)
        y = ((t - L / 2).clamp_min(0) / L * 2)[None] * (task[:, None] * 2 - 1).float()
        p = torch.stack([x, y + 0.02 * torch.randn(n_demo, L, generator=g)], -1)
        h = torch.cat([p, torch.sin(3 * p), torch.cos(3 * p)], -1)
        W = torch.randn(T, h.shape[-1], D, generator=g) / h.shape[-1] ** 0.5
        F = torch.tanh(torch.einsum("nlc,tcd->nltd", h, W)).reshape(n_demo * L, T, D).to(device)
        split = (torch.arange(n_demo) >= int(0.9 * n_demo)).to(torch.int8)
        temb = torch.randn(2, 4, 16, generator=g)
        return cls(F, torch.arange(n_demo) * L, torch.full((n_demo,), L), task, split, temb,
                   torch.ones(2, 4, dtype=torch.bool), K, delta, device,
                   states={"pos": p.reshape(-1, 2).to(device)}, task_names=["up", "down"],
                   task_suite=["syn", "syn"], norm=norm)
