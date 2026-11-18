"""测试时特征规范化：把受扰动（相机 / 光照）观测的特征 z_obs 映射成同一仿真状态在规范相机、规范光照下的特征 z_can，
再交给只见过干净单视角的 WM 和 IDM（RCAN 的思路，James et al. 2019，那里在像素空间；这里在冻结 DINOv3 特征空间、用 flow matching）。
模型就是 WM（K=1：z_obs 作为唯一的历史帧进 cross-attn 上下文；n_fut=1；语言为空），目标 = z_can（train --canon）：
  (a) 条件 FM：从纯噪声出发积分到 z_can（--p_warm 0）
  (b) 桥：从加噪的 z_obs 出发，x_τ0 = (1-τ0)ε + τ0·z_obs，积分到 τ=1 → z_can（--p_warm 1，τ0~U(.05,warm_hi)，模型以 τ0 为条件；
      即"加噪回 τ0 再去噪"的跨域版本，SDEdit 式翻译，但用配对数据训练）
  (r) 残差：--mode res --sigma σ，目标 = (z_can − z_obs)/rs（rs = 训练配对残差的逐通道尺度），从 σ·噪声出发
  --mode det 为确定性回归对照。
配对数据：camaug.py 生成的数据根（clean ∪ aug；aug demo 与同 task / demo 编号的 clean demo 逐帧是同一仿真状态）。
特征按 --canon_norm 那个 WM ckpt 的 norm 标准化，规范化后直接交给该 WM / IDM。

  python -m probwm.train --canon data_camaug,data_lightaug --K 1 --n_fut 1 --alpha 3 --p_warm 1 --warm_hi 0.7 --out runs/canon/b
  python -m probwm.canon --sets data_camaug,data_lightaug --models a:runs/canon/a/ckpt.pt:4:0,b:runs/canon/b/ckpt.pt:4:0.5
      离线（val demo）：对规范真值的误差（标准化空间 MSE）+ IDM 动作 nmse（历史和下一帧都规范化）；
      对照 = 不处理（raw）、单帧无条件先验净化（closed_loop --purify 同一函数）
  python -m probwm.closed_loop ... --canon runs/canon/b/ckpt.pt --canon_steps 4 --canon_tau0 0.5
"""
import argparse
import json
import os
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import torch

from . import fm
from .train import build


class _Parts:
    """(cam) 分片特征库（feats/<enc>/part_XXX.npy + info.json 的 parts = [demo 起, demo 止, 帧起, 帧止]）的只读拼接视图：只支持不跨分片的帧切片（一条 demo 不跨分片）。"""

    def __init__(s, d):
        info = json.loads((d / "info.json").read_text())
        s.parts = [(int(p[2]), int(p[3]), np.load(d / f"part_{k:03d}.npy", mmap_mode="r")) for k, p in enumerate(info["parts"])]
        s.shape = (s.parts[-1][1], *s.parts[0][2].shape[1:])

    def __getitem__(s, sl):
        a, b = sl.start, sl.stop
        for g0, g1, m in s.parts:
            if g0 <= a < g1:
                assert b <= g1, ("帧切片跨了分片", a, b, g0, g1)
                return m[a - g0:b - g0]
        raise IndexError(sl)


class Pairs:
    """若干个 camaug 数据根的配对（扰动帧 → 同一状态的干净帧），特征常驻 GPU（fp16，按给定 norm 标准化）。
    各根的干净一半都拷贝自同一份原特征，只载入一次（再次遇到时核对首尾帧逐位一致）；--aug_only 的根没有干净一半，
    要排在完整的根之后。
    p_id：batch 里换成恒等对（输入就是干净帧）的比例，让无扰动时近似恒等映射。
    text（WM 的文本表路径，--canon_wm 用）：按任务名对齐到这些根的 tasks.json，text() 与 data.Feats.text 相同。"""

    def __init__(s, roots, norm, dev, encoder="dinov3_vits16", splits=(0, 1), p_id=0.2, text=None):
        s.device, s.p_id, s.mu, s.sd, s.rs = dev, p_id, norm["mu"].to(dev), norm["sd"].to(dev), norm["rs"].to(dev)
        chunks, s.clean, o, tasks0 = [], {}, 0, None
        lazy = bool(os.environ.get("PROBWM_PAIRS_LAZY"))  # =1：先只记 memmap 切片，拷进 F 时再逐 demo 读（峰值内存 ≈ F 本身，而不是 2×；结果逐位相同）
        load = (lambda v: v) if lazy else np.array
        cols = {k: [] for k in ("io", "ic", "d0", "d1", "sp", "cv", "lv", "dr", "set", "tid")}
        for k, r in enumerate(map(Path, roots)):
            ix, tasks = np.load(r / "meta/index.npz"), json.loads((r / "meta/tasks.json").read_text())
            assert tasks0 is None or tasks == tasks0, f"{r} 的 tasks.json 与 {roots[0]} 不同"
            fp_ = r / f"feats/{encoder}/feats.npy"
            tasks0, f = tasks, (np.load(fp_, mmap_mode="r") if fp_.exists() else _Parts(r / f"feats/{encoder}"))  # (cam) 分片库（info.json 的 parts）
            st, ln, aug = ix["demo_start"], ix["demo_len"], ix["aug"]
            lv = ix["light_v"] if "light_v" in ix.files else np.zeros(len(aug), np.float32)
            dr = ix["dr"] if "dr" in ix.files else np.zeros(len(aug), np.int8)  # 领域随机化的类别位掩码（domrand.py）
            cv = ix["cam_v"] if "cam_v" in ix.files else np.zeros(len(aug), np.float32)  # vidmix merge 的根没有 cam_v
            ref = {(int(t), int(d)): j for j, (t, d) in enumerate(zip(ix["task_id"], ix["demo"])) if not aug[j]}
            for j in np.nonzero(aug)[0]:
                key, l, sp = (int(ix["task_id"][j]), int(ix["demo"][j])), int(ln[j]), int(ix["split"][j])
                if sp not in splits:
                    continue
                c = ref.get(key)
                assert c is not None or key in s.clean, f"{r} 没有干净 demo {key}：--aug_only 的根要排在完整的根之后"
                assert c is None or (ln[c] == l and ix["split"][c] == sp), (r, key)
                x = None if c is None else f[st[c]:st[c] + l]
                if key not in s.clean:
                    s.clean[key] = (o, l, sp, len(chunks))
                    chunks.append(load(x))
                    o += l
                elif c is not None:
                    ch = chunks[s.clean[key][3]]
                    assert np.array_equal(x[[0, l - 1]], ch[[0, l - 1]]), f"{r} 的干净 demo {key} 与先前的根不一致"
                c0 = s.clean[key][0]
                chunks.append(load(f[st[j]:st[j] + l]))
                for n_, v in (("io", np.arange(o, o + l)), ("ic", np.arange(c0, c0 + l)), ("d0", np.full(l, o)),
                              ("d1", np.full(l, o + l)), ("sp", np.full(l, sp)), ("cv", np.full(l, cv[j])),
                              ("lv", np.full(l, lv[j])), ("dr", np.full(l, dr[j])), ("set", np.full(l, k)), ("tid", np.full(l, key[0]))):
                    cols[n_].append(v)
                o += l
        fdev = torch.device("cpu") if os.environ.get("PROBWM_FEATS_CPU") else dev  # PROBWM_FEATS_CPU=1：特征放主机内存（四 suite 干净 + DR 约 170 GB），batch 时再拷上卡
        F = torch.empty((o, *chunks[0].shape[1:]), dtype=torch.float16, device=fdev)
        p = 0
        for k, ch in enumerate(chunks):
            F[p:p + len(ch)] = torch.from_numpy(np.array(ch) if lazy else ch).to(fdev)
            p += len(ch)
            chunks[k] = None
        for i in range(0, len(F), 4096):
            F[i:i + 4096] = ((F[i:i + 4096].to(dev).float() - s.mu) / s.sd).to(F.dtype).to(fdev)
        s.f, s.tasks = F, tasks0
        for k, v in cols.items():
            setattr(s, k, torch.from_numpy(np.concatenate(v)).to(dev))
        s.anchors = {sp: torch.nonzero(s.sp == sp).squeeze(1) for sp in (0, 1)}  # 扰动帧在 io / ic 里的位置
        if len(s.anchors[0]):  # 配对残差 z_can − z_obs 的逐通道尺度（--mode res 的目标按它归一，存进 ckpt 的 norm["rs"]）
            j = s.anchors[0][torch.randperm(len(s.anchors[0]), device=dev, generator=torch.Generator(device=dev).manual_seed(0))[:2000]]
            s.rs = (s.get(s.ic[j]).float() - s.get(s.io[j]).float()).flatten(0, 1).std(0).clamp_min(1e-6)
        s.temb = torch.zeros(1, 1, 1, device=dev)  # 不用语言：一个 null token
        if text:
            t = torch.load(text)
            names = [x["task"] for x in json.loads((Path(text).parent.parent / "meta/tasks.json").read_text())]
            sel = torch.tensor([names.index(x["task"]) for x in tasks0])
            s.temb, s.tmask = t["emb"][sel].to(dev), t["mask"][sel].to(dev)
        s._seq = {}
        torch.cuda.empty_cache()

    def text(s, task, rand=False, p_orig=0.0):  # 同 data.Feats.text
        if s.temb.dim() == 3:
            return s.temb[task], s.tmask[task]
        i = torch.randint(s.temb.shape[1], task.shape, device=task.device) if rand else torch.zeros_like(task)
        if rand and p_orig > 0:
            i = torch.where(torch.rand(task.shape, device=task.device) < p_orig, torch.zeros_like(i), i)
        return s.temb[task, i], s.tmask[task, i]

    def seq_batch(s, B, split, K, n, off):
        """时序版 → (上下文 (B,K,T,D)，目标 (B,n·T,D)，任务)：上下文 = 同一条扰动 demo 的 y_{t−K+1..t}（以 p_id 的概率换成
        同一状态的干净帧），目标 = 干净帧 z_{t+off..t+off+n−1}。off=0、n=1：时序规范化器；off=1、n=8：规范目标 WM。"""
        if (split, K, n, off) not in s._seq:
            a = s.anchors[split]
            s._seq[(split, K, n, off)] = a[(s.io[a] - (K - 1) >= s.d0[a]) & (s.io[a] + off + n - 1 < s.d1[a])]
        a = s._seq[(split, K, n, off)]
        i = a[torch.randint(len(a), (B,), device=s.device)]
        o, c = s.io[i], s.ic[i]
        h = torch.where(torch.rand(B, device=s.device)[:, None] < s.p_id, c[:, None], o[:, None]) + torch.arange(-K + 1, 1, device=s.device)
        return s.get(h), s.get(c[:, None] + torch.arange(off, off + n, device=s.device)).flatten(1, 2), s.tid[i]

    @property
    def norm(s):
        return {"mu": s.mu.cpu(), "sd": s.sd.cpu(), "rs": s.rs.cpu()}

    def batch(s, B, split=0):  # → (z_obs, z_can)，各 (B,T,D)
        a = s.anchors[split]
        i = a[torch.randint(len(a), (B,), device=s.device)]
        o, c = s.io[i], s.ic[i]
        return s.get(torch.where(torch.rand(B, device=s.device) < s.p_id, c, o)), s.get(c)

    def get(s, ix):  # 帧下标（GPU）→ 特征；特征在主机内存（PROBWM_FEATS_CPU=1）时经锁页缓冲拷上卡
        if s.f.device == ix.device:
            return s.f[ix]
        import threading
        r, k = ix.reshape(-1).cpu(), threading.get_ident()  # 缓冲按线程分开（train --prefetch 的后台线程）
        if not hasattr(s, "_bufs"):
            s._bufs = {}
        if k not in s._bufs or s._bufs[k].shape[0] != len(r):
            s._bufs[k] = torch.empty((len(r), *s.f.shape[1:]), dtype=s.f.dtype).pin_memory()
        torch.index_select(s.f, 0, r, out=s._bufs[k])
        return s._bufs[k].to(ix.device).view(*ix.shape, *s.f.shape[1:])


def load(path, dev):
    ck = torch.load(path, map_location="cpu")
    ma = Namespace(**ck["args"])
    ma.rs = ck["norm"]["rs"].to(dev)  # --mode res 的残差尺度
    ma.affine = [x.to(dev) for x in ck["affine"]] if "affine" in ck else None  # canon_tta.py --params affine 的输入端仿射
    m = build(ma, ck["dims"]).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, ma, ck["norm"]


@torch.no_grad()
def canonize(m, ma, z, steps=4, tau0=0.0):
    """z (B,T,D) 或 (B,K,T,D)（时序规范化器：最近 K 帧扰动观测）标准化特征 → 规范化后的 (B,T,D) float。tau0>0：从加噪回 τ0 的 z 出发（桥；模型带 τ0 嵌入时以 τ0 为条件）；0：从纯噪声出发；
    --mode res 的模型：从 σ·噪声出发积分出归一化的残差 Δ/rs，返回 z + rs·Δ。"""
    B, dev = len(z), z.device
    if getattr(ma, "affine", None) is not None:
        z = z.float() * (1 + ma.affine[0]) + ma.affine[1]
    with torch.autocast("cuda", torch.bfloat16):
        ctx = z[:, -ma.K:] if z.dim() == 4 else z[:, None]
        kv = m.context(ctx, torch.zeros(B, 1, 1, device=dev), torch.zeros(B, 1, dtype=torch.bool, device=dev))
        return fm.sample(m, kv, ctx[:, -1].float(), ma.mode, steps=steps, alpha=ma.alpha, tau0=tau0, sigma=ma.sigma, rs=ma.rs)[0].float()


def null_len(pa):
    """prior（train --prior）训练时拿完整文本表、mask 全 0 → L 个 null token；model.context 不做 attention mask，
    null 的个数会改变注意力权重，所以推理也要给同样 L 个（11-06 前推理只给 1 个，训推不一致）。"""
    t = torch.load(Path(pa.data) / f"text/{pa.text}.pt", map_location="cpu")["mask"]
    return t.shape[-1]


def purifier(path, spec, dev):  # 单帧无条件先验净化，与 closed_loop.Policy.purify 相同
    pk = torch.load(path, map_location="cpu")
    pa = Namespace(**pk["args"])
    pm = build(pa, pk["dims"]).to(dev).eval()
    pm.load_state_dict(pk["model"])
    t0, K, st, snr = (float(x) for x in spec.split(","))
    L = null_len(pa)

    def f(z):
        B = len(z)
        with torch.autocast("cuda", torch.bfloat16):
            kv = pm.context(torch.zeros(B, pa.K, *z.shape[1:], device=dev), torch.zeros(B, L, pk["dims"][2], device=dev),
                            torch.zeros(B, L, dtype=torch.bool, device=dev))
            return fm.sample(pm, kv, z.float(), "abs", steps=int(st), alpha=pa.alpha, tau0=t0, n_lan=int(K), snr=snr)[0].float()
    return f, pk["norm"]


@torch.no_grad()
def evaluate(a):
    from .idm import STRIDE, make, metrics
    dev = torch.device("cuda")
    torch.manual_seed(0)
    norm = torch.load(a.norm, map_location="cpu")["norm"]
    sets = a.sets.split(",")
    P = Pairs(sets, norm, dev, splits=(1,))
    Z = P.f
    idm, ia, st = make(torch.load(a.idm, map_location="cpu"), dev)
    n_act, offs = 2 * ia.H, torch.arange(-ia.ih, ia.H + 1, device=dev)
    A = torch.zeros(len(Z), n_act, 7, device=dev)  # 只填干净帧；扰动帧的动作 = 对应干净帧的
    for (t, d), (o, l, _, _) in P.clean.items():
        with h5py.File(Path(a.h5) / P.tasks[t]["suite"] / f"{P.tasks[t]['task']}_demo.hdf5", "r") as f:
            act = f[f"data/demo_{d}/actions"][()]
        A[o:o + l] = torch.from_numpy(act[np.minimum(STRIDE * np.arange(l)[:, None] + np.arange(n_act), len(act) - 1)]
                                      .astype(np.float32)).to(dev)
    cl = torch.cat([torch.arange(o, o + l, device=dev) for o, l, _, _ in P.clean.values()])
    cd0 = torch.cat([torch.full((l,), o, device=dev) for o, l, _, _ in P.clean.values()])
    cd1 = cd0 + torch.cat([torch.full((l,), l, device=dev) for o, l, _, _ in P.clean.values()])
    fst = torch.zeros(len(Z), dtype=torch.long, device=dev)  # 每帧所在 demo 的起点（时序规范化器取最近 K 帧用）
    fst[cl], fst[P.io] = cd0, P.d0
    if a.geo:  # 几何对齐（geowarp.py）：每条 demo 用第一帧对同任务训练 demo 的规范第一帧估一个网格仿射，整条 demo 先重采样
        from . import geowarp
        ftk = torch.zeros(len(Z), dtype=torch.long, device=dev)
        ftk[cl] = torch.cat([torch.full((l,), t, device=dev) for (t, _), (o, l, _, _) in P.clean.items()])
        ftk[P.io] = P.tid
        wa_ = Namespace(**torch.load(a.norm, map_location="cpu")["args"])
        ix = np.load(Path(wa_.data) / "meta/index.npz")
        nm = [x["task"] for x in json.loads((Path(wa_.data) / "meta/tasks.json").read_text())]
        ff = np.load(Path(wa_.data) / f"feats/{wa_.encoder}/feats.npy", mmap_mode="r")
        rows = [r for r in range(len(ix["split"])) if ix["split"][r] == 0 and nm[ix["task_id"][r]] in [t["task"] for t in P.tasks]]
        anc = ((torch.from_numpy(np.stack([ff[ix["demo_start"][r]] for r in rows])).to(dev).float() - P.mu) / P.sd)
        atk = torch.tensor([[t["task"] for t in P.tasks].index(nm[ix["task_id"][r]]) for r in rows], device=dev)
        Zg = Z.clone()
        for d in torch.unique(fst).tolist():
            idx = torch.nonzero(fst == d).squeeze(1)
            th = geowarp.estimate(Z[d], anc[atk == ftk[d]])
            Zg[idx] = geowarp.warp(Z[idx], th[None].expand(len(idx), -1, -1))[0]
        print("几何对齐：", len(torch.unique(fst)), "条 demo", flush=True)
    # 分档：相机 (0,.01] (.01,.02] (.02,.03]；只有光照的根按光照 (0,.15] (.15,.3] (.3,.5]；相机 + 光照的根按相机幅度；
    # 领域随机化的根按类别：只有一类（相机 / 贴图 / 光照 / 图像）各一档，两类及以上一档
    rows = {"clean": None}
    for k, r in enumerate(sets):
        m = P.set == k
        if (P.dr[m] > 0).any():
            for b, c in enumerate(("相机", "贴图", "光照", "图像", "物体")):
                rows[f"{Path(r).name}|只有{c}"] = m & (P.dr == 1 << b)
            rows[f"{Path(r).name}|组合"] = m & (P.dr > 0) & ((P.dr & (P.dr - 1)) > 0)
            rows[f"{Path(r).name}|全部"] = m
            continue
        cam = bool((P.cv[m] > 0).any())
        v, ed = (P.cv, (0, .01, .02, .03)) if cam else (P.lv, (0, .15, .3, .5))
        for lo, hi in zip(ed[:-1], ed[1:]):
            rows[f"{Path(r).name}|{'cam' if cam else 'light'} ({lo:g},{hi:g}]"] = m & (v > lo) & (v <= hi)
        rows[f"{Path(r).name}|全部"] = m
    ok = P.io + ia.H < P.d1  # IDM 需要下一帧在 demo 内
    cok = cl + ia.H < cd1
    methods = {"truth": None, "raw": lambda z: z.float()}  # truth：扰动帧换成规范真值（IDM 的参照）
    for spec in filter(None, a.purify.split(";")):
        f, pn = purifier(a.prior, spec, dev)
        print(f"先验 norm 与 WM norm 的最大差 {max((pn[k] - norm[k]).abs().max().item() for k in ('mu', 'sd')):.2e}")
        methods[f"prior {spec}"] = f
    if a.filters:  # WM 条件后验净化（probwm/postfilt.py）：逐 demo 递归，整份 Zm 一次算出
        from . import postfilt
        wk = torch.load(a.norm, map_location="cpu")
        wa = Namespace(**wk["args"])
        wm = build(wa, wk["dims"]).to(dev).eval()
        wm.load_state_dict(wk["model"])
        tx = torch.load(Path(wa.data) / f"text/{wa.text}.pt")
        te, tmk = (tx[k][:, 0] if tx[k].dim() == (4 if k == "emb" else 3) else tx[k] for k in ("emb", "mask"))
        names = [x["task"] for x in json.loads((Path(wa.data) / "meta/tasks.json").read_text())]
        sel = torch.tensor([names.index(t["task"]) for t in P.tasks])
        te, tmk = te[sel].to(dev), tmk[sel].to(dev)
        pk = torch.load(a.prior, map_location="cpu")
        pa = Namespace(**pk["args"])
        pm = build(pa, pk["dims"]).to(dev).eval()
        pm.load_state_dict(pk["model"])
        for spec in a.filters.split(";"):
            name, v = spec.split(":")
            flt = postfilt.Filter(wm, wa, pm, pa, pk["dims"][2], v)
            methods[name] = ("seq", lambda flt=flt: postfilt.run(P, flt, te, tmk, wa.K))
    for spec in filter(None, a.models.split(",")):
        name, path, steps, tau0 = spec.split(":")
        m, ma, mn = load(path, dev)
        assert all(torch.equal(mn[k], norm[k]) for k in ("mu", "sd")), f"{path} 的 norm 与 --norm 不一致"
        if a.geo and ma.K == 1:  # 先几何对齐再规范化
            methods[name + "+geo"] = ("seq", lambda m=m, ma=ma, n=int(steps), t=float(tau0): torch.cat(
                [canonize(m, ma, Zg[j:j + a.bs], n, t).to(Z.dtype) for j in range(0, len(Z), a.bs)]))
        methods[name] = (lambda z, m=m, ma=ma, n=int(steps), t=float(tau0): canonize(m, ma, z, n, t)) if ma.K == 1 else \
            ("seq", lambda m=m, ma=ma, n=int(steps), t=float(tau0): torch.cat([canonize(  # 时序规范化器：每帧取同一 demo 的最近 K 帧
                m, ma, Z[torch.maximum(j[:, None] + torch.arange(-ma.K + 1, 1, device=dev), fst[j][:, None])], n, t).to(Z.dtype)
                for j in torch.arange(len(Z), device=dev).split(a.bs)]))
    res = {}
    for name, fn in methods.items():
        if isinstance(fn, tuple):  # 递归滤波
            Zm = fn[1]()
        else:
            Zm = torch.empty_like(Z)
            for j in range(0, len(Z), a.bs):
                Zm[j:j + a.bs] = Z[j:j + a.bs] if fn is None else fn(Z[j:j + a.bs]).to(Z.dtype)
        if fn is None:
            Zm[P.io] = Z[P.ic]
        e = ((Zm[P.io].float() - Z[P.ic].float()) ** 2).mean((1, 2))
        ec = ((Zm[cl].float() - Z[cl].float()) ** 2).mean((1, 2))

        def idm_out(g, d0):
            return torch.cat([idm(Zm[torch.maximum(g[j:j + 1024, None] + offs, d0[j:j + 1024, None])]).float()
                              for j in range(0, len(g), 1024)])
        with torch.autocast("cuda", torch.bfloat16):
            out, cout = idm_out(P.io[ok], P.d0[ok]), idm_out(cl[cok], cd0[cok])
        r = {}
        for rn, m in rows.items():
            if m is None:
                r[rn] = {"err": ec.mean().item(), "idm": metrics(cout, A[cl[cok]], st)["nmse"], "n": len(ec)}
            elif m.any():
                r[rn] = {"err": e[m].mean().item(), "idm": metrics(out[m[ok]], A[P.ic[ok][m[ok]]], st)["nmse"],
                         "n": int(m.sum())}
        res[name] = r
        print(f"== {name}", flush=True)
        for rn, x in r.items():
            print(f"  {rn:<36} n {x['n']:>6}  对规范真值 MSE {x['err']:.4f}  IDM nmse {x['idm']:.4f}", flush=True)
        del Zm
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "results": res}, indent=1, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a("--sets", default="data_camaug,data_lightaug,data_mixaug"); a("--h5", default="libero_data")
    a("--norm", default="runs/goal/chunk8_t5l_paug_tnoise/ckpt.pt", help="WM ckpt：用它的 norm（与规范化模型的一致）")
    a("--idm", default="runs/goal/idm/idm_large_s4000.pt"); a("--prior", default="runs/goal/prior/ckpt.pt")
    a("--purify", default="0.5,5,4,0.15;0.7,5,4,0.15", help="先验净化的 τ0,K,steps,snr，多组用 ; 分隔")
    a("--models", default="", help="名字:ckpt:步数:τ0，多组用 , 分隔")
    a("--geo", action="store_true", help="另评测\"几何对齐（geowarp.py）+ 规范化器\"")
    a("--filters", default="", help="名字:κ,s,pool,steps,τ0[,cfg]：WM 条件后验净化（postfilt.py，WM = --norm 的 ckpt），多组用 ; 分隔")
    a("--bs", type=int, default=512); a("--out", default="runs/canon/offline.json")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
