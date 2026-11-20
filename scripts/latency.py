"""RoboTwin 策略的单次决策耗时（不开仿真，帧随机：耗时与内容无关）。照 probwm.robotwin_eval 的推理路径：双腕 WM 的编码分支（头部同 Policy.encode，含 canon）、
腕部 IDM、--idm_ens 集成、--idm_stale 高效模式。CPU uint8 帧 + 本体 → CPU 动作，前后同步；稳态 = 去掉前 --warmup 次决策与每集首次决策。
  crit  = 帧 → 动作拷回 CPU（--idm_stale 时 WM 这一步在动作之后做，不在关键路径上）；
  total = 帧 → 本次决策全部计算结束；encode = H2D + 编码（+ canon）；act = WM + IDM（+ 集成）+ D2H（+ 动作之后的 WM 步）。
  --prof 另跑一遍分项计时（编码 / canon / WM / IDM，各项前后同步；只看分项，关键路径与总计以不插同步的那遍为准）。
例（WC2 高效版，B=1，开全部提速选项）：
  python scripts/latency.py --wm $WMRUNS/rteasy/wmX2F5Clw3WC2/ckpt.pt --wm_data $WMDATA --idm $WMRUNS/rteasy/idmPLwr.pt \
      --idm_ens $WMRUNS/rteasy/idmPLwrs1.pt,$WMRUNS/rteasy/idmPLwrs2.pt --idm_stale obs --graphs --idm_graph --idm_ms --compile default --prof --B 1 --out lat_wc2st.json
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True); p.add_argument("--idm", required=True); p.add_argument("--canon", default=""); p.add_argument("--canon_steps", type=int, default=4)
    p.add_argument("--idm_ens", default=""); p.add_argument("--idm_stale", default=""); p.add_argument("--graphs", action="store_true")
    p.add_argument("--idm_ens_par", action="store_true", help="closed_loop --idm_ens_par（集成 vmap 一次前向）")
    p.add_argument("--idm_graph", action="store_true", help="closed_loop --idm_graph（IDM 骨干捕获 CUDA graph）")
    p.add_argument("--chk_graph", action="store_true", help="配 --idm_graph：截下 IDM 的输入，同 fixed_eps 下逐网络比原 forward 与 graph 版（Net.infer）")
    p.add_argument("--params", action="store_true", help="另数参数量（从 ckpt / 权重建出的模块数 parameters()）：WM（含腕部输入层）、单个 IDM、canon、DINOv3-S、flan-t5-large 编码器")
    p.add_argument("--compile", default="", help="closed_loop --compile 模式（default 等；首次决策要编译，算在 --warmup 里）"); p.add_argument("--compile_on", default="wm,canon,idm,enc")
    p.add_argument("--idm_ms", action="store_true", help="closed_loop --idm_ms（集成成员多流并发）")
    p.add_argument("--chk_ms", action="store_true", help="配 --idm_ms：① 同帧同种子下另建一个顺序版 Policy，逐次决策比动作（逐位）；② 截下 IDM 输入，fixed_eps 与重置种子两种噪声下逐网络比顺序 graph 版 / 原 forward 与多流版")
    p.add_argument("--chk_par", action="store_true", help="配 --idm_ens_par：先用顺序版跑几次决策、截下 IDM 的输入，同噪声（fixed_eps）下逐成员比顺序版与并行版的输出")
    p.add_argument("--B", type=int, default=1, help="同时决策的 env 数（同 batch 对比）"); p.add_argument("--n", type=int, default=300)
    p.add_argument("--ep", type=int, default=100, help="每这么多次决策算一集（首次决策从头采）"); p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--wm_data", default="", help="closed_loop --wm_data：WM 训练数据根改用本机同内容的路径（info.json / 文本表 / meta/tasks.json）；默认空 = 用 ckpt 里的 --data")
    p.add_argument("--prof", action="store_true", help="计时之后再跑一遍 --n 次分项计时：编码 / canon / WM / IDM（主 IDM 与集成成员分开），各项前后同步（只看分项，关键路径 / 总计算取前一遍不插同步的数）")
    o = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from probwm.closed_loop import Policy
    from probwm.encode import pool
    from probwm.robotwin_eval import cl_args
    dev = torch.device("cuda")
    torch.manual_seed(0)
    kw = dict(policy="wm+idm", wm=o.wm, idm=o.idm, bc="", text="", sampler="scratch4", cfg=3.0, cfg_far=-1.0, plan=1, exec=2, canon=o.canon,
              canon_steps=o.canon_steps, canon_tau0=0.0, fsteps=2, roll="1,0,8", seed=0, n_env=o.B, graphs=o.graphs)
    if o.idm_ens:
        kw["idm_ens"] = o.idm_ens
    if o.idm_stale:
        kw["idm_stale"] = o.idm_stale
    if o.wm_data:
        kw["wm_data"] = o.wm_data
    if o.idm_ens_par:
        kw["idm_ens_par"] = True
    if o.idm_graph:
        kw["idm_graph"] = True
    if o.idm_ms:
        kw["idm_ms"] = True
    if o.compile:
        kw["compile"], kw["compile_on"] = o.compile, o.compile_on
    pol = Policy(cl_args(**kw), dev)
    xw = bool(pol.wa.__dict__.get("tgt_tok", 0))  # 同 robotwin_eval：train --tgt_tok 的多视角 WM，上下文带双腕
    T, D = pol.dims
    wT = getattr(pol.na, "wT", None) or T

    @torch.no_grad()
    def encode(ob, pol=pol):  # 同 robotwin_eval.encode 的 xw / 腕部 IDM 分支（非视频编码器）；pol 可换成另一个 Policy（--chk_ms）
        if xw:
            nw_, pol.nw = pol.nw, False
            try:
                z = pol.encode([q[0] for q in ob])
            finally:
                pol.nw = nw_
            nx = T - z.shape[1]
            x = torch.from_numpy(np.stack([q[j] for j in (1, 2) for q in ob])).to(dev)
            with torch.autocast("cuda", torch.bfloat16):
                f = pol.enc(x)
            wl_, wr_ = pool(f, int(round((f.shape[1] / (nx // 2)) ** 0.5))).chunk(2)
            zw_ = None
            if pol.nw:
                wl0, wr0 = f.chunk(2)
                k_ = int(round((f.shape[1] / (wT // 2)) ** 0.5))
                zw_ = ((torch.cat([pool(wl0, k_), pool(wr0, k_)], 1).float() - pol.wmu) / pol.wsd).half()
            return torch.cat([z, ((torch.cat([wl_, wr_], 1).float() - pol.mu) / pol.sd).half()], 1), zw_
        if not pol.nw:
            return pol.encode(ob), None
        with torch.autocast("cuda", torch.bfloat16):
            f = pol.enc(torch.from_numpy(np.stack([q[j] for j in range(3) for q in ob])).to(dev))
        h, wl, wr = f.chunk(3)
        k = int(round((T / (wT // 2)) ** 0.5))
        w = torch.cat([pool(wl, k), pool(wr, k)], 1)
        z = ((h.float() - pol.mu) / pol.sd).half()
        return (pol.canonize(z) if o.canon and not pol.rawh else z), ((w.float() - pol.wmu) / pol.wsd).half()

    B = o.B
    acc = {}

    def timed(name, f):  # --prof：前后同步计时，累加到 acc[name]
        def g(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = f(*a, **k)
            torch.cuda.synchronize()
            acc[name] = acc.get(name, 0.0) + (time.perf_counter() - t) * 1e3
            return r
        return g

    class TNet:  # IDM 包一层计时，其余属性照转
        def __init__(s, net, name):
            s.__dict__["_n"], s.__dict__["_name"] = net, name

        def __call__(s, *a, **k):
            return timed(s._name, s._n)(*a, **k)

        def __getattr__(s, k):
            return getattr(s._n, k)
    rgb = np.random.randint(0, 255, (64, 3, 256, 256, 3), np.uint8)  # (帧, 视角 head / wl / wr, H, W, 3)
    hist = torch.zeros(B, pol.K, T, D, dtype=torch.float16, device=dev)
    whist = torch.zeros(B, pol.nw, wT, D, dtype=torch.float16, device=dev) if pol.nw else None
    phist = torch.zeros(B, pol.npf, pol.na.pdim, device=dev) if pol.npf else None
    idx, ti, sync, rec = torch.arange(B, device=dev), torch.zeros(B, dtype=torch.long, device=dev), torch.cuda.synchronize, []
    def run(n, prof):
      nonlocal hist, whist, phist
      for i in range(n):
          fresh = i % o.ep == 0
          fr = torch.full((B,), fresh, device=dev)
          ob = [tuple(rgb[(i + b) % len(rgb)]) for b in range(B)]
          q = np.zeros((B, 14), np.float32)
          sync()
          t0 = time.perf_counter()
          z, zw = encode(ob)
          sync()
          t1 = time.perf_counter()
          f4 = fr[:, None, None, None]
          h = torch.cat([hist[:, 1:], z[:, None]], 1)
          hist = h = torch.where(f4, z[:, None].expand_as(h), h)
          wh = ph = None
          if pol.nw:
              wh = torch.cat([whist[:, 1:], zw[:, None]], 1)
              whist = wh = torch.where(f4, zw[:, None].expand_as(wh), wh)
          if pol.npf:
              pv = torch.from_numpy(q).to(dev)
              ph = torch.cat([phist[:, 1:], pv[:, None]], 1)
              phist = ph = torch.where(fr[:, None, None], pv[:, None].expand_as(ph), ph)
          pol.t_act = None
          with torch.no_grad():
              pol.act(h, ti, idx, fr, wh, ph)
          sync()
          t3 = time.perf_counter()
          ta = pol.t_act or t3
          r_ = dict(i=i, fresh=fresh, crit=(ta - t0) * 1e3, total=(t3 - t0) * 1e3, encode=(t1 - t0) * 1e3, act=(t3 - t1) * 1e3)
          if prof:
              r_.update({f"p_{k}": v for k, v in acc.items()}); acc.clear()
          (prec if prof else rec).append(r_)

    prm = {}
    if o.params:
        from probwm.encode import ENCODERS, src
        from transformers import AutoModel, T5EncoderModel
        cnt = lambda m: int(sum(x.numel() for x in m.parameters()))
        prm["wm"] = cnt(pol.wm)
        prm["wm_wrist_in"] = int(sum(x.numel() for k, x in pol.wm.named_parameters() if k.startswith(("cx_in", "xpos"))))
        prm["idm_single"] = cnt(pol.net)
        if o.canon:
            from probwm.canon import load as cload
            prm["canon"] = cnt(cload(o.canon, "cpu")[0])
        enc_name = json.loads((Path(pol.wa.data) / f"feats/{pol.wa.encoder}/info.json").read_text()).get("base", pol.wa.encoder)
        prm["vision_" + enc_name] = cnt(AutoModel.from_pretrained(src(ENCODERS[enc_name][0])))
        prm["text_flan-t5-large_encoder"] = cnt(T5EncoderModel.from_pretrained(src("google/flan-t5-large")))
        print("PARAMS", json.dumps(prm), flush=True)
    prec, chk = [], {}
    if o.chk_ms:  # ① 端到端：同帧、每次决策前同种子，多流版 Policy 与另建的顺序版 Policy 逐次决策比动作
        assert pol.ms is not None, "--chk_ms 要配 --idm_ms --idm_ens"
        ref = Policy(cl_args(**{**kw, "idm_ms": False}), dev)
        assert ref.ms is None

        def e2e(P, n):
            hs = torch.zeros(B, P.K, T, D, dtype=torch.float16, device=dev)
            ws = torch.zeros(B, P.nw, wT, D, dtype=torch.float16, device=dev) if P.nw else None
            ps = torch.zeros(B, P.npf, P.na.pdim, device=dev) if P.npf else None
            outs = []
            for i in range(n):
                fr = torch.full((B,), i == 0, device=dev)
                ob = [tuple(rgb[(i + b) % len(rgb)]) for b in range(B)]
                torch.manual_seed(1000 + i)
                z, zw = encode(ob, P)
                f4 = fr[:, None, None, None]
                hs = torch.where(f4, z[:, None].expand(-1, P.K, -1, -1), torch.cat([hs[:, 1:], z[:, None]], 1))
                if P.nw:
                    ws = torch.where(f4, zw[:, None].expand(-1, P.nw, -1, -1), torch.cat([ws[:, 1:], zw[:, None]], 1))
                if P.npf:
                    pv = torch.zeros(B, ps.shape[-1], device=dev)
                    ps = torch.where(fr[:, None, None], pv[:, None].expand_as(ps), torch.cat([ps[:, 1:], pv[:, None]], 1))
                with torch.no_grad():
                    outs.append(P.act(hs, ti, idx, fr, ws, ps))
                torch.cuda.synchronize()
            return outs
        a_ms, a_ref = e2e(pol, 6), e2e(ref, 6)
        chk["e2e"] = dict(n=len(a_ms), bitwise=[bool(np.array_equal(x, y)) for x, y in zip(a_ms, a_ref)],
                          max_abs=[float(np.abs(x - y).max()) for x, y in zip(a_ms, a_ref)], scale=float(max(np.abs(y).max() for y in a_ref)))
        del ref
        torch.cuda.empty_cache()
        got, net0 = {}, pol.net  # ② 逐网络：截下多流版第 3 次决策主 IDM 的输入

        class Rec3:
            def __init__(s_, n):
                s_.__dict__["_n"] = n

            def __call__(s_, *a, **k):
                got["a"], got["k"] = a, k
                return s_._n(*a, **k)

            def __getattr__(s_, k):
                return getattr(s_._n, k)
        pol.net = Rec3(net0)
        run(3, False)
        pol.net = net0
        rec.clear()
        a_, k_ = got["a"], got["k"]
        nets = [net0] + pol.enets
        blk = lambda n_: n_(*a_, **k_).float()
        res_ = {}
        for mode in ("fixed_eps", "rng"):
            eps = torch.randn(len(a_[0]), net0.n_act, net0.ad, device=dev) if mode == "fixed_eps" else None
            for n_ in nets:
                n_.fixed_eps = eps
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                torch.manual_seed(7)
                sq = [blk(n_) for n_ in nets]
                torch.manual_seed(7)
                mm, me = pol.ens_ms(blk, 1, every=True)
                torch.cuda.synchronize()
                eg = [type(n_).forward(n_, *a_, **k_).float() for n_ in nets] if mode == "fixed_eps" else None
            for n_ in nets:
                n_.fixed_eps = None
            res_[mode] = dict(member_bitwise=[bool(torch.equal(x, y)) for x, y in zip(sq, me)], mean_bitwise=bool(torch.equal(torch.stack(sq).mean(0), mm)),
                              max_abs=[float((x - y).abs().max()) for x, y in zip(sq, me)], scale=float(max(x.abs().max() for x in sq)),
                              **({"eager_bitwise": [bool(torch.equal(x, y)) for x, y in zip(eg, me)]} if eg is not None else {}))
        chk["member"] = res_
        print("CHK_MS", json.dumps(chk), flush=True)
    if o.chk_par:  # 顺序版跑 3 次决策（第 1 次新 episode），截下最后一次主 IDM 的输入；同 fixed_eps 下比各成员输出
        assert pol.ens_par is not None, "--chk_par 要配 --idm_ens_par --idm_ens"
        par_, got = pol.ens_par, {}
        net0 = pol.net

        class Rec:
            def __init__(s_, n):
                s_.__dict__["_n"] = n

            def __call__(s_, *a, **k):
                got["a"], got["k"] = a, k
                return s_._n(*a, **k)

            def __getattr__(s_, k):
                return getattr(s_._n, k)
        pol.ens_par, pol.net = None, Rec(net0)
        run(3, False)
        pol.net, pol.ens_par = net0, par_
        rec.clear()
        a_, k_ = got["a"], got["k"]
        eps = torch.randn(len(a_[0]), net0.n_act, net0.ad, device=dev)
        for n_ in [net0] + pol.enets:
            n_.fixed_eps = eps
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            seq_ = torch.stack([n_(*a_, **k_).float() for n_ in [net0] + pol.enets])
            par_o = pol.ens_par(*a_, **k_).float()
        for n_ in [net0] + pol.enets:
            n_.fixed_eps = None
        d_ = (seq_ - par_o).abs()
        chk = dict(shape=list(par_o.shape), max_abs=[float(x) for x in d_.flatten(1).max(1).values], scale=float(seq_.abs().max()),
                   mean_max_abs=float((seq_.mean(0) - par_o.mean(0)).abs().max()), bitwise=bool(torch.equal(seq_, par_o)),
                   allclose_bf16=bool(torch.allclose(seq_, par_o, rtol=1.6e-2, atol=1.6e-2)))
        print("CHK_PAR", json.dumps(chk), flush=True)
    if o.chk_graph:  # 截下第 3 次决策主 IDM 的输入（graph 版照常跑），同 fixed_eps 下逐网络比：原 forward（eager 骨干）对 Net.infer（graph 骨干）
        assert o.idm_graph, "--chk_graph 要配 --idm_graph"
        got, net0 = {}, pol.net

        class Rec2:
            def __init__(s_, n):
                s_.__dict__["_n"] = n

            def __call__(s_, *a, **k):
                got["a"], got["k"] = a, k
                return s_._n(*a, **k)

            def __getattr__(s_, k):
                return getattr(s_._n, k)
        pol.net = Rec2(net0)
        run(3, False)
        pol.net = net0
        rec.clear()
        a_, k_ = got["a"], got["k"]
        eps = torch.randn(len(a_[0]), net0.n_act, net0.ad, device=dev)
        res_ = []
        for n_ in [net0] + pol.enets:
            n_.fixed_eps = eps
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                o_old = type(n_).forward(n_, *a_, **k_).float()
                o_new = n_(*a_, **k_).float()
                o_new2 = n_(*a_, **k_).float()  # 第二次走已捕获的图（第一次是捕获）
            n_.fixed_eps = None
            res_.append(dict(bitwise=bool(torch.equal(o_old, o_new) and torch.equal(o_old, o_new2)), max_abs=float(max((o_old - o_new).abs().max(), (o_old - o_new2).abs().max())),
                             scale=float(o_old.abs().max())))
        chk = dict(graph=res_)
        print("CHK_GRAPH", json.dumps(chk), flush=True)
    run(o.n, False)
    if o.prof:
        pol.enc, pol.canonize = timed("enc", pol.enc), timed("canon", pol.canonize)
        for m in ("rolling", "rolling_pre", "roll_commit"):
            setattr(pol, m, timed("wm", getattr(pol, m)))
        if pol.ens_par is not None:  # 并行集成：一次 vmap 前向算全部成员，记作 idm_par
            pol.ens_par = timed("idm_par", pol.ens_par)
        elif pol.ms is not None:  # 多流集成：整段（主 IDM + 成员并发）记作 idm_ms，不逐网络插同步（会把并发串起来）
            pol.ens_ms = timed("idm_ms", pol.ens_ms)
        else:
            pol.net = TNet(pol.net, "idm")
            pol.enets = [TNet(e, "idm_ens") for e in (pol.enets or [])]
        acc.clear()
        run(o.n, True)

    def st(rows, k):
        v = [r[k] for r in rows]
        return dict(n=len(v), mean=float(np.mean(v)), p50=float(np.median(v)), p95=float(np.percentile(v, 95))) if v else dict(n=0)
    ss = [r for r in rec[o.warmup:] if not r["fresh"]]
    ps = [r for r in prec[o.warmup:] if not r["fresh"]]
    pk = sorted({k for r in ps for k in r if k.startswith("p_")})
    res = dict(args=vars(o), device=torch.cuda.get_device_name(), chk_par=chk, params=prm, xw=xw, nw=int(pol.nw or 0), n_ens=len(pol.enets or []),
               steady={k: st(ss, k) for k in ("crit", "total", "encode", "act")}, prof={k[2:]: st([{k: r.get(k, 0.0)} for r in ps], k) for k in pk} if ps else {}, fresh={k: st([r for r in rec[o.warmup:] if r["fresh"]], k) for k in ("crit", "total")},
               records=rec)
    Path(o.out).parent.mkdir(parents=True, exist_ok=True)
    Path(o.out).write_text(json.dumps(res, indent=1))
    res["prof_records"] = prec
    print(json.dumps({k: v for k, v in res.items() if k not in ("records", "prof_records")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
