"""线性 flow matching：x_τ = (1-τ)ε + τ·x1（τ=1 为数据端），u ≈ x1 - ε，score s = -(x - τu)/(1-τ)。
mode: abs = 绝对目标（主线 A，推理用 B：加噪→Langevin→去噪）；res = 残差（方案 C）；det = 确定性回归（D）。"""
import torch
import torch.nn.functional as F


def shift(u, a):  # a>1 时偏向噪声端
    return u / (a - (a - 1) * u)


def unshift(t, a):
    return a * t / (1 + (a - 1) * t)


def cvbase(z_t, z_p, n):  # 匀速外推 z_t + j·(z_t − z_{t-1})，j = 1..n → (B, n·T, D)
    j = torch.arange(1, n + 1, device=z_t.device, dtype=torch.float)[None, :, None, None]
    return (z_t.float()[:, None] + j * (z_t.float() - z_p.float())[:, None]).flatten(1, 2)


def target(mode, tgt, z_t, rs, z_p=None):  # cv：相对匀速外推的残差（需要 z_p = z_{t-1}）
    if mode == "cv":
        return (tgt.float() - cvbase(z_t, z_p, tgt.shape[1] // z_t.shape[1])) / rs
    z_t = z_t.repeat(1, tgt.shape[1] // z_t.shape[1], 1)
    return (tgt.float() - z_t.float()) / rs if mode == "res" else tgt.float()


def loss(model, kvs, x1, mode, alpha=3.0, p_uni=0.3, sigma=1.0, lam_tc=0.0, z_t=None, p_warm=0.0, warm_hi=0.4,
         kvs2=None, lam_pc=0.0, pw=None, slots=0, roll=0.0, churn=0.0, roll_sort=False, sw=None, dist=None, fpos=None, split=0, det_sw=False, sync=False,
         gf=0.0, gf_near=0, af=None, lead=0.0, lead_near=0, ae=None):  # pw (B,N)：逐 patch 的损失权重（train --lw_motion / --lw_obj，均值为 1）
    # ae（train --aux_ae，cap）：可调用 ae(h, ts) → 辅助损失；h = 输出头之前的逐 token 特征（model(..., ret_h=True)），ts = 逐槽层级；只接逐槽路径；默认 None = 原行为
    # det_sw（train --det_hw，ctrl）：mode det 时也按逐槽权重 sw（--hw）加权，与 flow 版同一损失配比；默认 False = 原行为（det 不加权）
    # sync（train --sync_slots，gawm P1）：逐槽路径（为了 sw 逐槽权重）但每个样本所有槽同一噪声层级，不用滚动剖面；默认 False = 原行为
    # gf > 0（rteasy 臂 G，train --goal_first）：以 gf 的概率换成目标先行两阶段（closed_loop --goal_first）的状态——阶段 1：近块（前 gf_near 个槽）纯噪声、远块任意；
    #   阶段 2：远块干净（u = 1）、近块任意；块内一半按槽降序排（滚动）、一半同一层级（新 episode 的同步采样）。干净的槽不计损失。默认 0 = 原行为
    # af (slots,)（rteasy 臂 L，train --alpha_far）：滚动样本逐槽的 shift 参数（近槽 alpha、远槽 alpha_far；同步样本不变）；默认 None = 原行为
    # lead > 0（rteasy 臂 G2，train --lead）：以 lead 的概率换成远槽领先的多步去噪（closed_loop --lead_k）的中间状态：起点（一半滚动剖面、一半同一层级）
    #   沿各槽自己到干净的路径前进，进度 s ~ U(0,1)，远槽（前 lead_near 个之后）走 1 − (1 − s)²、近槽走 s²；默认 0 = 原行为
    # split > 0（train --far_aux）：roll_sort 剖面里前 split 个槽（近槽）与其余槽（远槽）各自降序排——近槽的剖面同只有近槽的 MSN；默认 0 = 原行为
    # slots > 0：逐槽独立的噪声层级（diffusion forcing 式；train --slot_noise），x1 = slots 个槽
    # roll > 0（train --roll）：以 roll 的概率用滚动剖面（Rolling Diffusion 式，槽 j = 0 最近）：u_j = 1 − (j + φ)/slots，φ ~ U[0,1)，
    #   再以 0.5 的概率逐槽减去 U[0, churn]（推理时的回噪，均匀时间 u 上），τ_j = shift(u_j, α)；其余样本所有槽同一层级（同原训练）
    #   roll_sort：滚动样本改用"逐槽独立 U(0,1) 再按槽降序排"（近槽轻、远槽重的任意单调剖面；多尺度滚动推理的各种中间状态都覆盖到）
    # sw (slots,)：逐槽损失权重（train --hw：∝ 时距^p，均值 1）
    # (hz2) fpos (B, slots, 1, d)：目标槽的时间槽嵌入（train --any_far：按各样本的实际时距查表），传给 model；sw 也可以是逐样本的 (B, slots)
    B, dev = x1.shape[0], x1.device
    if slots and mode == "abs":
        u = torch.rand(B, slots, device=dev)
        ts = torch.where(torch.rand(B, slots, device=dev) < p_uni, u, shift(u, alpha))
        if sync:  # (gawm P1) 同滚动版里"同步"那一半样本的抽法：取第 0 槽的层级给全部槽
            ts = ts[:, :1].expand(-1, slots)
        if roll > 0:
            if roll_sort:
                ur = torch.rand(B, slots, device=dev).sort(1, descending=True).values
                if split:  # (gawm) --far_aux
                    ur = torch.cat([torch.rand(B, split, device=dev).sort(1, descending=True).values,
                                    torch.rand(B, slots - split, device=dev).sort(1, descending=True).values], 1)
                if dist is not None:  # dist (B, slots)：各槽的实际时距（train --ev_roll 的事件槽时距可变）→ 按时距排：第 k 近的槽拿第 k 轻的层级
                    ur = ur.gather(1, dist.sort(dim=1, stable=True).indices.argsort(1))
            else:
                ur = 1 - (torch.arange(slots, device=dev) + torch.rand(B, 1, device=dev)) / slots
            ur = (ur - churn * torch.rand(B, slots, device=dev) * (torch.rand(B, 1, device=dev) < 0.5)).clamp(0, 1)
            ts = torch.where(torch.rand(B, 1, device=dev) < roll, shift(ur, alpha if af is None else af), ts[:, :1].expand(-1, slots))
        if gf > 0:
            def blk(m):  # 块内层级：一半按槽降序排、一半同一层级
                return torch.where(torch.rand(B, 1, device=dev) < 0.5, torch.rand(B, m, device=dev).sort(1, descending=True).values,
                                   torch.rand(B, 1, device=dev).expand(-1, m))
            un, uf, s1 = blk(gf_near), blk(slots - gf_near), torch.rand(B, 1, device=dev) < 0.5
            ug = torch.cat([torch.where(s1, torch.zeros_like(un), un), torch.where(s1, uf, torch.ones_like(uf))], 1)
            ts = torch.where(torch.rand(B, 1, device=dev) < gf, shift(ug, alpha), ts)
        if lead > 0:
            ub = torch.where(torch.rand(B, 1, device=dev) < 0.5, torch.rand(B, slots, device=dev).sort(1, descending=True).values,
                             torch.rand(B, 1, device=dev).expand(-1, slots))
            sp = torch.rand(B, 1, device=dev)
            fr = torch.where(torch.arange(slots, device=dev)[None] >= lead_near, 1 - (1 - sp) ** 2, sp ** 2)
            ts = torch.where(torch.rand(B, 1, device=dev) < lead, shift(ub + fr * (1 - ub), alpha), ts)
        t = ts.repeat_interleave(x1.shape[1] // slots, 1)[..., None]
        e = torch.randn_like(x1)
        xt, v = (1 - t) * e + t * x1, x1 - e
        la = 0.0
        if ae is not None:  # (cap) --aux_ae：同一次前向另取输出头之前的特征
            u, hh = model(xt, ts, kvs, ret_h=True) if fpos is None else model(xt, ts, kvs, fpos=fpos, ret_h=True)
            la = ae(hh, ts)
        else:
            u = model(xt, ts, kvs) if fpos is None else model(xt, ts, kvs, fpos=fpos)
        w = (1.0 if pw is None else pw) * (1.0 if sw is None else sw.repeat_interleave(x1.shape[1] // slots, -1) if sw.dim() == 2 else sw.repeat_interleave(x1.shape[1] // slots)[None])
        if gf > 0:  # 干净的槽（阶段 2 的远块）：输入就是目标，速度 x1 − ε 不可预测，不计损失
            w = w * (ts < 1).float().repeat_interleave(x1.shape[1] // slots, 1)
            lg = (((u.float() - v.float()) ** 2).mean(-1) * w).mean()
            return lg if ae is None else lg + la
        lm = F.mse_loss(u.float(), v.float()) if pw is None and sw is None else (((u.float() - v.float()) ** 2).mean(-1) * w).mean()
        return lm if ae is None else lm + la
    if mode == "det":
        if det_sw and sw is not None and slots:  # (ctrl) 逐槽权重，均值 1（同 flow 分支的 w）
            e = ((model(torch.zeros_like(x1), torch.zeros(B, device=dev), kvs).float() - x1.float()) ** 2).mean(-1)
            return (e * sw.repeat_interleave(x1.shape[1] // slots)[None]).mean()
        return F.mse_loss(model(torch.zeros_like(x1), torch.zeros(B, device=dev), kvs).float(), x1.float())
    u = torch.rand(B, device=dev)
    t = torch.where(torch.rand(B, device=dev) < p_uni, u, shift(u, alpha))[:, None, None]
    e = torch.randn_like(x1) * (sigma if mode in ("res", "cv") else 1.0)
    if p_warm > 0 and mode == "abs":  # 训练时模拟热启动：以 p_warm 的概率从加噪的 z_t（层级 τ0）出发
        w = torch.rand(B, device=dev) < p_warm
        t0 = torch.where(w, 0.05 + (warm_hi - 0.05) * torch.rand(B, device=dev), torch.zeros(B, device=dev))[:, None, None]
        x0 = (1 - t0) * e + t0 * z_t.repeat(1, x1.shape[1] // z_t.shape[1], 1).to(x1.dtype)
        t = t0 + (1 - t0) * t                                   # τ ∈ [τ0, 1]
        xt, v = x0 + (t - t0) / (1 - t0) * (x1 - x0), (x1 - x0) / (1 - t0)
        u = model(xt, t.flatten(), kvs, t0.flatten())
    else:
        xt, v = (1 - t) * e + t * x1, x1 - e
        u = model(xt, t.flatten(), kvs) if fpos is None else model(xt, t.flatten(), kvs, fpos=fpos)
    l = F.mse_loss(u.float(), v.float()) if pw is None else (((u.float() - v.float()) ** 2).mean(-1) * pw).mean()
    if kvs2 is not None:  # 改写一致性：同一 (x_τ, τ, 历史) 换一条改写指令（另加独立文本噪声），两次预测的速度场要一致
        u2 = model(xt, t.flatten(), kvs2)
        l = 0.5 * (l + F.mse_loss(u2.float(), v.float())) + lam_pc * F.mse_loss(u.float(), u2.float())
    if lam_tc > 0 and mode == "abs":  # FlowWM 式时间一致性：一步投影终点的帧间差分匹配真实差分，按 τ 加权
        T = z_t.shape[1]
        p, g = (xt + (1 - t) * u).float().unflatten(1, (-1, T)), x1.float().unflatten(1, (-1, T))
        z0 = z_t.float()[:, None]
        d = torch.diff(torch.cat([z0, p], 1), dim=1) - torch.diff(torch.cat([z0, g], 1), dim=1)
        l = l + lam_tc * (t.flatten() * (d ** 2).mean((1, 2, 3))).mean()
    return l


@torch.no_grad()
def dc_score(model, kv, kv_null, x1, n_t=4, alpha=3.0, sigma=1.0, mode="abs"):
    """隐式分类器 log p(c|x) − log p(c) 的去噪估计（Diffusion Classifier 式）：Σ_τ [‖u_∅ − v‖² − ‖u_c − v‖²] / n_t。
    x1 为目标空间的样本 (B,N,D)；τ 取训练分布的 n_t 个分位点，ε 在整个 batch 共用（候选之间是公共随机数比较）。"""
    kvc, dev, s = cat_kv(kv, kv_null), x1.device, 0.0
    x1 = x1.float()
    for k in range(n_t):
        t = shift(torch.tensor((k + 0.5) / n_t), alpha).item()
        e = torch.randn_like(x1[:1]).expand_as(x1) * (sigma if mode in ("res", "cv") else 1.0)
        xt, v, tt = (1 - t) * e + t * x1, x1 - e, torch.full((2 * len(x1),), t, device=dev)
        uc, un = model(torch.cat([xt, xt]), tt, kvc).float().chunk(2)
        s = s + ((un - v) ** 2).mean((1, 2)) - ((uc - v) ** 2).mean((1, 2))
    return s / n_t


@torch.no_grad()
def dc_err(model, kvc, x1, ts, ns, keep=0):
    """逐 τ、逐槽的 Diffusion Classifier 去噪误差（explore：测试时计划选择，closed_loop --sel、scripts/explore_sel.py）。
    x1 (B,N,D) 候选（abs 目标空间，N = ns 槽 × T），kvc = cat_kv(条件, 无条件)，ts τ 列表（τ=1 数据端），ε 每个 τ 一份、候选间共用（公共随机数）；
    keep > 0：前 keep 个槽（近处）保持干净（τ=1），只给其余槽加噪 → 估计 log p(远 | 近, 历史, c) − log p(远 | 近, 历史)。
    → (3, len(ts), B, ns)：‖u_c − v‖²、‖u_∅ − v‖²、‖u_c − u_∅‖²（token / 通道平均）；log p(c|x) − log p(c) 的估计 ∝ Σ_τ (L_∅ − L_c)。"""
    B, N, D = x1.shape
    T, dev, x1, out = N // ns, x1.device, x1.float(), []
    for t in ts:
        tv = torch.full((ns,), float(t), device=dev)
        tv[:keep] = 1.0
        tt = tv.repeat_interleave(T)[None, :, None]
        e = torch.randn(1, N, D, device=dev).expand(B, -1, -1)
        xt, v = (1 - tt) * e + tt * x1, x1 - e
        uc, un = model(torch.cat([xt, xt]), tv[None].expand(2 * B, -1), kvc).float().chunk(2)
        f = lambda q: q.pow(2).view(B, ns, -1).mean(-1)
        out.append(torch.stack([f(uc - v), f(un - v), f(uc - un)]))
    return torch.stack(out, 1)


def rep_kv(a, k):  # (rteasy) 上下文 kv 每行重复 k 次（batch 维 repeat_interleave；附带属性同 cat_kv）
    out = type(a)((k1.repeat_interleave(k, 0), v1.repeat_interleave(k, 0)) for k1, v1 in a)
    for nm in ("heat", "cur", "kbias", "lang"):  # lang：(cap MoE-T) 池化语言嵌入
        x = getattr(a, nm, None)
        if x is not None:
            setattr(out, nm, x.repeat_interleave(k, 0))
    return out


def cat_kv(a, b):
    out = type(a)((torch.cat([k1, k2]), torch.cat([v1, v2])) for (k1, v1), (k2, v2) in zip(a, b))
    ha, hb = getattr(a, "heat", None), getattr(b, "heat", None)  # model.KV：where 热图随 batch 一起拼
    if ha is not None and hb is not None:
        out.heat = torch.cat([ha, hb])
    ca, cb = getattr(a, "cur", None), getattr(b, "cur", None)  # warp-kf：形变的源帧（当前帧）随 batch 一起拼
    if ca is not None and cb is not None:
        out.cur = torch.cat([ca, cb])
    ka, kb = getattr(a, "kbias", None), getattr(b, "kbias", None)  # (mika) --sh_alibi：交叉注意力的 key 偏置随 batch 一起拼
    if ka is not None and kb is not None:
        out.kbias = torch.cat([ka, kb])
    la, lb = getattr(a, "lang", None), getattr(b, "lang", None)  # (cap MoE-T) 池化语言嵌入随 batch 一起拼（CFG 的有 / 无语言两半）
    if la is not None and lb is not None:
        out.lang = torch.cat([la, lb])
    return out


@torch.no_grad()
def sample(model, kv, z_t, mode, steps=10, alpha=3.0, tau0=0.0, n_lan=0, snr=0.15,
           sigma=1.0, rs=None, cfg=1.0, kv_null=None, pc=False, n_sm=1, comp=None, guide=None, stagger=None, x0=None, det_cfg=False):
    """返回 (ẑ_{t+Δ}（标准化空间）, NFE)。cfg≠1 时需要传 kv_null（语言被丢掉时的上下文）。
    det_cfg（closed_loop --det_cfg，ctrl）：mode det 时也做 CFG，x̂ = x̂_∅ + cfg·(x̂_c − x̂_∅)（条件均值外推）；默认 False = 原行为（det 不做 CFG）。
    n_sm>1：平滑引导，kv 含每个样本 n_sm 份（相邻排列）加噪文本的上下文，条件速度取 n_sm 份的平均（文本嵌入空间的随机平滑）。
    comp=(kvs, W)：多条件组合引导（Composable Diffusion 式），u = u_∅ + Σ_i W[:, i]·(u_i − u_∅)，kvs 为 n 个条件各自的上下文、
    W (B,n) 逐样本权重（负权重 = 负向条件）；此时 kv / cfg 不用，需要 kv_null。"""
    z_t = z_t.float()
    B, dev, nfe = z_t.shape[0], z_t.device, 0
    kvc = cat_kv(kv, kv_null) if cfg != 1.0 and n_sm == 1 else kv
    if comp is not None:
        assert n_sm == 1 and kv_null is not None
        kvc, W = kv_null, comp[1].float()
        for k in comp[0]:
            kvc = cat_kv(kvc, k)

    def u(x, t, c0=0.0):
        nonlocal nfe
        nfe += 1
        tt = t.to(dev)[None].expand(len(x), -1) if torch.is_tensor(t) else torch.full((len(x),), t, device=dev)  # 张量 t (ns,)：逐槽层级
        t0 = torch.full((len(x),), c0, device=dev)
        if n_sm > 1:
            r = lambda v: v.repeat_interleave(n_sm, 0)
            uc = model(r(x), r(tt), kv, r(t0)).unflatten(0, (B, n_sm)).mean(1)
            return uc if cfg == 1.0 else (un := model(x, tt, kv_null, t0)) + cfg * (uc - un)
        if comp is not None:
            r = len(comp[0]) + 1
            un, *us = model(x.repeat(r, 1, 1), tt.repeat(r, *[1] * (tt.dim() - 1)), kvc, t0.repeat(r)).chunk(r)
            return un + sum(W[:, i, None, None] * (ui - un) for i, ui in enumerate(us))
        if cfg == 1.0:
            return model(x, tt, kvc, t0)
        uc, un = model(torch.cat([x, x]), torch.cat([tt, tt]), kvc, torch.cat([t0, t0])).chunk(2)
        return un + cfg * (uc - un)

    if mode == "det":
        if det_cfg and cfg != 1.0 and kv_null is not None and n_sm == 1 and comp is None:  # (ctrl)
            xc, xn = model(torch.zeros_like(z_t).repeat(2, 1, 1), torch.zeros(2 * B, device=dev), cat_kv(kv, kv_null)).chunk(2)
            return xn + cfg * (xc - xn), 1
        return model(torch.zeros_like(z_t), torch.zeros(B, device=dev), kv), 1
    if mode in ("res", "cv"):  # cv：z_t 传进来的是匀速外推的基线（rollout 里算）
        x, tau0 = sigma * torch.randn_like(z_t), 0.0
    elif tau0 > 0:
        if pc:  # 对照（不用 z_t）：从纯噪声沿同一网格积分到 τ0，再做同样的 Langevin
            x, g = torch.randn_like(z_t), shift(torch.linspace(0, 1, steps + 1), alpha)
            g = g[: int(torch.argmin((g - tau0).abs())) + 1].tolist()
            for t0, t1 in zip(g[:-1], g[1:]):
                x = x + (t1 - t0) * u(x, t0)
            tau0 = g[-1] if g[-1] > 0 else tau0
        else:
            x = (1 - tau0) * (torch.randn_like(z_t) if x0 is None else x0.float()) + tau0 * z_t   # 你的方案：z_t 加噪回 τ0（x0：给定的噪声）
        for _ in range(n_lan):                                  # 在 τ0 层做 Langevin
            s = -(x - tau0 * u(x, tau0)) / (1 - tau0)
            xi = torch.randn_like(x)
            eta = 2 * (snr * xi.flatten(1).norm(dim=1) / s.flatten(1).norm(dim=1).clamp_min(1e-8)) ** 2
            x = x + eta[:, None, None] * s + (2 * eta)[:, None, None].sqrt() * xi
    else:
        x = torch.randn_like(z_t) if x0 is None else x0.float()  # x0：给定的初始噪声（如每个 episode 固定一份，相邻决策落在同一模态）
    if stagger is not None and mode == "abs":  # 交错去噪（逐槽噪声层级训练的模型）：远槽领先、近槽滞后；stagger = (L, 槽数)，槽按时距由近到远排列
        L, ns = stagger
        lag = L * (1 - torch.arange(ns, dtype=torch.float) / max(ns - 1, 1))  # 最近的槽滞后 L，最远的不滞后
        lvl = lambda k: shift((k / steps * (1 + L) - lag).clamp(0, 1), alpha)
        T_ = x.shape[1] // ns
        for k in range(steps):
            a0, a1 = lvl(k), lvl(k + 1)
            x = x + (a1 - a0).to(dev).repeat_interleave(T_)[None, :, None] * u(x, a0)
        return x, nfe
    c0 = tau0 if (getattr(model, "warm", False) and tau0 > 0 and not pc and mode == "abs") else 0.0
    ts = shift(torch.linspace(unshift(torch.tensor(tau0), alpha).item(), 1, steps + 1), alpha).tolist()
    for t0, t1 in zip(ts[:-1], ts[1:]):                         # Euler 积分 τ0 → 1
        if guide is not None and getattr(guide, "jac", False) and t0 <= guide.tmax:  # 穿过 WM 的引导梯度：∇_x E(x + (1−τ)u(x))
            with torch.enable_grad():
                xg = x.detach().requires_grad_(True)
                vg = u(xg, t0, c0)
                g, = torch.autograd.grad(guide.energy(xg + (1 - t0) * vg), xg)
            v = vg.detach()
            nv, ng = v.flatten(1).norm(dim=1), g.flatten(1).norm(dim=1).clamp_min(1e-8)
            v = v - guide.w * (nv / ng)[:, None, None] * g
        else:
            v = u(x, t0, c0)
            if guide is not None:  # 能量引导：guide(x̂1 = x + (1−τ)v, v, τ) → 速度增量（objtok.make_guide）
                v = v + guide(x + (1 - t0) * v, v, t0)
        x = x + (t1 - t0) * v
    return (z_t + x * rs if mode in ("res", "cv") else x), nfe


@torch.no_grad()
def roll(model, kv, x, alpha, k=1, churn=0.0, n_lan=0, grow=False, snr=0.15, cfg=1.0, kv_null=None):
    """滚动去噪的一次决策（closed_loop --roll；WM 最好用 train --roll 训过）。x (B, n·T, D)：上一次决策末的窗口，第 j 槽（j = 0 最近）
    处于均匀时间 u = 1 − j/n（τ = shift(u, α)，第 0 槽已干净）。平移一帧（最远槽补纯噪声）→ 回噪 churn（u 上；grow 时第 j 槽回 churn·(j+1)/n；
    按线性插值的前向核 x' = (τ'/τ)x + σξ）→ n_lan 步 Langevin 校正（各槽在自己的层级，s = −(x − τû)/(1−τ)，步长按 SNR）→ k 个 Euler 步把各槽推到
    u = 1 − j/n。返回 (新窗口, 给 IDM 的计划 = 最后一步的一步估计 x + (1−τ)û（第 0 槽即干净帧）, NFE)。"""
    n = model.n_fut
    B, N, D = x.shape
    T, dev = N // n, x.device
    kvc = cat_kv(kv, kv_null) if cfg != 1.0 else kv
    rep = lambda q: q.repeat_interleave(T)[None, :, None]

    def vel(x, tau):  # tau (n,)：逐槽层级
        tt = tau[None].expand(len(x), -1)
        if cfg == 1.0:
            return model(x, tt, kvc).float()
        uc, un = model(torch.cat([x, x]), torch.cat([tt, tt]), kvc).float().chunk(2)
        return un + cfg * (uc - un)

    ue = 1 - torch.arange(n, device=dev, dtype=torch.float) / n
    us = ue - 1 / n
    x = torch.cat([x[:, T:].float(), torch.randn(B, T, D, device=dev)], 1)
    if churn > 0:
        u2 = (us - churn * ((torch.arange(n, device=dev) + 1) / n if grow else 1.0)).clamp_min(0)
        t1, t2 = shift(us, alpha), shift(u2, alpha)
        r = torch.where(t1 > 1e-6, t2 / t1.clamp_min(1e-6), torch.ones_like(t1))
        x = rep(r) * x + rep(((1 - t2) ** 2 - (r * (1 - t1)) ** 2).clamp_min(0).sqrt()) * torch.randn_like(x)
        us = u2
    tl = shift(us, alpha)
    nrm = lambda q: q.view(B, n, T * D).norm(dim=-1).repeat_interleave(T, 1)[..., None]
    for _ in range(n_lan):
        s = -(x - rep(tl) * vel(x, tl)) / rep(1 - tl).clamp_min(1e-3)
        xi = torch.randn_like(x)
        eta = 2 * (snr * nrm(xi) / nrm(s).clamp_min(1e-8)) ** 2
        x = x + eta * s + (2 * eta).sqrt() * xi
    for i in range(k):
        ta, tb = shift(us + (ue - us) * i / k, alpha), shift(us + (ue - us) * (i + 1) / k, alpha)
        v = vel(x, ta)
        p = x + rep(1 - ta) * v
        x = x + rep(tb - ta) * v
    return x, p, k + n_lan  # NFE 同 sample 的口径（CFG 的两路算一次）


@torch.no_grad()
def msroll_init(c, has, alpha, churn=0.0, rank=None):
    """(xden) msroll 去噪前的状态（与 msroll 内部逐行相同）：→ (x (B, n·T, D) 各槽加噪回自己层级的窗口, us (B, n) 均匀时间层级)。
    闭环 --idm_tau pre：噪声感知 IDM 直接读这个未去噪的窗口（τ = shift(us, α)），WM 这一步可以放到动作发出之后再做（msroll(init=…)）。"""
    B, n, T, D = c.shape
    rk = torch.arange(n, device=c.device)[None] if rank is None else rank
    us = torch.where(has, (1 - (rk + 1) / n - churn).clamp_min(0), torch.zeros(B, n, device=c.device))
    t = shift(us, alpha)
    return (t[..., None, None] * c.float() + (1 - t[..., None, None]) * torch.randn(B, n, T, D, device=c.device)).flatten(1, 2), us


@torch.no_grad()
def msroll(model, kv, c, has, alpha, k=1, churn=0.0, cfg=1.0, kv_null=None, rank=None, cfgv=None, init=None, us=None, uf=None, lead=None):
    """多尺度（稀疏时距槽，如 {1,2,4,8}）滚动的一次决策（closed_loop --roll 配 train --tgt_offs 的 WM）。c (B, n, T, D)：各槽对应绝对时刻
    最近一次的干净估计，has (B, n)：有没有估计。有估计的槽加噪回该槽的层级 u_j = 1 − (j+1)/n − churn（"到站"时重新加噪，近槽轻、远槽重），
    没有的（新出现的最远帧、新 episode）从纯噪声开始；以新的真实历史为条件，各槽沿各自的层级用 k 个 Euler 步同时去噪到干净。→ (干净估计 (B, n, T, D), NFE)"""
    B, n, T, D = c.shape
    dev = c.device
    kvc = cat_kv(kv, kv_null) if cfg != 1.0 or cfgv is not None else kv  # cfgv (n,)：逐槽 CFG 权重（closed_loop --cfg_far），给了就忽略 cfg
    rk = torch.arange(n, device=dev)[None] if rank is None else rank  # rank (B, n)：各槽按时距的名次（closed_loop 的滚动事件槽；默认 = 槽序）
    # (rteasy 臂 G) us (B, n)：给定各槽的起始层级（代替按槽序的滚动层级）；uf (B, n)：各槽去噪到的层级（默认 1 = 干净；uf = us 的槽不动）
    # (rteasy 臂 G2) lead (n,) bool：远槽领先——第 i 步的进度远槽取 1 − (1 − i/k)²、其余取 (i/k)²；alpha 可以是逐槽的 (n,)（臂 L）
    if init is not None:  # (xden) 给定去噪前的状态 (x, us)（msroll_init 算好的，--idm_tau pre），跳过加噪
        x, us = init
    else:
        us = torch.where(has, (1 - (rk + 1) / n - churn).clamp_min(0), torch.zeros(B, n, device=dev)) if us is None else us
        t = shift(us, alpha)
        x = (t[..., None, None] * c.float() + (1 - t[..., None, None]) * torch.randn(B, n, T, D, device=dev)).flatten(1, 2)
    ue = 1 if uf is None else uf
    fp = (lambda q: q) if lead is None else (lambda q: torch.where(lead, 1 - (1 - q) ** 2, q ** 2))
    for i in range(k):
        ta, tb = shift(us + (ue - us) * fp(i / k), alpha), shift(us + (ue - us) * fp((i + 1) / k), alpha)
        if cfg == 1.0 and cfgv is None:
            v = model(x, ta, kvc).float()
        else:
            uc, un = model(torch.cat([x, x]), torch.cat([ta, ta]), kvc).float().chunk(2)
            v = un + (cfg if cfgv is None else cfgv.float().repeat_interleave(T)[None, :, None]) * (uc - un)
        x = x + (tb - ta).repeat_interleave(T, 1)[..., None] * v
    return x.unflatten(1, (n, T)), k


@torch.no_grad()
def step1(model, kv, x, t, end, cfg=1.0, kv_null=None, fpos=None):
    """一次逐槽 Euler 前向（闭环 --prog / --abs_far，见 closed_loop.Policy.rolling_x）：x (B, n, T, D) 在各槽层级 t (B, n)（已 shift）上，
    → (干净端点估计 x + (1 − t)·v, 走到 end (B, n) 的部分态 x + (end − t)·v)，都是 (B, n, T, D)。fpos：替换目标槽的时间槽嵌入（model.trunk）。"""
    B, n, T, D = x.shape
    xf = x.float().flatten(1, 2)
    fk = {} if fpos is None else {"fpos": fpos if cfg == 1.0 else torch.cat([fpos, fpos])}
    if cfg == 1.0:
        v = model(xf, t, kv, **fk).float()
    else:
        uc, un = model(torch.cat([xf, xf]), torch.cat([t, t]), cat_kv(kv, kv_null), **fk).float().chunk(2)
        v = un + cfg * (uc - un)
    v = v.unflatten(1, (n, T))
    return x.float() + (1 - t)[..., None, None] * v, x.float() + (end - t)[..., None, None] * v


def roll_init(p, n, alpha):
    """新 episode：一整段干净计划 p (B, n·T, D)（scratch 采样）→ 按前向核加噪到决策末的滚动剖面（第 j 槽 τ = shift(1 − j/n)）。"""
    t = shift(1 - torch.arange(n, device=p.device, dtype=torch.float) / n, alpha).repeat_interleave(p.shape[1] // n)[None, :, None]
    return t * p.float() + (1 - t) * torch.randn_like(p.float())


@torch.no_grad()
def rollout(model, ctx, txt, tmask, H, mode, obj=None, hcfg=None, heat=None, kf=None, kfcfg=None, demo=None, mem=None, **kw):
    """自回归推进 H 帧（模型一次生成 n_fut 帧）。ctx (B,K,T,D) → preds (B,H,T,D)，每帧平均 NFE。
    obj：物体 token 输入（model.WM n_obj > 0；整段推演都用当前观测的这一份）。
    hcfg=(w_h, w_t)：历史 / 文本两路分开的引导 u = u_∅∅ + w_h (u_h∅ − u_∅∅) + w_t (u_ht − u_h∅)（需 train --p_hdrop 的模型；
    用 sample 的 comp：基 = 两路都丢，条件 = [只有历史, 历史 + 文本]，权重 [w_h − w_t, w_t]）；此时忽略 cfg。"""
    out, nfe, K, n = [], 0, ctx.shape[1], getattr(model, "n_fut", 1)
    ok = {} if obj is None else {"obj": obj}
    if heat is not None:  # where 热图（WM --where）：条件 / 无条件两路都带
        ok["heat"] = heat
    if kf is not None:  # 关键帧条件（WM --kfcond）：同上
        ok["kf"] = kf
    if demo is not None:  # (mika) 示范 / 全历史 token（WM n_dfr > 0；train --demo_bank / --self_hist）：条件 / 无条件两路都带（CFG 只丢语言）
        ok["demo"] = demo
    if mem is not None:  # (mika C2) 记忆 token（WM n_mem > 0；train --mem）：条件 / 无条件两路都带
        ok["mem"] = mem
    null = torch.ones(len(ctx), dtype=torch.bool, device=ctx.device)
    m, sig = kw.pop("smooth", (1, 0.0))   # 平滑引导：每个样本 m 份文本嵌入，各加相对 RMS 为 sig 的噪声（同训练的 --txt_noise）
    if m > 1:
        tm_ = tmask.repeat_interleave(m, 0)
        t_ = txt.float().repeat_interleave(m, 0)
        txt_m = t_ + sig * t_.pow(2).mean(-1, keepdim=True).sqrt() * torch.randn_like(t_) * tm_[..., None]
    while len(out) < H:
        kv = model.context(ctx, txt, tmask, **ok) if m == 1 else model.context(ctx.repeat_interleave(m, 0), txt_m, tm_)
        kvn = model.context(ctx, txt, tmask, drop=null, **ok) if kw.get("cfg", 1.0) != 1.0 or hcfg else None
        if hcfg:
            kw = {**kw, "cfg": 1.0, "comp": ([kvn, kv], torch.tensor([[hcfg[0] - hcfg[1], hcfg[1]]], device=ctx.device).expand(len(ctx), 2))}
            kvn = model.context(ctx, txt, tmask, drop=null, hdrop=null, **ok)
        if kfcfg and kf is not None:  # 关键帧引导（WM --kfcond，训练时丢过关键帧）：u = u_b + w_t (u_tk − u_b) + (w_k − 1)(u_tk − u_t0)，
            # u_b = 丢语言、保留关键帧，u_t0 = 有语言、关键帧置 0；用 sample 的 comp：条件 [u_tk, u_t0]，权重 [w_t + w_k − 1, −(w_k − 1)]
            kv0 = model.context(ctx, txt, tmask, **{k: v for k, v in ok.items() if k != "kf"})
            wt = kw.get("cfg", 1.0)
            kw = {**kw, "cfg": 1.0, "comp": ([kv, kv0], torch.tensor([[wt + kfcfg - 1, -(kfcfg - 1)]], device=ctx.device).expand(len(ctx), 2))}
            kvn = kvn if kvn is not None else model.context(ctx, txt, tmask, drop=null, **ok)
        Tm = model.pos.shape[1]  # 预测的 token 数；ctx 每帧多出的 token 是只作上下文的额外视角（train --tgt_tok）
        z0 = cvbase(ctx[:, -1, :Tm], ctx[:, -2, :Tm], n) if mode == "cv" else ctx[:, -1, :Tm].repeat(1, n, 1)
        z, k = sample(model, kv, z0, mode, kv_null=kvn, n_sm=m, **kw)
        z = z.unflatten(1, (n, -1))
        out.extend(z.unbind(1))
        nfe += k
        if ctx.shape[2] > Tm:  # 预测帧没有额外视角：沿用最后一个观测帧的
            z = torch.cat([z, ctx[:, -1:, Tm:].expand(-1, n, -1, -1).to(z.dtype)], 2)
        ctx = torch.cat([ctx, z.to(ctx.dtype)], 1)[:, -K:]
    return torch.stack(out[:H], 1), nfe / len(out)
