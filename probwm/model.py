import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, sh, sc):  # sh / sc：(B,d) 每个样本一份，或 (B,N,d) 每个 token 一份（逐槽噪声层级）
    return x * (1 + (sc if sc.dim() == 3 else sc[:, None])) + (sh if sh.dim() == 3 else sh[:, None])


def tsin(t, dim=256):
    f = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    a = t[:, None].float() * 1000 * f
    return torch.cat([a.cos(), a.sin()], -1)


def rope_table(n_tok, dh, n_rep, extra, dev, base=100.0):
    """2D RoPE 表 → (cos, sin) 各 (N, dh/2)：前 n_rep 份 √n_tok×√n_tok 网格 token（旋转维的一半按行、一半按列），之后 extra 个非空间 token
    （语言 / 物体 / 本体感觉：旋转维置 0 → 它们与 patch 的注意力分数与位置无关）。每个头只旋转前 dh/2 维，后一半不带位置。"""
    g, nr = int(round(n_tok ** 0.5)), dh // 2
    f = base ** (-torch.arange(nr // 4, device=dev, dtype=torch.float) / (nr // 4))
    r, c = torch.arange(g, device=dev).repeat_interleave(g).float(), torch.arange(g, device=dev).repeat(g).float()
    a = torch.cat([r[:, None] * f, c[:, None] * f], 1)                   # (T, nr/2)
    a = torch.cat([a, a], 1).repeat(n_rep, 1)                            # rotate_half 约定：(T·n_rep, nr)
    z = torch.zeros(extra, nr, device=dev)
    return torch.cat([a.cos(), z]), torch.cat([a.sin(), z])


def rope(x, pe):  # x (B,h,N,dh)，pe = (cos, sin) (N, dh/2)：前 dh/2 维旋转（非空间 token 置 0），其余不变
    cos, sin = pe
    nr = cos.shape[-1]
    xr, xp = x[..., :nr], x[..., nr:]
    x1, x2 = xr.chunk(2, -1)
    return torch.cat([xr * cos.to(x.dtype) + torch.cat([-x2, x1], -1) * sin.to(x.dtype), xp], -1)


class Attn(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h
        s.q, s.kv, s.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        s.qn, s.kn = nn.RMSNorm(d // h), nn.RMSNorm(d // h)  # QK norm

    def kvs(s, m, pe=None):
        B, M, _ = m.shape
        k, v = s.kv(m).view(B, M, 2, s.h, -1).permute(2, 0, 3, 1, 4)
        k = s.kn(k)
        return (k if pe is None else rope(k, pe)), v

    def forward(s, x, kv=None, pe=None, mask=None):  # pe：2D RoPE 表（rope_table）；自注意力时 q、k 都转，交叉注意力时只转 q（kv 已按上下文的表转好）；mask (N,N) bool：True = 可见
        B, N, d = x.shape
        q = s.qn(s.q(x).view(B, N, s.h, -1).transpose(1, 2))
        q = q if pe is None else rope(q, pe)
        k, v = s.kvs(x, pe) if kv is None else kv
        if isinstance(mask, int):  # (gawm) 整数 n：前 n 个 query 只看前 n 个 key，其余 query 看全部（= 布尔单向 mask；拆成两次无 mask 的 SDPA，可走 flash）
            o = torch.cat([F.scaled_dot_product_attention(q[:, :, :mask], k[:, :, :mask], v[:, :, :mask]),
                           F.scaled_dot_product_attention(q[:, :, mask:], k, v)], 2)
            return s.o(o.transpose(1, 2).reshape(B, N, d))
        return s.o(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, N, d))


def mlp(d, w):
    return nn.Sequential(nn.Linear(d, w), nn.GELU(approximate="tanh"), nn.Linear(w, d))


class _AuxGrad(torch.autograd.Function):  # (cap) 前向返回 x 的拷贝；反向给 aux 一个单位梯度 → 负载均衡损失的梯度经它流进路由器（不用另行汇总损失）
    @staticmethod
    def forward(ctx, x, aux):
        ctx.dt = aux.dtype
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return g, torch.ones((), dtype=ctx.dt, device=g.device)


class MoE(nn.Module):  # (cap) train --moe E,k,m[,cf]：目标流 FFN 换成 E 个专家（各 d → m·d → d）、逐 token 选 top-k（门控 = top-k 概率归一，Mixtral 式）。
    # 训练：GShard 式静态容量 C = ⌈cf·N·k/E⌉（第一选择优先占位，超出的分派丢掉、只剩残差），专家用 baddbmm 一次算完，形状固定 → 可进 torch.compile、无同步；
    #   Switch 式负载均衡损失 E·Σ f_e·P_e × lb 经 _AuxGrad 注入。推理（eval 模式）：逐专家变长分派、不丢 token（不进 compile）。
    def __init__(s, d, E, k, m, cf, lb, dl=0, shared=0, bias_g=0.0):
        super().__init__()
        s.E, s.k, s.cf, s.lb, h = E, k, cf, lb, int(m * d)
        s.router = nn.Linear(d, E, bias=False)
        nn.init.normal_(s.router.weight, std=0.02)
        b1, b2 = 1 / math.sqrt(d), 1 / math.sqrt(h)  # 同 nn.Linear 的默认初始化
        s.w1, s.b1 = nn.Parameter(torch.empty(E, d, h).uniform_(-b1, b1)), nn.Parameter(torch.empty(E, h).uniform_(-b1, b1))
        s.w2, s.b2 = nn.Parameter(torch.empty(E, h, d).uniform_(-b2, b2)), nn.Parameter(torch.empty(E, d).uniform_(-b2, b2))
        # (cap MoE-T，train --moe_lang / --moe_shared / --moe_bias；都不开 = 原行为)：
        #   dl > 0：路由器另吃池化的语言嵌入（context 里按 mask 取平均、被丢的语言为 0；LayerNorm 后线性映射到 E，零初始化 → 起点同纯 token 路由；等价于把语言嵌入拼进路由器输入）；
        #   shared：加一个所有 token 都走的共享专家（宽同专家，输出直接相加）；
        #   bias_g > 0：无辅助损失的负载均衡（DeepSeek-V3）——选专家按 p + 偏置、门控仍用 p（选中的归一）；训练时累计各专家被选次数，每步优化后 bias_step 按负载 ±bias_g 调偏置（不进梯度）。
        s.dl, s.bias_g = dl, bias_g
        if dl:
            s.router_l = nn.Linear(dl, E, bias=False)
            nn.init.zeros_(s.router_l.weight)
        s.shared = mlp(d, h) if shared else None
        if bias_g > 0:
            s.register_buffer("ebias", torch.zeros(E))
            s.register_buffer("eload", torch.zeros(E))

    def route(s, x, lt=None):
        if lt is None and s.bias_g <= 0:
            p = s.router(x).float().softmax(-1)              # (N, E)
            g, ei = p.topk(s.k, -1)
            return p, g / g.sum(-1, keepdim=True), ei
        lg = s.router(x).float()
        if lt is not None:
            lg = lg + s.router_l(lt).float()
        p = lg.softmax(-1)
        if s.bias_g > 0:  # 偏置只影响选谁，不影响门控值
            ei = (p + s.ebias).topk(s.k, -1).indices
            g = p.gather(-1, ei)
        else:
            g, ei = p.topk(s.k, -1)
        return p, g / g.sum(-1, keepdim=True), ei

    def forward(s, x, lang=None):  # lang (B, dl)：(MoE-T) 池化的语言嵌入；None = 原行为
        sh = x.shape
        x = x.reshape(-1, sh[-1])
        lt = None
        if lang is not None and s.dl:
            lt = F.layer_norm(lang.float(), (s.dl,))[:, None].expand(sh[0], sh[1], -1).reshape(-1, s.dl).to(x.dtype)
        y = s.fwd_cap(x, lt) if s.training else s.fwd_exact(x, lt)
        if s.shared is not None:
            y = y + s.shared(x)
        return y.reshape(sh)

    @torch.no_grad()
    def bias_step(s):  # (MoE-T) 每步优化后：按这一步累计的各专家负载调偏置（负载低于均值的 +bias_g、高于的 −bias_g），清零计数
        if s.bias_g <= 0 or float(s.eload.sum()) == 0:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(s.eload)
        s.ebias += s.bias_g * torch.sign(s.eload.mean() - s.eload)
        s.eload.zero_()

    def fwd_cap(s, x, lt=None):
        N, d = x.shape
        p, g, ei = s.route(x, lt)
        C = int(math.ceil(s.cf * N * s.k / s.E))
        ef = ei.t().reshape(-1)                          # (kN,)：按选择序排（先放所有 token 的第一选择）
        oh = F.one_hot(ef, s.E)
        if s.bias_g > 0:
            s.eload.add_(oh.sum(0).to(s.eload.dtype))    # (MoE-T) 本步各专家被选次数（buffer，不进梯度）
        pos = ((oh.cumsum(0) - 1) * oh).sum(-1)          # 在所选专家里的位次
        keep = pos < C
        slot = torch.where(keep, ef * C + pos, torch.full_like(pos, s.E * C))  # 丢掉的分派写到最后一行哑槽
        tok = torch.arange(N, device=x.device).repeat(s.k)
        buf = x.new_zeros(s.E * C + 1, d).index_copy(0, slot, x[tok])[:-1].view(s.E, C, d)
        y = torch.baddbmm(s.b2[:, None], F.gelu(torch.baddbmm(s.b1[:, None], buf, s.w1), approximate="tanh"), s.w2)
        y = torch.cat([y.reshape(s.E * C, d), y.new_zeros(1, d)])
        gf = g.t().reshape(-1) * keep
        if s.lb > 0:
            gf = _AuxGrad.apply(gf, s.lb * s.E * (oh.float().mean(0) * p.mean(0)).sum())
        return y.new_zeros(N, d).index_add(0, tok, y[slot] * gf[:, None].to(y.dtype))

    @torch.compiler.disable
    def fwd_exact(s, x, lt=None):
        p, g, ei = s.route(x, lt)
        g, flat = g.to(x.dtype), ei.flatten()
        order = flat.argsort()
        cnt = torch.bincount(flat, minlength=s.E)
        tok, gs = order // s.k, g.flatten()[order]
        xs, ys, o = x[tok], [], 0
        for e, c in enumerate(cnt.tolist()):
            if c:
                ys.append(F.gelu(xs[o:o + c] @ s.w1[e] + s.b1[e], approximate="tanh") @ s.w2[e] + s.b2[e])
            o += c
        return torch.zeros_like(x).index_add_(0, tok, torch.cat(ys) * gs[:, None])

    @torch.no_grad()
    def load(s, x):  # 诊断：各专家分到的 token 占比、按容量会丢掉的分派比例
        x = x.reshape(-1, x.shape[-1])
        p, g, ei = s.route(x)
        ef = ei.t().reshape(-1)
        oh = F.one_hot(ef, s.E)
        pos = ((oh.cumsum(0) - 1) * oh).sum(-1)
        return oh.float().mean(0), float((pos >= math.ceil(s.cf * len(x) * s.k / s.E)).float().mean())


class CBlock(nn.Module):  # 上下文编码层：每次推进只算一次
    def __init__(s, d, h):
        super().__init__()
        s.n1, s.sa, s.n2, s.mlp = nn.LayerNorm(d), Attn(d, h), nn.LayerNorm(d), mlp(d, 4 * d)

    def forward(s, x, pe=None, kb=None):  # kb：(B,h,1,L) key 的加性偏置（train --sh_alibi）；None = 原行为
        x = x + s.sa(s.n1(x), pe=pe, mask=kb)
        return x + s.mlp(s.n2(x))


class Block(nn.Module):  # 目标流：AdaLN(τ) 自注意力 + 交叉注意力（KV 缓存）+ MLP
    def __init__(s, d, h, moe=None):  # moe = (E, k, m, lb)：(cap) train --moe，FFN 换成 MoE；None = 原行为
        super().__init__()
        s.n1, s.n3 = nn.LayerNorm(d, elementwise_affine=False), nn.LayerNorm(d, elementwise_affine=False)
        s.n2, s.sa, s.ca, s.mlp = nn.LayerNorm(d), Attn(d, h), Attn(d, h), (mlp(d, 4 * d) if moe is None else MoE(d, *moe))
        s.ada = nn.Linear(d, 6 * d)

    def forward(s, x, c, kv, pe=None, mask=None, kv_aux=None, n_main=None, kb=None, lang=None):  # kb：交叉注意力 key 的加性偏置 (B,h,1,L)（train --sh_alibi）；lang：(MoE-T) 池化语言嵌入，传给 MoE 路由
        """mask：目标 token 自注意力的可见性（训练专用辅助槽：主 chunk token 看不到辅助槽）；kv_aux / n_main：前 n_main 个 token 用 kv，其余（辅助槽）用 kv_aux 做交叉注意力。"""
        a, b, g1, e, f, g2 = s.ada(c).chunk(6, -1)
        x = x + (g1 if g1.dim() == 3 else g1[:, None]) * s.sa(modulate(s.n1(x), a, b), pe=pe, mask=mask)
        if kv_aux is None:
            x = x + s.ca(s.n2(x), kv, pe=pe, mask=kb)
        else:
            y = s.n2(x)
            pm, pa = (None, None) if pe is None else ((pe[0][:n_main], pe[1][:n_main]), (pe[0][n_main:], pe[1][n_main:]))
            x = x + torch.cat([s.ca(y[:, :n_main], kv, pe=pm), s.ca(y[:, n_main:], kv_aux, pe=pa)], 1)
        if lang is not None:
            return x + (g2 if g2.dim() == 3 else g2[:, None]) * s.mlp(modulate(s.n3(x), e, f), lang)
        return x + (g2 if g2.dim() == 3 else g2[:, None]) * s.mlp(modulate(s.n3(x), e, f))


class KV(list):  # 每层交叉注意力的 (k,v) 列表；.heat = "where" 热图（WM --where），目标流在 forward 里读它
    heat = None
    lang = None  # (cap MoE-T) 池化的语言嵌入 (B, Dt)，目标流的 MoE 路由用；None = 原行为


class STN(nn.Module):
    """(pro) 几何 canon（train --canon_stn）：从上下文帧的 g×g patch 特征回归 2D 仿射（6）/ 单应（8）参数（相对恒等的偏移），
    用 grid_sample 把特征网格对齐到标准视角。最后一层置零 = 恒等初始化；输出写成 x + (warp(x, θ) − warp(x, θ_I))，恒等时与输入逐位相同。
    pen()：最近一次前向的 ‖θ − θ_I‖²（train --stn_reg 加进损失）。"""

    def __init__(s, D, kind="homog", g=16, w=64):
        super().__init__()
        s.kind, s.g, s.theta = kind, g, None
        s.net = nn.Sequential(nn.Conv2d(D, w, 1), nn.GELU(), nn.Conv2d(w, w, 3, 2, 1), nn.GELU(), nn.Conv2d(w, w, 3, 2, 1), nn.GELU(),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(w, 8 if kind == "homog" else 6))
        nn.init.zeros_(s.net[-1].weight)
        nn.init.zeros_(s.net[-1].bias)

    def grid(s, d):  # d (B, 6|8) 相对恒等的偏移 → grid_sample 的采样网格 (B, g, g, 2)（归一化坐标，像素中心，align_corners=False）
        B, g = len(d), s.g
        c = (torch.arange(g, device=d.device, dtype=torch.float32) * 2 + 1) / g - 1
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        pts = torch.stack([xx, yy, torch.ones_like(xx)], -1).view(1, g * g, 3)
        M = torch.cat([d, torch.zeros(B, 9 - d.shape[1], device=d.device)], 1).view(B, 3, 3) + torch.eye(3, device=d.device)
        p = pts @ M.transpose(1, 2)
        xy = p[..., :2] if s.kind == "affine" else p[..., :2] / p[..., 2:].clamp_min(0.1)
        return xy.view(B, g, g, 2)

    def forward(s, x):  # x (B, g·g, D) → 对齐后的 (B, g·g, D)
        B, T, D = x.shape
        f = x.float().transpose(1, 2).reshape(B, D, s.g, s.g)
        d = s.net(f)
        s.theta = d
        y = F.grid_sample(f, s.grid(d), mode="bilinear", padding_mode="zeros", align_corners=False)
        y0 = F.grid_sample(f, s.grid(torch.zeros_like(d)), mode="bilinear", padding_mode="zeros", align_corners=False)
        return x + (y - y0).reshape(B, D, T).transpose(1, 2).to(x.dtype)

    def pen(s):
        return s.theta.float().pow(2).sum(1).mean()


class WM(nn.Module):
    def __init__(s, D, n_tok, K, d_txt, d=512, depth=8, heads=8, ctx_depth=2, head_w=1024, n_fut=1, warm=False, n_obj=0, d_obj=0, n_aux=0,
                 rope=False, where=False, n_xtok=0, kfc=False, n_auxs=0, kfa=False, n_dfr=0, n_dtok=0, n_hz=0, alibi=False, n_mem=0, n_ae=0, moe=None,
                 stn="", t0h=0):
        super().__init__()
        s.n_mem = n_mem
        s.stn = STN(D, stn, int(round(n_tok ** 0.5))) if stn else None  # (pro) 几何 canon：上下文帧先过空间变换头（默认关）
        if n_mem:  # (mika C2) 记忆 token：S 个槽（WM 宽度 d），接在上下文最后进上下文编码器；输出在记忆位置上的状态经门控即新记忆（context 返回的 kv.mem）。
            s.mem0 = nn.Parameter(torch.randn(1, n_mem, d) * 0.02)     # episode 开头的记忆（可学）；不带任何时间嵌入，只有槽位嵌入
            s.mpos = nn.Parameter(torch.randn(1, n_mem, d) * 0.02)
            s.mnorm, s.mgate = nn.LayerNorm(d), nn.Linear(2 * d, d)
            nn.init.zeros_(s.mgate.weight), nn.init.zeros_(s.mgate.bias)
        s.n_dfr, s.n_dtok = n_dfr, n_dtok
        if alibi:  # (mika) train --sh_alibi：历史（示范）token 的 key 偏置 −m_h·Δt；前 2 个头 m = 0（长程检索），其余按几何级数 1/512 … （每帧）
            m = [0.0, 0.0] + [2.0 ** -(9 - i * 5 / max(heads - 3, 1)) for i in range(heads - 2)]
            s.register_buffer("alibi_m", torch.tensor(m[:heads]), persistent=False)
        s.alibi = alibi
        s.n_hz = n_hz  # (hz2) 任意时距远槽（train --any_far）：目标槽的时间槽嵌入按整数时距 1..n_hz 查表（hz_pos），不再按槽序；目标槽数可变
        if n_dfr:  # (demo) 示范视频条件（train --demo_bank）：同任务另一条 demo 的 n_dfr 帧（进度 0..1 均匀），每帧池化成 n_dtok 个 token，接在语言 token 之后作上下文；
            s.dpos = nn.Parameter(torch.randn(1, 1, n_dtok, d) * 0.02)  # 输入投影与历史帧共用 c_in；丢掉的帧 / 整段示范换成 dnull（形状不变，SDPA 走 flash）
            s.dprog = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
            s.dtag, s.dnull = nn.Parameter(torch.zeros(1, 1, 1, d)), nn.Parameter(torch.zeros(1, 1, 1, d))
        s.n_auxs = n_auxs  # 训练专用的辅助目标槽（train --aux_offs / --aux_kf）：远时距帧 / 事件关键帧，与 chunk 一起去噪，但 chunk token 看不到它们 → 推理时去掉不影响 chunk
        s.K, s.n_fut, s.warm, s.n_obj, s.rope, s.dh, s.where, s.kfc = K, n_fut, warm, n_obj, rope, d // heads, where, kfc
        if kfc:  # 关键帧条件（train --kfcond）：下一个接触关键帧的特征作为一组额外的上下文 token（自己的时间槽嵌入），接在历史帧后面
            s.kslot = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        s.kfa = kfa
        if kfa:  # 条件增广（train --kf_aug）：关键帧条件的噪声层级 τ_c 也作为输入（零初始化），kf 以 (特征, τ_c) 二元组传进来
            s.ktemb = nn.Linear(256, d)
            nn.init.zeros_(s.ktemb.weight), nn.init.zeros_(s.ktemb.bias)
        if where:  # "where" 瓶颈（train --where）：下一个接触点的 16×16 热图逐 patch 加到历史帧和带噪目标帧的 token 上（零初始化）
            s.wh = nn.Linear(1, d)
            nn.init.zeros_(s.wh.weight), nn.init.zeros_(s.wh.bias)
        if n_aux:  # 辅助头（train --aux_eef）：上下文 token 平均 → chunk 内各帧的末端位移（3D，相对当前帧）
            s.aux = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(approximate="tanh"), nn.Linear(d, n_aux))
        if n_obj:  # 物体 token（probwm/objtok.py）：每个槽位 (池化特征 ⊕ 位置特征) → d，加槽位嵌入，接在上下文后面
            s.o_in, s.oslot = nn.Linear(d_obj, d), nn.Parameter(torch.randn(1, n_obj, d) * 0.02)
        s.x_in, s.c_in, s.t_in = nn.Linear(D, d), nn.Linear(D, d), nn.Linear(d_txt, d)
        if rope:  # 2D RoPE（train --rope）：不要绝对空间位置嵌入，patch 之间只有相对位置（自注意力 / 交叉注意力里转 q、k）
            s.register_buffer("pos", torch.zeros(1, n_tok, d), persistent=False)
        else:
            s.pos = nn.Parameter(torch.randn(1, n_tok, d) * 0.02)       # 空间位置
        s.fpos = nn.Parameter(torch.randn(1, K + (n_hz or n_fut) + n_auxs, 1, d) * 0.02)   # 时间槽：K 个历史 + n_fut 个目标（+ 训练专用的辅助槽）；n_hz > 0 时目标部分为时距 1..n_hz 的表
        if n_xtok:  # 只作上下文的额外视角 token（train --tgt_tok）：各自的可学位置嵌入（= 视角 + 空间），不进预测目标
            s.xpos = nn.Parameter(torch.randn(1, n_xtok, d) * 0.02)
        s.ttag, s.null = nn.Parameter(torch.zeros(1, 1, d)), nn.Parameter(torch.zeros(1, 1, d))
        s.temb = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        if warm:  # 训练时模拟热启动：起点层级 τ0 的嵌入（τ0=0 即标准 FM）
            s.wemb = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
            nn.init.zeros_(s.wemb[-1].weight), nn.init.zeros_(s.wemb[-1].bias)
        s.cblocks = nn.ModuleList(CBlock(d, heads) for _ in range(ctx_depth))
        if moe is not None and len(moe) > 5:  # (cap MoE-T) moe = (E, k, m, cf, lb, 语言路由 0/1, 共享专家 0/1, 偏置步长)：语言路由时 dl = 文本维
            s.moe_lang = bool(moe[5])
            moe = tuple(moe[:5]) + (d_txt if moe[5] else 0,) + tuple(moe[6:])
        s.blocks = nn.ModuleList((Block(d, heads) if moe is None else Block(d, heads, moe)) for _ in range(depth))  # moe：(cap) train --moe
        s.hn, s.hada = nn.LayerNorm(d, elementwise_affine=False), nn.Linear(d, 2 * d)
        s.head = nn.Sequential(nn.Linear(d, head_w), nn.GELU(approximate="tanh"), nn.Linear(head_w, D))  # 浅而宽
        for m in [*(b.ada for b in s.blocks), s.hada, s.head[-1]]:
            nn.init.zeros_(m.weight), nn.init.zeros_(m.bias)
        if n_ae:  # (cap) train --aux_ae：远槽隐状态（槽内 token 平均）→ 该槽时刻的末端位移（n_ae = 3 × 臂数）；只在训练损失里用，推理不读。最后建，其余参数的初始化与不开时相同
            s.ae = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(approximate="tanh"), nn.Linear(d, n_ae))
        if t0h:  # (pro 11-17 X3cSG) train --t0_head：独立的 t+0 小头——读上下文编码器输出里当前帧的 token（stop-gradient），回归同状态的干净当前帧；
            # 梯度不回传 WM 主体。最后建，其余参数的初始化与不开时相同；推理只在需要 t+0 时（context(t0=True)）算
            s.t0h = nn.Sequential(nn.LayerNorm(d), *[nn.TransformerEncoderLayer(d, heads, 2 * d, 0.0, "gelu", batch_first=True, norm_first=True) for _ in range(t0h)],
                                  nn.LayerNorm(d), nn.Linear(d, D))

    def context(s, ctx, txt, tmask, drop=None, obj=None, hdrop=None, aux=False, heat=None, kf=None, demo=None, mem=None, mem_only=False, t0=False):
        # mem (B,S,d)：(mika C2) 上一步的记忆，None = 开头的记忆；mem_only：只要新记忆（训练预热用），不算各层交叉注意力的 KV
        """ctx (B,K,T,D)，txt (B,L,Dt)，tmask (B,L)，drop (B,) → 每层交叉注意力的 (k,v)。
        padding 和被丢掉的语言都替换成 null token，这样不需要 mask，SDPA 可以走 flash。
        obj (B,n_obj,d_obj)：物体 token 的输入（objtok.tokens）；n_obj > 0 的模型不给时按全部不可见（全 0）处理。CFG 只丢语言，物体 token 保留。
        aux=True：另返回辅助头的输出（(kvs, aux)）。heat (B,T)：where 热图（WM --where；不给 = 全 0），也随 kvs 传给目标流。
        hdrop (B,) bool：丢掉视觉上下文（历史帧置 0 = 标准化后的均值帧，物体 token 置为不可见），用于历史 / 文本两路分开的引导（train --p_hdrop，fm.rollout hcfg）。"""
        B, K, T, _ = ctx.shape
        if getattr(s, "stn", None) is not None:  # (pro) 几何 canon：每个上下文帧的 patch 网格先对齐到标准视角
            assert T == s.stn.g ** 2, ("--canon_stn 只接单视角 g×g 的上下文", T)
            ctx = s.stn(ctx.flatten(0, 1)).view(B, K, T, -1)
        if hdrop is not None:
            ctx = torch.where(hdrop[:, None, None, None], torch.zeros_like(ctx), ctx)
            obj = None if obj is None else torch.where(hdrop[:, None, None], torch.zeros_like(obj), obj)
        w, hx = s.c_in.weight.dtype, None
        if T > s.pos.shape[1]:  # 每帧前 n_tok 个 token 是预测视角，其余是只作上下文的额外视角（接在历史帧 token 之后）
            T = s.pos.shape[1]
            xl = getattr(s, "xlast", False)  # (pro) train --xtok_last：额外视角只取当前帧（最后一个历史槽 K−1）
            hx = ((s.cx_in if getattr(s, "cx_in", None) is not None else s.c_in)((ctx[:, -1:, T:] if xl else ctx[:, :, T:]).to(w)) + s.xpos[:, None]
                  + (s.fpos[:, K - 1:K] if xl else s.fpos[:, :K])).flatten(1, 2)  # cx_in：(rteasy) train --xin_zero
            ctx = ctx[:, :, :T]
        h = s.c_in(ctx.to(w)) + s.pos[:, None] + s.fpos[:, :K]
        if s.where:
            heat = torch.zeros(B, T, device=ctx.device) if heat is None else heat
            h = h + s.wh(heat.to(w)[:, None, :, None])
        h = h.reshape(B, K * T, -1)
        if s.kfc:  # kf (B,T,D)：关键帧特征（不给 = 全 0，即训练时的"丢掉"）
            kf, ktau = kf if isinstance(kf, tuple) else (kf, None)
            kt = s.c_in((torch.zeros_like(ctx[:, 0]) if kf is None else kf).to(w)) + s.pos + s.kslot
            if getattr(s, "kfa", False):
                kt = kt + s.ktemb(tsin(torch.zeros(B, device=ctx.device) if ktau is None else ktau).to(w))[:, None]
            h = torch.cat([h, kt.to(h.dtype)], 1)
        h = h if hx is None else torch.cat([h, hx.to(h.dtype)], 1)
        keep = tmask if drop is None else tmask & ~drop[:, None]
        t = torch.where(keep[..., None], s.t_in(txt.to(w)) + s.ttag, s.null.to(h.dtype))
        m = torch.cat([h, t.to(h.dtype)], 1)
        if getattr(s, "n_dfr", 0):  # (demo) demo = (特征 (B, n_dfr, n_dtok, D) 已标准化, keep (B, n_dfr) bool[, 进度 (B, n_dfr)])；None = 没有示范（全 dnull）
            M, P = s.n_dfr, s.n_dtok
            if demo is None:
                dt = s.dnull.to(h.dtype).expand(B, M, P, -1)
            else:
                f, keep = demo[0], demo[1]
                pg = demo[2] if len(demo) > 2 else torch.linspace(0, 1, M, device=ctx.device)[None].expand(B, -1)
                dt = s.c_in(f.to(w)) + s.dpos + s.dprog(tsin(pg.flatten()).to(w)).view(B, M, 1, -1) + s.dtag
                dt = torch.where(keep[:, :, None, None], dt, s.dnull.to(dt.dtype))
            m = torch.cat([m, dt.reshape(B, M * P, -1).to(m.dtype)], 1)
        kb = None
        if getattr(s, "alibi", False) and demo is not None and len(demo) > 3:  # (mika) --sh_alibi：历史 token 位置的 key 偏置 −m_h·Δt（demo[3] = Δt 帧数），其余位置 0
            M_, P_ = s.n_dfr, s.n_dtok
            kd = -(s.alibi_m[None, :, None] * demo[3].float()[:, None, :]).repeat_interleave(P_, -1)  # (B, h, M·P)
            kb = torch.cat([torch.zeros(B, len(s.alibi_m), m.shape[1] - M_ * P_, device=m.device), kd], -1)
        if s.n_obj:
            o = s.o_in(obj.to(w)) if obj is not None else s.o_in.bias.expand(B, s.n_obj, -1)
            m = torch.cat([m, (o + s.oslot).to(m.dtype)], 1)
            kb = None if kb is None else torch.cat([kb, torch.zeros(B, kb.shape[1], s.n_obj, device=m.device)], -1)
        kb = None if kb is None else kb[:, :, None].to(m.dtype)
        if getattr(s, "n_mem", 0):  # (mika C2) 记忆 token 接在最后
            mp = s.mem0.expand(B, -1, -1) if mem is None else mem
            m = torch.cat([m, (mp + s.mpos).to(m.dtype)], 1)
            kb = None if kb is None else torch.cat([kb, torch.zeros(*kb.shape[:3], s.n_mem, device=m.device, dtype=kb.dtype)], -1)
        nsp = K + int(s.kfc)  # 带网格位置的帧数（历史 + 关键帧）
        pe = rope_table(T, s.dh, nsp, m.shape[1] - nsp * T, m.device) if s.rope else None
        for b in s.cblocks:
            m = b(m, pe) if kb is None else b(m, pe, kb)
        kv = KV() if mem_only else KV(b.ca.kvs(m, pe) for b in s.blocks)
        kv.heat = heat if s.where else None
        if t0 and getattr(s, "t0h", None) is not None:  # (pro X3cSG) t+0 小头：当前帧（第 K 个历史槽）的上下文 token，detach → 不回传 WM 主体
            kv.t0 = s.t0h(m[:, (K - 1) * T:K * T].detach())
        if getattr(s, "n_mem", 0):  # (mika C2) 新记忆 = 门控（候选 = 记忆位置的输出过 LayerNorm）；读写共用这一次前向
            mp = (s.mem0.expand(B, -1, -1) if mem is None else mem).float()
            cand = s.mnorm(m[:, -s.n_mem:].float())
            g = torch.sigmoid(s.mgate(torch.cat([cand, mp], -1).to(s.mgate.weight.dtype))).float()
            kv.mem = g * cand + (1 - g) * mp
        kv.kbias = kb  # (mika) --sh_alibi：目标流交叉注意力也用同一份 key 偏置；None = 原行为
        if getattr(s, "moe_lang", False):  # (cap MoE-T) 池化的语言嵌入（按 keep 平均，CFG 丢掉 / 全 padding 时为 0）
            kf_ = keep[..., None].float()
            kv.lang = (txt.float() * kf_).sum(1) / kf_.sum(1).clamp_min(1)
        return (kv, s.aux(m.mean(1))) if aux else kv

    def hz_pos(s, hz):  # (hz2) n_hz > 0：各目标槽的整数时距 hz (B, ns) → 时间槽嵌入 (B, ns, 1, d)，传给 trunk 的 fpos
        return s.fpos[0, s.K + hz.long() - 1]

    def forward(s, x, tau, kvs, tau0=None, fpos=None, ret_h=False):
        if ret_h:  # (cap) --aux_ae：另返回输出头之前的逐 token 特征
            h = s.trunk(x, tau, kvs, tau0, fpos)
            return s.head(h), h
        return s.head(s.trunk(x, tau, kvs, tau0, fpos))

    def trunk(s, x, tau, kvs, tau0=None, fpos=None):  # 输出头之前（调制后）的逐 token 特征；warp-kf（probwm/warpkf.py）在它上面另接形变头
        B, T = len(x), s.pos.shape[1]   # x: (B, n_fut*T, D)；训练时可带辅助槽 (B, (n_fut+n_auxs)*T, D)
        ns = x.shape[1] // T            # fpos (B, ns, 1, d)：替换目标槽的时间槽嵌入（闭环 --abs_far：按真实时距插值的远槽，见 closed_loop.Policy.rolling_x）；默认 None = 原样
        if fpos is None and getattr(s, "fq", None) is not None:  # (hz2) 任意时距远槽 WM 的推理：闭环设的查询时距嵌入 (1, ns, 1, d)（closed_loop --any_far）
            fpos = s.fq
        h = s.x_in(x.to(s.x_in.weight.dtype)).view(B, ns, T, -1) + s.pos[:, None] + (s.fpos[:, s.K:s.K + ns] if fpos is None else fpos.to(s.fpos.dtype))
        if s.where:
            hm = getattr(kvs, "heat", None)
            h = h + s.wh((torch.zeros(B, T, device=x.device) if hm is None else hm).to(h.dtype)[:, None, :, None])
        h = h.flatten(1, 2)
        if tau.dim() == 2:  # 逐槽噪声层级 (B, ns)（diffusion forcing 式训练 / 交错去噪）：每个 token 用自己槽的 τ 嵌入
            c = s.temb(tsin(tau.flatten()).to(h.dtype)).view(B, ns, 1, -1).expand(B, ns, T, -1).reshape(B, ns * T, -1)
        else:
            c = s.temb(tsin(tau).to(h.dtype))
        if s.warm:
            c = c + s.wemb(tsin(torch.zeros_like(tau) if tau0 is None else tau0).to(h.dtype))
        pe = rope_table(T, s.dh, ns, 0, h.device) if s.rope else None
        mask = kva = None
        nm = s.n_fut * T
        nmain = getattr(s, "n_main", 0)  # (gawm) train --far_aux：前 n_main 个槽（近槽）在每一层都看不到之后的槽（远槽），远槽看全部；未设 = 原行为
        if nmain and ns > nmain:
            mask = nmain * T  # 整数 mask：见 Attn.forward
        elif ns > s.n_fut:  # 辅助槽：主 token 只看主 token，辅助 token 看全部；辅助槽可用另一份上下文（--aux_notext：语言丢掉）
            mask = torch.ones(ns * T, ns * T, dtype=torch.bool, device=x.device)
            mask[:nm, nm:] = False
            kva = getattr(kvs, "aux_kv", None)
        kb = getattr(kvs, "kbias", None)
        lang = getattr(kvs, "lang", None)  # (cap MoE-T) 给 MoE 路由的池化语言嵌入；None = 原行为
        for i, (b, kv) in enumerate(zip(s.blocks, kvs)):
            if lang is not None:
                h = b(h, c, kv, pe, mask, None if kva is None else kva[i], nm, kb, lang=lang)
                continue
            h = b(h, c, kv, pe, mask, None if kva is None else kva[i], nm) if kb is None else b(h, c, kv, pe, mask, None if kva is None else kva[i], nm, kb)
        sh, sc = s.hada(c).chunk(2, -1)
        return modulate(s.hn(h), sh, sc)
