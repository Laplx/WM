"""帧 → 冻结特征（多卡分片写 memmap）；指令 → 文本嵌入。格式见 README「数据格式」。"""
import argparse, json, os, time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # LVD-1689M 权重用 ImageNet 统计量
WEIGHTS = Path(os.environ.get("PROBWM_WEIGHTS", "weights"))  # 本地权重目录 weights/<HF id>（例如从 ModelScope 下载）


def src(hf):
    return str(WEIGHTS / hf) if (WEIGHTS / hf).is_dir() else hf


def load_dinov3(hf, layers, img, dev, fuse="", grid_to=0):  # fuse（gawm T1）："" = 各层直接平均（原行为）；ln = 各层先过无仿射 LayerNorm 再平均；lncat = 各层 LN 后按通道拼接
    # grid_to（gawm 方案 C，超采样）：>0 且小于 img/patch 时，ViT 输出的 (img/patch)² 网格面积池化（adaptive avg）回 grid_to²（如 384 输入 24×24 → 16×16）；0 = 原行为
    from transformers import AutoModel
    m = AutoModel.from_pretrained(src(hf), attn_implementation="sdpa").float().to(dev).eval()
    blocks = m.layer if hasattr(m, "layer") else m.model.layer  # 4.56/4.57 与新版的模块路径不同
    blocks, L = blocks[: max(layers)], set(layers)
    npre = 1 + m.config.num_register_tokens  # 序列 = [CLS, register×4, patch...]
    pos = m.rope_embeddings(torch.zeros(1, 3, img, img, device=dev))  # RoPE 只依赖分辨率，预先算好
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1) * 255
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1) * 255

    def fwd(x):  # uint8 (B,H,W,3) → (B, g*g, D) fp16
        x = x.permute(0, 3, 1, 2).float()
        if x.shape[-2:] != (img, img):
            x = F.interpolate(x, size=(img, img), mode="bilinear", antialias=True, align_corners=False)
        h, acc, cat = m.embeddings((x - mean) / std), 0, []
        for i, blk in enumerate(blocks, 1):
            h = blk(h, position_embeddings=pos)  # 第 i 个 block 的输出（不过最后的 norm）
            if i in L:
                y = h[:, npre:].float()
                y = F.layer_norm(y, y.shape[-1:]) if fuse else y  # (gawm T1) 原行为里第 12 层的尺度是第 3 / 6 / 9 层的 10–35 倍，平均几乎就是第 12 层
                if fuse == "lncat":
                    cat.append(y)
                else:
                    acc = acc + y
        y = torch.cat(cat, -1) if fuse == "lncat" else acc / len(L)
        g = int(round(y.shape[1] ** 0.5))
        if grid_to and grid_to < g:  # (gawm 方案 C) fp32 上做面积池化，再转 fp16
            B, _, D = y.shape
            y = F.adaptive_avg_pool2d(y.view(B, g, g, D).permute(0, 3, 1, 2), grid_to).permute(0, 2, 3, 1).reshape(B, grid_to * grid_to, D)
        return y.half()
    return fwd


class VJEPA21(torch.nn.Module):
    """V-JEPA 2.1 编码器：官方 facebookresearch/vjepa2 app/vjepa_2_1/models/vision_transformer.py（RoPE、视频 / 图像双 tokenizer、
    层级输出各带 LayerNorm）的最小复现；参数名与官方 ckpt 的 ema_encoder 一致（strict 加载），scripts/enc.sh 与官方实现逐层核对过。
    forward：x (B,3,T,H,W) 已标准化 → 最后一个 tubelet 的 token 在 sel 层（1 起，须为层级输出层）各自过对应 norms_block 后的平均。
    T == 1 走图像 tokenizer（patch_embed_img + img_mod_embed），否则走视频 tokenizer（tubelet 2）。"""

    def __init__(s, D, depth, heads, ps, tb, hier):
        super().__init__()
        nn, M = torch.nn, torch.nn.Module
        s.heads, s.ps, s.tb, s.hier = heads, ps, tb, hier
        s.patch_embed, s.patch_embed_img = M(), M()
        s.patch_embed.proj = nn.Conv3d(3, D, (tb, ps, ps), (tb, ps, ps))
        s.patch_embed_img.proj = nn.Conv3d(3, D, (1, ps, ps), (1, ps, ps))
        s.img_mod_embed, s.video_mod_embed = nn.Parameter(torch.zeros(1, 1, D)), nn.Parameter(torch.zeros(1, 1, D))
        s.blocks = nn.ModuleList()
        for _ in range(depth):
            b = M()
            b.norm1, b.norm2, b.attn, b.mlp = nn.LayerNorm(D, eps=1e-6), nn.LayerNorm(D, eps=1e-6), M(), M()
            b.attn.qkv, b.attn.proj, b.mlp.fc1, b.mlp.fc2 = nn.Linear(D, 3 * D), nn.Linear(D, D), nn.Linear(D, 4 * D), nn.Linear(4 * D, D)
            s.blocks.append(b)
        s.norms_block = nn.ModuleList([nn.LayerNorm(D, eps=1e-6) for _ in hier])

    def rope(s, nt, g, dev):  # 3D RoPE（帧 / 行 / 列各占 head_dim 的 2·⌊hd/3/2⌋ 维，其余不转）；预训练网格 256/16，interpolate_rope
        hd = s.blocks[0].attn.proj.in_features // s.heads
        d = 2 * ((hd // 3) // 2)
        i = torch.arange(nt * g * g, device=dev)
        sc = (256 // s.ps - 1) / (g - 1)
        pos = [(i // (g * g)).float(), (i % (g * g) // g).float() * sc, (i % g).float() * sc]
        om = 1.0 / 10000 ** (torch.arange(d // 2, device=dev).float() / (d / 2))
        f = torch.cat([(p[:, None] * om).repeat_interleave(2, -1) for p in pos], -1)  # (N, 3d)
        return f.cos(), f.sin(), 3 * d

    def forward(s, x, sel):
        B, _, T, H, W = x.shape
        g, img = H // s.ps, T == 1
        h = (s.patch_embed_img if img else s.patch_embed).proj(x).flatten(2).transpose(1, 2)
        h = h + (s.img_mod_embed if img else s.video_mod_embed)
        nt = T if img else T // s.tb
        if getattr(s, "_rk", None) != (nt, g, x.device):
            s._rk, s._r = (nt, g, x.device), s.rope(nt, g, x.device)
        cos, sin, r = s._r

        def rot(q):  # 相邻两维一组旋转：(x1, x2) → (x1 cos − x2 sin, x2 cos + x1 sin)
            a, rest = q[..., :r].float(), q[..., r:]
            y = torch.stack((-a[..., 1::2], a[..., ::2]), -1).flatten(-2)
            return torch.cat([(a * cos + y * sin).to(q.dtype), rest], -1)

        acc, n = 0, g * g
        for i, b in enumerate(s.blocks, 1):
            q, k, v = b.attn.qkv(b.norm1(h)).unflatten(-1, (3, s.heads, -1)).permute(2, 0, 3, 1, 4)
            h = h + b.attn.proj(F.scaled_dot_product_attention(rot(q), rot(k), v).transpose(1, 2).flatten(2))
            h = h + b.mlp.fc2(F.gelu(b.mlp.fc1(b.norm2(h))))
            if i in sel:
                acc = acc + s.norms_block[s.hier.index(i)](h[:, -n:]).float()
            if i == max(sel):
                break
        return acc / len(sel)


def load_vjepa21(hf, layers, img, dev):
    """weights/<hf>/{config.json, encoder.pt}（scripts/enc.sh 从官方 ckpt 的 ema_encoder 抽出）。返回 uint8 (B,W,H,W,3) 因果短窗
    （时间顺序，最后一帧 = 当前帧；W = 1 走图像 tokenizer，否则 W 为 tubelet 的倍数）→ (B, g*g, D) fp16：最后一个 tubelet 的 token。"""
    p = Path(src(hf))
    c = json.loads((p / "config.json").read_text())
    m = VJEPA21(c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], c["patch_size"], c["tubelet_size"], c["hier_layers"])
    m.load_state_dict(torch.load(p / "encoder.pt", map_location="cpu", weights_only=True))
    m = m.float().to(dev).eval()
    assert all(l in c["hier_layers"] for l in layers), (layers, c["hier_layers"])
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1) * 255
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1) * 255

    def fwd(x):
        B, W = x.shape[:2]
        x = x.flatten(0, 1).permute(0, 3, 1, 2).float()
        if x.shape[-2:] != (img, img):
            x = F.interpolate(x, size=(img, img), mode="bilinear", antialias=True, align_corners=False)
        x = ((x - mean) / std).unflatten(0, (B, W)).transpose(1, 2)  # (B,3,W,H,W)
        return m(x, layers).half()
    return fwd


ENCODERS = {  # 名字 → (HF id 或 weights/ 下的目录名, 加载函数)；加载函数返回 uint8 帧（视频编码器为因果短窗）→ patch 特征 的函数
    "dinov3_vits16": ("facebook/dinov3-vits16-pretrain-lvd1689m", load_dinov3),
    "dinov3_vitb16": ("facebook/dinov3-vitb16-pretrain-lvd1689m", load_dinov3),
    "dinov3_vitl16": ("facebook/dinov3-vitl16-pretrain-lvd1689m", load_dinov3),
    # V-JEPA 2.1 ViT-L/16（官方 vjepa2_1_vitl_dist_vitG_384.pt）；因果短窗由 --alloc --clip 决定，写进 info.json 的 clip
    "vjepa21_vitl16": ("facebook/vjepa2.1-vitl16-384", load_vjepa21),
}


def pool(x, k):  # (B, g·g, D) patch 特征 → 网格上 k×k 平均池化 → (B, (g/k)², D)；encode --pool 与 RoboCasa 闭环共用
    B, n, D = x.shape
    g = int(round(n ** 0.5))
    y = F.avg_pool2d(x.float().view(B, g, g, D).permute(0, 3, 1, 2), k)
    return y.permute(0, 2, 3, 1).reshape(B, -1, D).to(x.dtype)


def vid_idx(n, fps, hz=10):  # 原生 n 帧、fps → hz（默认 10 Hz）的帧下标（按时间取最近帧，四舍五入：30 Hz 隔 3、20 Hz 隔 2、15 Hz 交替 2 / 1）；fps < hz 时全取
    if fps < hz:  # (ptdata P2) hz 可改，如 RoboTwin 数据的 25/3 ≈ 8.33 Hz；默认 10 与原来逐位相同
        return np.arange(n)
    i = np.floor(np.arange(int(n * hz / fps) + 2) * (fps / hz) + 0.5).astype(np.int64)
    return i[i < n]


def vid_frames(spec, key=None, img=256):
    """--src_vid（预训练线 P1，scripts/pretrain_data.py）：spec = [mp4, 起始秒, 原生帧数 n, fps, w, h] → (len(vid_idx), img, img, 3) uint8。
    ffmpeg（dav1d）从起始秒解 n 帧，只留 vid_idx 的帧，中心方形裁剪后 cv2 INTER_AREA 缩放（同 render_robotwin.to256）。
    解出的帧不足 n（片段末尾的时间戳舍入）时用最后一帧补齐。
    (ptdata P2，可选，默认行为不变) spec 末尾再给 [hz, stretch]：目标帧率、stretch = 1 时整幅非等比拉伸到 img²（同 render_robotwin.to256，不裁剪）；
    spec = ["h5", 路径, stride]：RoboTwin 格式 HDF5 的头部相机，直接用 render_robotwin.legacy_frames（RMBench 等）。"""
    import subprocess
    import cv2
    if spec[0] == "h5":
        from .render_robotwin import legacy_frames
        return legacy_frames(spec[1], stride=spec[2], img=img)
    p, t0, n, fps, w, h = spec[:6]
    hz, stretch = (spec[6], bool(spec[7])) if len(spec) > 6 else (10, False)
    keep = vid_idx(n, fps, hz)
    pr = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-ss", f"{max(t0 - 0.25 / fps, 0):.6f}", "-i", p,
                           "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=1 << 22)
    s = min(w, h)
    y0, x0, fb, out, j = (h - s) // 2, (w - s) // 2, w * h * 3, [], 0
    if stretch:
        y0, x0, s = 0, 0, max(w, h)  # 切片 [0:s] 越界部分自动截断 = 整幅
    ks = set(keep.tolist())
    while j < n:
        b = pr.stdout.read(fb)
        if len(b) < fb:
            break
        if j in ks:
            out.append(cv2.resize(np.frombuffer(b, np.uint8).reshape(h, w, 3)[y0:y0 + s, x0:x0 + s], (img, img), interpolation=cv2.INTER_AREA))
        j += 1
    pr.stdout.close()
    pr.wait()
    assert out, f"{p} @ {t0}：没解出帧"
    if len(out) < len(keep):
        print(f"  [vid] {p} @ {t0}：解出 {j}/{n} 帧，补 {len(keep) - len(out)} 帧", flush=True)
        out += [out[-1]] * (len(keep) - len(out))
    return np.stack(out)


def alloc(a, root, out):
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig
    hf = ENCODERS[a.encoder][0]
    vid = ENCODERS[a.encoder][1] is load_vjepa21
    cfg = Namespace(**json.loads((Path(src(hf)) / "config.json").read_text())) if vid else AutoConfig.from_pretrained(src(hf))
    if src(hf) == hf and not vid:
        snapshot_download(hf, allow_patterns=["*.json", "*.safetensors"])  # 先下好权重，免得 8 个进程同时下载
    layers = [int(s) for s in a.layers.split(",")]
    assert all(1 <= l <= cfg.num_hidden_layers for l in layers), layers
    assert a.img % cfg.patch_size == 0, (a.img, cfg.patch_size)
    clip = [int(s) for s in a.clip.split(",")] if a.clip else []
    assert bool(clip) == vid, "--clip 只用于（且必须用于）视频编码器"
    assert not clip or (clip == sorted(clip) and clip[-1] == 0 and len(set(clip)) == len(clip)
                        and (len(clip) == 1 or len(clip) % cfg.tubelet_size == 0)), clip
    idx = np.load(root / "meta/index.npz")
    st, ln = idx["demo_start"], idx["demo_len"]
    assert (st == np.concatenate([[0], np.cumsum(ln)[:-1]])).all(), "demo_start 与 demo_len 不连续"
    N, g, D = int(ln.sum()), a.img // cfg.patch_size // a.pool, cfg.hidden_size * (len(layers) if a.fuse == "lncat" else 1)
    if a.grid_to:  # (gawm 方案 C) 超采样：--img 大于 grid_to × patch，编码后面积池化回 grid_to²（与 --pool 互斥）
        assert a.pool == 1 and a.grid_to < g and not vid, (a.grid_to, g, a.pool)
        g = a.grid_to
    if a.proj:  # (gawm T1 门 1) 编码时直接做线性投影（同 pca() 的 proj.pt：(f − mu) @ W），只存投影后的维数
        D = int(torch.load(a.proj, map_location="cpu")["W"].shape[1])
    f = out / "feats.npy"
    if f.exists() or (out / "info.json").exists() and a.parts > 1:
        raise SystemExit(f"{out} 已有特征；确认不要了再手动删除后重新 alloc")
    out.mkdir(parents=True, exist_ok=True)
    if a.proj:
        import shutil
        shutil.copyfile(a.proj, out / "proj.pt")
    for d in out.glob("done_*.txt"):  # 旧的完成记录作废
        d.unlink()
    parts = {}
    if a.parts > 1:  # 分片特征库：demo 按帧数大致均分成 parts 段，每段一个 part_XXX.npy（编码时再建），编完可挪到别的机器（info 的 remote）
        cut = np.searchsorted(np.cumsum(ln), np.arange(1, a.parts) * N / a.parts) + 1
        b = np.concatenate([[0], cut, [len(ln)]])
        parts = {"parts": [[int(b[i]), int(b[i + 1]), int(st[b[i]]), int(st[b[i]] + ln[b[i]:b[i + 1]].sum())] for i in range(a.parts)],
                 **({"remote": a.remote} if a.remote else {})}  # [demo_lo, demo_hi, frame_lo, frame_hi]
    else:
        np.lib.format.open_memmap(f, mode="w+", dtype=np.float16, shape=(N, g * g, D)).flush()
    info = {"encoder": a.encoder, "layers": layers, "grid": [g, g], "dim": D, "img": a.img, **({"clip": clip} if clip else {}),
            **({"pool": a.pool} if a.pool > 1 else {}), **({"sspool": a.grid_to, "base": a.encoder} if a.grid_to else {}), **parts,  # sspool：闭环 / 编码按它建 load_dinov3(grid_to=)；base：特征目录名带后缀时闭环据此找编码器
            **({"fuse": a.fuse, "base": a.encoder} if a.fuse else {}),  # (gawm T1) 闭环按 base 建编码器、按 fuse 融合
            **({"proj": "proj.pt", "proj_src": a.proj} if a.proj else {})}  # 闭环见到 proj 就在编码后做同一个线性映射
    (out / "info.json").write_text(json.dumps(info))
    print(f"alloc {f} {(N, g * g, D)} {N * g * g * D * 2 / 1e9:.1f} GB  {info}")


class Groups(Dataset):  # 每项 = 若干 demo 拼成的约 bs 帧
    def __init__(s, groups, paths, ln, key="rgb", rd=None): s.groups, s.paths, s.ln, s.key, s.rd = groups, paths, ln, key, rd
    def __len__(s): return len(s.groups)

    def __getitem__(s, g):
        xs = []
        for i in s.groups[g]:
            if s.rd is not None:  # 直接从源文件解帧（--src_h5：RoboTwin 官方 HDF5），不经 npz
                x = s.rd(s.paths[i], s.key)
            else:
                with np.load(s.paths[i]) as z:
                    x = z[s.key]
            assert len(x) == s.ln[i], (str(s.paths[i]), x.shape, int(s.ln[i]))
            xs.append(x)
        return g, torch.from_numpy(np.concatenate(xs))


def encode(a, root, out):
    info = json.loads((out / "info.json").read_text())  # 层、分辨率以 alloc 时为准
    tasks = json.loads((root / "meta/tasks.json").read_text())
    idx = np.load(root / "meta/index.npz")
    st, ln = idx["demo_start"], idx["demo_len"]
    paths = [Path(a.render or root / "render") / tasks[t]["suite"] / tasks[t]["task"] / f"demo_{d:03d}.npz"
             for t, d in zip(idx["task_id"], idx["demo"])]
    rd = None
    if a.src_h5:  # meta/src_h5.json：每条 demo 的官方 HDF5 路径（render_robotwin legacy），帧 = 按 stride 抽取、to256
        from functools import partial
        from .render_robotwin import legacy_frames
        sh = json.loads((root / "meta/src_h5.json").read_text())
        paths, rd = sh["paths"], partial(legacy_frames, stride=sh["stride"], img=sh["img"])
        assert len(paths) == len(st), (len(paths), len(st))
    if a.src_vid:  # meta/src_vid.json：每条 demo 的 LeRobot 视频片段（scripts/pretrain_data.py build）；null = 已有特征（不在待编的分片里）
        from functools import partial
        sh = json.loads((root / "meta/src_vid.json").read_text())
        paths, rd = sh["paths"], partial(vid_frames, img=sh["img"])
        assert len(paths) == len(st), (len(paths), len(st))
    done ={int(s) for f in out.glob("done_*.txt") for s in f.read_text().split()}  # 合并所有分片的记录
    lo, hi, f0 = 0, len(st), 0
    if "parts" in info:  # 分片特征库：只编第 a.part 段，写 part_XXX.npy（帧下标减去这段的起点）
        lo, hi, f0, f1 = info["parts"][a.part]
        pf = out / f"part_{a.part:03d}.npy"
        if not pf.exists():
            np.lib.format.open_memmap(pf, mode="w+", dtype=np.float16, shape=(f1 - f0, info["grid"][0] * info["grid"][1], info["dim"])).flush()
    todo = [i for i in range(lo, hi) if i % a.nshard == a.shard and i not in done]
    total = int(ln[todo].sum()) if todo else 0
    print(f"[shard {a.shard}/{a.nshard}] 待处理 {len(todo)} demo / {total} 帧，全局已完成 {len(done)} demo")
    if not todo:
        return
    groups = [[]]
    for i in todo:  # 贪心打包成 ≤bs 帧的组，避免小 batch
        if groups[-1] and ln[groups[-1]].sum() + ln[i] > a.bs:
            groups.append([])
        groups[-1].append(i)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    hf, load = ENCODERS[info["encoder"]]
    clip = torch.tensor(info["clip"], device=dev) if "clip" in info else None
    fwd = load(hf, info["layers"], info["img"], dev, **({"fuse": info["fuse"]} if info.get("fuse") else {}), **({"grid_to": info["sspool"]} if info.get("sspool") else {}))
    pj = None
    if "proj" in info:  # (gawm T1 门 1) 编码时投影（--alloc --proj）
        pj = torch.load(out / info["proj"], map_location=dev)
        pj = (pj["mu"].float(), pj["W"].float())
    if a.compile:
        fwd = torch.compile(fwd)

    def run(x):  # 编译时补齐到固定 batch，避免重编译
        n = len(x)
        if a.compile and n < a.bs:
            x = torch.cat([x, x.new_zeros(a.bs - n, *x.shape[1:])])
        return fwd(x)[:n]

    mm = np.load(out / (f"part_{a.part:03d}.npy" if "parts" in info else "feats.npy"), mmap_mode="r+")
    assert mm.shape == (int(ln[lo:hi].sum()), info["grid"][0] * info["grid"][1], info["dim"]), mm.shape
    dl = DataLoader(Groups(groups, paths, ln, a.key, rd), batch_size=None, num_workers=a.workers,
                    pin_memory=True, prefetch_factor=4 if a.workers > 0 else None)
    t0, nf = time.time(), 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), \
            open(out / f"done_{a.shard}of{a.nshard}.txt", "a") as fd:
        for k, (g, x) in enumerate(dl):
            x = x.to(dev, non_blocking=True)
            if clip is not None:  # 因果短窗：第 t 帧 → 同一 demo 的 t+o 帧（o ≤ 0；demo 开头之前用第一帧补，同闭环里历史重复第一帧）
                l = torch.as_tensor(ln[groups[g]], device=dev)
                b0 = torch.repeat_interleave(torch.cumsum(l, 0) - l, l)
                x = x[b0[:, None] + (torch.arange(len(x), device=dev) - b0)[:, None].add(clip).clamp_min(0)]
            y = torch.cat([run(x[s:s + a.bs]) for s in range(0, len(x), a.bs)])
            if info.get("pool", 1) > 1:
                y = pool(y, info["pool"])
            if pj is not None:
                y = ((y.float() - pj[0]) @ pj[1]).half()
            assert torch.isfinite(y).all(), f"group {g} 出现非有限值"
            if k == 0:
                yf = y.float()
                print(f"  特征统计 mean {yf.mean():.3f} std {yf.std():.3f} absmax {yf.abs().max():.1f}")
            y, o = y.cpu().numpy(), 0
            for i in groups[g]:
                mm[st[i] - f0:st[i] - f0 + ln[i]] = y[o:o + ln[i]]
                o += ln[i]
            fd.write("".join(f"{i}\n" for i in groups[g]))
            fd.flush()
            if a.rm_src and rd is None:  # 编码完立即删掉这组的渲染 npz（磁盘峰值 ≈ 特征本身）
                for i in groups[g]:
                    paths[i].unlink(missing_ok=True)
            nf += len(y)
            if k % 50 == 0 or k == len(groups) - 1:
                print(f"[shard {a.shard}] {nf}/{total} 帧  {nf / (time.time() - t0):.0f} fps")
    mm.flush()
    print(f"[shard {a.shard}] 完成")


def pca(a, root):
    """PCA 降维：data/feats/<encoder> → data/feats/<encoder>_p<n>/{feats.npy (N,T,n) fp16, proj.pt, info.json}。
    WM 的宽度 d=512 < D（768 / 1024）时，x_in 把带噪输入投到 512 维，其余方向的噪声 WM 看不见、去不掉，采样结果里会残留整份初始噪声
    （离线 z_nmse ≈ (D−512)/D + 模型误差）→ 大编码器先降到 ≤ 512 维再用同一配方。投影 = 训练帧（固定种子抽 4096 帧的全部 token）的
    协方差前 n 个主成分；闭环见到 info.json 的 proj 就在编码后做同一个线性映射（f − mu）@ W。"""
    src = root / "feats" / (a.encoder + a.feat_sfx)  # --feat_sfx（gawm T1）：对带后缀的特征库降维；默认空 = 原行为
    info, f = json.loads((src / "info.json").read_text()), np.load(src / "feats.npy", mmap_mode="r")
    out = root / "feats" / f"{a.encoder}{a.feat_sfx}_p{a.pca}"
    ix = np.load(root / "meta/index.npz")
    tr = np.concatenate([np.arange(s, s + l) for s, l, sp in zip(ix["demo_start"], ix["demo_len"], ix["split"]) if sp == 0])
    sel = np.sort(np.random.default_rng(0).choice(tr, min(4096, len(tr)), replace=False))
    x = torch.from_numpy(np.array(f[sel])).cuda().double().flatten(0, 1)
    mu = x.mean(0)
    ev, V = torch.linalg.eigh(torch.cov((x - mu).T))
    ev, V = ev.flip(0), V.flip(1)
    W, r = V[:, :a.pca], (ev[:a.pca].sum() / ev.sum()).item()
    print(f"PCA {a.encoder}: {f.shape[-1]} → {a.pca} 维，保留方差 {r:.4f}（256: {(ev[:256].sum() / ev.sum()).item():.4f}，"
          f"384: {(ev[:384].sum() / ev.sum()).item():.4f}，512: {(ev[:512].sum() / ev.sum()).item():.4f}）")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"mu": mu.float().cpu(), "W": W.float().cpu(), "var": r}, out / "proj.pt")
    if a.store:  # feats.npy 放到别处（如 /dev/shm），这里只留软链
        Path(a.store).mkdir(parents=True, exist_ok=True)
        (out / "feats.npy").unlink(missing_ok=True)
        (out / "feats.npy").symlink_to(Path(a.store) / f"{out.name}.npy")
    o = np.lib.format.open_memmap(out / "feats.npy", mode="w+", dtype=np.float16, shape=(len(f), f.shape[1], a.pca))
    mu, W = mu.float(), W.float()
    # 只投影本数据根的 demo 覆盖的帧（全量根 = 全部帧，同原行为；视图根如 oursC_clean 只写它的帧，其余留空洞）
    for s0, l0 in zip(ix["demo_start"], ix["demo_len"]):
        for i in range(int(s0), int(s0 + l0), 1024):
            j = min(i + 1024, int(s0 + l0))
            o[i:j] = ((torch.from_numpy(np.array(f[i:j])).cuda().float() - mu) @ W).half().cpu().numpy()
    o.flush()
    (out / "info.json").write_text(json.dumps({**info, "dim": a.pca, "base": a.encoder, "proj": "proj.pt", "proj_var": r}))  # 最后写，作完成标记


def text_encoder(hf, dev, maxlen=32):
    """→ embed(langs) → (emb fp16 (n,L,Dt)，填充位置为 0；mask bool (n,L)；被截断条数)。闭环里在线编码新指令也用它（与训练表一致）。"""
    from transformers import AutoModel, AutoTokenizer, T5EncoderModel
    tok = AutoTokenizer.from_pretrained(src(hf))
    if "siglip" in hf.lower():  # SigLIP / SigLIP2 文本塔（与视觉对齐）：按训练方式小写、padding 到 64、不给 attention mask；
        st = AutoModel.from_pretrained(src(hf)).text_model.float().to(dev).eval()  # 输出第 0 个 token = 池化 + head 的对齐嵌入（L2 归一后 ×√Dt，逐维 RMS 1），其后 = 实词 token 的末层状态，共 maxlen 个

        @torch.inference_mode()
        def embed_sig(langs):
            b = tok([x.lower() for x in langs], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
            o = st(input_ids=b.input_ids)
            nr = (b.input_ids != tok.pad_token_id).sum(1)  # 右侧 padding
            pool = F.normalize(o.pooler_output, dim=-1) * o.pooler_output.shape[-1] ** 0.5
            emb = torch.cat([pool[:, None], o.last_hidden_state[:, :maxlen - 1]], 1)
            mask = torch.arange(maxlen, device=dev)[None] <= nr[:, None].clamp(max=maxlen - 1)
            return (emb * mask[..., None]).half(), mask, int((nr > maxlen - 1).sum())
        return embed_sig
    m = (T5EncoderModel if "t5" in hf else AutoModel).from_pretrained(src(hf)).float().to(dev).eval()

    @torch.inference_mode()
    def embed(langs):
        if "e5" in hf.split("/")[-1]:
            langs = ["query: " + x for x in langs]
        b = tok(langs, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt").to(dev)
        ntr = sum(len(ids) > maxlen for ids in tok(langs).input_ids)
        emb, mask = m(input_ids=b.input_ids, attention_mask=b.attention_mask).last_hidden_state, b.attention_mask.bool()
        return (emb * mask[..., None]).half(), mask, ntr
    return embed


def text(a, root):
    hf = a.text_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = json.loads((root / "meta/tasks.json").read_text())
    langs = [t["language"] for t in tasks]
    if a.para_file:  # 增广表：每个任务 = 原指令 + 改写；不足 P 条的用原指令补齐
        pa = json.loads(Path(a.para_file).read_text())
        rows = [[t["language"]] + [x.lower() for x in pa.get(t["task"], [])] for t in tasks]
        P = max(map(len, rows))
        langs = [r[i] if i < len(r) else r[0] for r in rows for i in range(P)]
    extra = {}
    if a.per_demo:  # 逐 episode 指令（meta/langs.json，下标 = meta 的 demo 序号）：行 = 去重排序后的指令，demo_row 给每条 demo 的行号
        dl = json.loads((root / "meta/langs.json").read_text())
        langs = sorted(set(dl))
        extra = {"demo_row": torch.tensor([langs.index(x) for x in dl], dtype=torch.int32)}
    if a.per_demo_aug:  # 逐 episode 的指令增广表（meta/langs_aug.json = {"rows": [[指令...]], "demo_row"}，如 RoboTwin 每条 demo 的官方 seen 指令）：
        la = json.loads((root / "meta/langs_aug.json").read_text())  # → emb (n_row, P, L, Dt)，不足 P 条的行用第 0 条补齐；demo_row 给每条 demo 的行
        P = min(a.per_demo_aug, max(map(len, la["rows"])))
        langs = [r[i] if i < len(r) else r[0] for r in la["rows"] for i in range(P)]
        extra = {"demo_row": torch.tensor(la["demo_row"], dtype=torch.int32)}
    embed = text_encoder(hf, dev, a.text_maxlen)
    out = [embed(langs[i:i + 2048]) for i in range(0, len(langs), 2048)]  # 分块（逐 demo 增广表可有几十万条）
    emb, mask, ntr = torch.cat([o[0].cpu() for o in out]), torch.cat([o[1].cpu() for o in out]), sum(o[2] for o in out)
    if a.para_file:
        emb, mask = emb.view(len(tasks), P, *emb.shape[1:]), mask.view(len(tasks), P, -1)
    if a.per_demo_aug:
        emb, mask = emb.view(len(la["rows"]), P, *emb.shape[1:]), mask.view(len(la["rows"]), P, -1)
    f = root / "text" / f"{hf.split('/')[-1]}{'_' + (a.tag or 'aug') if a.para_file or a.per_demo_aug else ''}.pt"
    f.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"emb": emb.cpu(), "mask": mask.cpu(), **extra}, f)
    print(f"text {f} emb {tuple(emb.shape)}，被截断 {ntr} 条，最长 {int(mask.flatten(0, -2).sum(1).max())} token")


def main():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a("--data", default="data"); a("--encoder", default="dinov3_vits16", choices=list(ENCODERS))
    a("--layers", default="3,6,9,12"); a("--img", type=int, default=256)  # 只在 --alloc 时生效（V-JEPA 2.1 ViT-L：6,12,18,24）
    a("--clip", default="", help="--alloc，视频编码器：因果短窗的帧偏移（数据帧 = 10 Hz，≤0，升序，以 0 结尾），如 -2,0；单帧图像模式为 0")
    a("--alloc", action="store_true"); a("--text", action="store_true")
    a("--text_model", default="google/flan-t5-base", help="--text 用的文本编码器；输出 data/text/<模型名>.pt")
    a("--para_file", default="", help="--text 时给出 {任务名: [改写...]}，输出增广表 data/text/<模型名>_<tag>.pt")
    a("--tag", default="", help="增广表的输出后缀，默认 aug（如 --tag augllm → data/text/<模型名>_augllm.pt）")
    a("--per_demo", action="store_true", help="--text：逐 episode 指令表（meta/langs.json，如 RoboCasa），另存 demo_row")
    a("--per_demo_aug", type=int, default=0, help="--text：逐 episode 增广表，每行最多这么多条（meta/langs_aug.json，如 RoboTwin），输出 <模型名>_<tag>.pt")
    a("--text_maxlen", type=int, default=32, help="--text：token 上限（RoboTwin 指令长，用 64）")
    a("--src_h5", action="store_true", help="帧直接从 meta/src_h5.json 列出的 RoboTwin 官方 HDF5 解码（render_robotwin legacy），不经 npz")
    a("--src_vid", action="store_true", help="帧直接从 meta/src_vid.json 列出的 LeRobot 视频片段解码（10 Hz、中心方形裁剪、256²；预训练线 P1）")
    a("--shard", type=int, default=0); a("--nshard", type=int, default=1)
    a("--bs", type=int, default=512); a("--workers", type=int, default=12); a("--compile", action="store_true")
    a("--key", default="rgb", help="npz 里编码哪个图像：rgb（agentview）| wrist（腕部相机，输出 feats/<encoder>_wrist）")
    a("--parts", type=int, default=1, help="--alloc：分片特征库（part_XXX.npy，按帧数均分 demo），大库编完一段挪到别的机器")
    a("--remote", default="", help="--alloc --parts：分片最终存放处 user@host:dir（info.json 的 remote；Feats.load 本地没有时用 ssh 流式读进内存）")
    a("--part", type=int, default=0, help="分片特征库：编第几段")
    a("--rm_src", action="store_true", help="每组编码完、写入 done 记录后删掉对应的渲染 npz（大特征库、磁盘紧时用）")
    a("--grid_to", type=int, default=0, help="(gawm 方案 C) --alloc：超采样——按 --img（如 384）编码，ViT 网格面积池化回 N×N 再存（如 16，存储同 256 输入）；写进 info.json 的 sspool，闭环照它；默认 0 = 关")
    a("--pool", type=int, default=1, help="--alloc：网格上 k×k 平均池化后再存（RoboCasa 腕部相机用 2 → 8×8，省 3/4 磁盘）")
    a("--render", default="", help="npz 所在的 render 目录（默认 <data>/render）；帧序始终按 <data>/meta")
    a("--pca", type=int, default=0, help=">0：把 feats/<encoder> 用 PCA 降到这么多维 → feats/<encoder>_p<n>（见 pca()）")
    a("--store", default="", help="--pca：feats.npy 实际写到这个目录（软链），如 /dev/shm/probwm_enc/feats")
    a("--feat_sfx", default="", help="特征目录后缀：feats/<encoder>[_<key>]<后缀>（同一编码器多档分辨率并存，如 _384）；alloc / 编码都要给")
    a("--proj", default="", help="(gawm T1 门 1) --alloc：编码时直接做线性投影（pca() 输出的 proj.pt，含 mu / W），存投影后的维数；proj.pt 拷进特征目录，闭环照 info 投影")
    a("--fuse", default="", choices=["", "ln", "lncat"], help="(gawm T1) --alloc：多层融合方式，存进 info.json（编码 / 闭环照它）；空 = 各层直接平均（原行为）；"
                                                            "ln = 各层无仿射 LayerNorm 后平均；lncat = 各层 LN 后拼接（维数 × 层数，可再 --pca）")
    a = p.parse_args()
    root = Path(a.data)
    out = root / "feats" / (a.encoder + ("" if a.key == "rgb" else "_" + a.key) + a.feat_sfx)
    if a.pca:
        pca(a, root)
    elif a.alloc:
        alloc(a, root, out)
    elif a.text:
        text(a, root)
    else:
        encode(a, root, out)


if __name__ == "__main__":
    main()
