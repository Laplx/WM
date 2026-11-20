"""真机 legacy 根的离线图像增强副本（在建根之前做，可选）。

每条训练 episode 生成 K 份增强版：同一份副本里、同一路相机的所有帧用同一组参数（WM 要预测未来帧的特征，逐帧随机会让它去学预测抖动）；
验证 episode 原样放在最后、不增强。动作、本体、末端位姿原样复制（不做镜像等会改变关节含义的增强）。
输出仍是 legacy 格式（docs/real_robot.md 第 3 节），建根时用脚本最后打印的 --val_from 划验证集。

增强（每份副本、每路相机各抽一组）：亮度 ×U(1±b)、对比度 ×U(1±c)、饱和度 ×U(1±s)、色调 ±h（OpenCV 0–179 刻度的比例）；
几何：放大 U(1, 1+z)、平移 ±t（相对边长）、旋转 ±r 度（反射填边）。--no_geo 关几何、只做颜色；--geo_wrist 0 只对头部相机做几何。

用法：
  python scripts/aug_legacy.py --src /path/to/真机根 --dst /path/to/增强根 --tasks <任务名> --k 4 --n_val 5 --workers 8
  python -m probwm.render_robotwin legacy --out $D --clean_root /path/to/增强根 --tasks <任务名> --P 4 --stride 2 --workers 8 --val_from <打印的值>
"""
import argparse
import glob
import json
import os
import re
import shutil
from multiprocessing import get_context

import numpy as np

CAMS = ("head_camera", "left_camera", "right_camera")


def sample(rng, a, wrist):
    p = dict(b=rng.uniform(1 - a.bright, 1 + a.bright), c=rng.uniform(1 - a.contrast, 1 + a.contrast),
             s=rng.uniform(1 - a.sat, 1 + a.sat), h=rng.uniform(-a.hue, a.hue) * 180)
    if not a.no_geo and (not wrist or a.geo_wrist):
        p.update(z=rng.uniform(1, 1 + a.zoom), tx=rng.uniform(-a.shift, a.shift), ty=rng.uniform(-a.shift, a.shift), r=rng.uniform(-a.rot, a.rot))
    return {k: float(v) for k, v in p.items()}


def apply(img, p):  # img：解码出的 (H, W, 3) uint8，按 RGB 处理（legacy 约定直接编 RGB 数组）
    import cv2
    x = img.astype(np.float32)
    x = x * p["b"]
    m = x.mean()
    x = (x - m) * p["c"] + m
    hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + p["h"]) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * p["s"], 0, 255)
    y = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if "z" in p:
        h, w = y.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), p["r"], p["z"])
        M[0, 2] += p["tx"] * w
        M[1, 2] += p["ty"] * h
        y = cv2.warpAffine(y, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return y


def write_imgs(g, name, src_ds, blobs):
    import h5py
    if src_ds.dtype.kind == "S":  # 官方 legacy：定长字节串（补零）
        g.create_dataset(name, data=np.array(blobs, dtype=f"S{max(len(b) for b in blobs)}"))
    else:  # 其余一律写成变长 uint8（不能用变长字符串：JPEG 里有 0 字节）
        ds = g.create_dataset(name, (len(blobs),), dtype=h5py.vlen_dtype(np.uint8))
        for i, b in enumerate(blobs):
            ds[i] = np.frombuffer(b, np.uint8)


def one(job):
    import cv2
    import h5py
    src, dst, params, q = job
    if params is None:
        shutil.copyfile(src, dst)
        return dst
    with h5py.File(src, "r") as f, h5py.File(dst, "w") as o:
        def visit(n, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            cam = n.split("/")[1] if n.startswith("observation/") and n.endswith("/rgb") else None
            g = o.require_group(os.path.dirname(n)) if os.path.dirname(n) else o
            if cam in params:
                blobs = []
                for x in obj[()]:
                    im = cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR)
                    blobs.append(cv2.imencode(".jpg", apply(im, params[cam]), [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes())
                write_imgs(g, os.path.basename(n), obj, blobs)
            else:
                g.create_dataset(os.path.basename(n), data=obj[()])
        f.visititems(visit)
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="真机 legacy 根（<根>/<任务>/data/episode<k>.hdf5、instructions/episode<k>.json）")
    p.add_argument("--dst", required=True, help="输出根（不能与 --src 相同）")
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--k", type=int, default=4, help="每条训练 episode 的增强副本数")
    p.add_argument("--n_val", type=int, default=-1, help="编号最大的这么多条作验证集（原样、不增强）；-1 = 约 10%（至少 1 条）")
    p.add_argument("--bright", type=float, default=0.2); p.add_argument("--contrast", type=float, default=0.2)
    p.add_argument("--sat", type=float, default=0.2); p.add_argument("--hue", type=float, default=0.03)
    p.add_argument("--zoom", type=float, default=0.08); p.add_argument("--shift", type=float, default=0.05); p.add_argument("--rot", type=float, default=3.0)
    p.add_argument("--no_geo", action="store_true", help="只做颜色增强")
    p.add_argument("--geo_wrist", type=int, default=1, help="0 = 腕部相机不做几何增强")
    p.add_argument("--jpeg_q", type=int, default=95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    assert os.path.abspath(a.src) != os.path.abspath(a.dst)
    rng = np.random.default_rng(a.seed)
    jobs, vfs = [], set()
    for t in a.tasks:
        ks = sorted(int(re.findall(r"episode(\d+)\.hdf5$", x)[0]) for x in glob.glob(os.path.join(a.src, t, "data", "episode*.hdf5")))
        assert ks == list(range(len(ks))), f"{t}：episode 编号须为 0..N−1 连续"
        nv = a.n_val if a.n_val >= 0 else max(1, round(0.1 * len(ks)))
        tr, va = ks[:len(ks) - nv], ks[len(ks) - nv:]
        plan = [(k, None) for k in tr]  # 原样训练集在前
        for c in range(1, a.k + 1):
            for k in tr:
                plan.append((k, {cam: sample(rng, a, cam != "head_camera") for cam in CAMS}))
        plan += [(k, None) for k in va]  # 验证集原样放最后
        vf = len(tr) * (1 + a.k)
        vfs.add(vf)
        os.makedirs(os.path.join(a.dst, t, "data"), exist_ok=True)
        os.makedirs(os.path.join(a.dst, t, "instructions"), exist_ok=True)
        amap = []
        for j, (k, par) in enumerate(plan):
            shutil.copyfile(os.path.join(a.src, t, "instructions", f"episode{k}.json"), os.path.join(a.dst, t, "instructions", f"episode{j}.json"))
            jobs.append((os.path.join(a.src, t, "data", f"episode{k}.hdf5"), os.path.join(a.dst, t, "data", f"episode{j}.hdf5"), par, a.jpeg_q))
            amap.append(dict(out=j, src=k, split="val" if j >= vf else "train", params=par))
        json.dump(dict(val_from=vf, k=a.k, args=vars(a), episodes=amap), open(os.path.join(a.dst, t, "aug_map.json"), "w"), indent=1)
        print(f"{t}：训练 {len(tr)} 条 × (1 + {a.k}) = {vf}，验证 {len(va)} 条（输出编号 {vf}..{vf + len(va) - 1}）", flush=True)
    with get_context("spawn").Pool(a.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(one, jobs), 1):
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i} / {len(jobs)}", flush=True)
    assert len(vfs) == 1, f"各任务的 --val_from 不同 {sorted(vfs)}：请分任务建根"
    print(f"建根时用：--val_from {vfs.pop()}")


if __name__ == "__main__":
    main()
