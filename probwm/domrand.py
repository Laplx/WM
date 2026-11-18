"""canon v2 的领域随机化：我们自己的随机化（不用 LIBERO-Plus 的资产、配置或代码），每条 demo 一个固定的组合扰动。
camaug.py --dr p 调用：sample 抽参数 → apply 改仿真（相机 / 贴图 / 光照，demo 之间由 snapshot 复原）→ 渲染 → image 在 GPU 上做图像退化 → 编码。
  cam    agentview 绕"视轴与桌面的交点"做轨道旋转（方位 ±60°、俯仰 −15°..+30°），沿视轴远近 ×[0.9, 2.0]，原地朝向抖动 ≤10°，
         平移 ≤3 cm，fovy 40–55°；各项独立以 1/2 概率出现（幅度参照 LIBERO-Plus 论文正文的描述：远近 1.01–2×、轨道 15–75°、朝向 2–10°）
  tex    地面 / 桌面 / 桌腿 / 墙面贴图换成程序生成的纹理（分形噪声、条纹、棋盘、木纹、斑块；随机 2–3 色调色板、颗粒）+ 随机重复次数
  light  两盏灯的漫射颜色（含近单色光）、强度、镜面、环境光，方向在 50° 锥内随机，阴影随机开关；headlight 强度
  img    高斯噪声、高斯模糊、任意方向的运动模糊、雾（平滑噪声场与灰白混合）、对比度 / 亮度 / gamma / 饱和度（每类 1/3 概率）
四大类各以概率 p 出现（至少一类）。LIBERO-Plus 噪声维度里的 zoom blur、glass blur 不在训练里（留作没见过的类型）。
p="objw"（canon v4）：照常抽四大类，另外物体外观必出（物体材质 7 成被改：一半换程序纹理、全部换色 / 染色，几何不变）。
{"objc": {"seed"}}（pro，PRO Obj 修法；layoutgen_any --render --objrand objc 用）：可动物体（env 的 objects_dict，目标物与干扰物）的材质只换色——
  这些材质各以 7 成概率把 rgba 乘一个随机色（0.15–1），不换贴图，原贴图的图案（标签 / 印花）留着当身份线索，几何不变；固定件、场景、机器人都不动。
p="wide"（canon v3 的训练数据）：相机必出且范围更大（方位 ±80°、俯仰 −15..+45°、朝向 ≤12°、fovy 35–60°），其余三类仍各 1/2。
评测用的单类扰动（camaug --dr_eval，不进训练）：sphere（轨道方位 ±30–80°、俯仰 −10..+45°，超出训练范围）、zoom（缩放模糊）、glass（玻璃模糊：局部像素随机错位 + 模糊）。"""
import numpy as np

Z_TABLE = 0.9  # 轨道中心所在平面（桌面附近）
TEX = ("texplane", "tex-table", "tex-table-legs", "tex-wall", "textable")
SCENE = ("floorplane", "table_texture", "table_legs", "walls_mat", "table_mat")  # 场景材质；其余不以 robot0_ 开头的材质算物体


def _rot(axis, ang):  # 绕单位轴转 ang 弧度的旋转矩阵
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _q2m(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _m2q(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def _unit(r):
    v = r.normal(size=3)
    return v / np.linalg.norm(v)


def sample(r, p=0.5):
    """→ 参数 dict（可 json 化）；"img" 的退化在主进程里按同一 dict 做。p 为字符串时是评测用的单类扰动（见文件头）。"""
    if isinstance(p, str):
        if p == "objw":
            s = sample(r, 0.5)
            s["obj"] = {"seed": int(r.integers(2 ** 31))}
            return s
        if p == "sphere":
            return {"cam": dict(az=float(r.choice([-1, 1]) * r.uniform(30, 80)), el=float(r.uniform(-10, 45)), dist=1.0, rot=0.0,
                                shift=0.0, fovy=45.0, seed=int(r.integers(2 ** 31)))}
        if p in ("zoom", "glass"):
            g = {"zoom": float(r.uniform(1.1, 1.5))} if p == "zoom" else {"glass": (float(r.uniform(0.5, 2.5)), int(r.integers(1, 6)))}
            return {"img": {**g, "seed": int(r.integers(2 ** 31))}}
    wide = p == "wide"
    if wide:
        p = 0.5
    on = r.random(4) < p
    on[0] |= wide
    if not on.any():
        on[r.integers(4)] = True
    s = {}
    if on[0]:
        c = {k: bool(r.random() < 0.5) for k in ("orbit", "dist", "orient", "shift")}
        if not any(c.values()):
            c[("orbit", "dist", "orient", "shift")[r.integers(4)]] = True
        az, el, rot, fv = ((80, (-15, 45), 12, (35, 60)) if wide else (60, (-15, 30), 10, (40, 55)))
        s["cam"] = dict(az=float(r.uniform(-az, az)) if c["orbit"] else 0.0, el=float(r.uniform(*el)) if c["orbit"] else 0.0,
                        dist=float(r.uniform(0.9, 2.0)) if c["dist"] else 1.0, rot=float(r.uniform(0, rot)) if c["orient"] else 0.0,
                        shift=float(r.uniform(0, 0.03)) if c["shift"] else 0.0, fovy=float(r.uniform(*fv)), seed=int(r.integers(2 ** 31)))
    if on[1]:
        s["tex"] = dict(seed=int(r.integers(2 ** 31)), which=[t for t in TEX if r.random() < 0.7] or ["textable"])
    if on[2]:
        s["light"] = dict(seed=int(r.integers(2 ** 31)))
    if on[3]:
        g = {}
        if r.random() < 1 / 3:
            g["noise"] = float(r.uniform(0.02, 0.12))
        if r.random() < 1 / 3:
            g["blur"] = float(r.uniform(0.5, 4.0))
        if r.random() < 1 / 3:
            g["motion"] = (float(r.uniform(3, 25)), float(r.uniform(0, 180)))
        if r.random() < 1 / 3:
            g["haze"] = (float(r.uniform(0.15, 0.7)), float(r.uniform(0.6, 1.0)), int(r.integers(2 ** 31)))
        if r.random() < 1 / 3 or not g:
            g["color"] = (float(r.uniform(0.5, 1.5)), float(r.uniform(-0.2, 0.2)), float(np.exp(r.uniform(-0.5, 0.5))),
                          float(r.uniform(0.3, 1.5)))
        g["seed"] = int(r.integers(2 ** 31))
        s["img"] = g
    return s


def _buf(m):  # 贴图像素：mujoco 2.x 叫 tex_rgb，3.x 叫 tex_data
    return m.tex_data if hasattr(m, "tex_data") else m.tex_rgb


def _upload(env, t):  # robosuite 的 upload_texture 把包装类传给了 mujoco，这里直接调
    import mujoco
    rc = env.sim._render_context_offscreen
    rc.gl_ctx.make_current()
    mujoco.mjr_uploadTexture(env.sim.model._model, rc.con, t)


def _tid(m, name):
    import mujoco
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, name)


def snapshot(env):  # → 复原函数（env 在 demo 之间不重建）
    m, sim = env.sim.model._model, env.sim
    i = env.sim.model.camera_name2id("agentview")
    h = m.vis.headlight
    arrs = [m.cam_pos[i], m.cam_quat[i], m.cam_fovy[i:i + 1], m.light_diffuse, m.light_specular, m.light_ambient, m.light_pos,
            m.light_dir, m.light_castshadow, h.ambient, h.diffuse, h.specular, m.mat_rgba, m.mat_texrepeat]
    saved = [x.copy() for x in arrs]
    tids = [t for t in range(m.ntex) if m.tex_type[t] != 2]  # 场景和物体的贴图（天空盒除外）
    tex = {t: _buf(m)[m.tex_adr[t]:m.tex_adr[t] + m.tex_width[t] * m.tex_height[t] * 3].copy() for t in tids}

    def restore(dirty):
        for x, x0 in zip(arrs, saved):
            x[...] = x0
        for t in dirty:
            _buf(m)[m.tex_adr[t]:m.tex_adr[t] + len(tex[t])] = tex[t]
            _upload(env, t)
    return restore


def _noise2d(r, h, w, oct0=4, n_oct=5):  # 分形噪声 ∈ [0,1]，h×w
    from scipy.ndimage import zoom
    out, amp = np.zeros((h, w)), 1.0
    for o in range(n_oct):
        g = oct0 * 2 ** o
        out += amp * zoom(r.random((g + 1, g + 1)), (h / (g + 1), w / (g + 1)), order=1)[:h, :w]
        amp *= 0.5
    out -= out.min()
    return out / max(out.max(), 1e-6)


def texture(r, h, w):  # 程序纹理 uint8 (h,w,3)：在 ≤512² 上生成再最近邻放大
    hs, ws = min(h, 512), min(w, 512)
    y, x = np.mgrid[0:hs, 0:ws] / max(hs, ws)
    kind = r.integers(5)
    if kind == 0:
        t = _noise2d(r, hs, ws, int(r.integers(2, 8)))
    elif kind == 1:  # 条纹
        a, f = r.uniform(0, np.pi), r.uniform(4, 40)
        t = 0.5 + 0.5 * np.sin(2 * np.pi * f * (x * np.cos(a) + y * np.sin(a)) + 3 * _noise2d(r, hs, ws, 3, 3) * r.random())
    elif kind == 2:  # 棋盘 / 砖
        n = r.integers(3, 24)
        t = ((np.floor(x * n) + np.floor(y * n * r.uniform(0.5, 2))) % 2).astype(float)
    elif kind == 3:  # 木纹 / 大理石
        t = 0.5 + 0.5 * np.sin(r.uniform(10, 60) * (x + r.uniform(0.5, 4) * _noise2d(r, hs, ws, 2, 4)))
    else:  # 斑块
        t = (_noise2d(r, hs, ws, int(r.integers(3, 10))) > r.uniform(0.3, 0.7)).astype(float)
    pal = r.random((int(r.integers(2, 4)), 3))
    k = t * (len(pal) - 1)
    lo = np.floor(k).astype(int).clip(0, len(pal) - 2)
    c = pal[lo] * (1 - (k - lo))[..., None] + pal[lo + 1] * (k - lo)[..., None]
    c = (c + r.uniform(0, 0.15) * (r.random((hs, ws, 1)) - 0.5)).clip(0, 1)  # 颗粒
    c = np.tile(c, (-(-h // hs), -(-w // ws), 1))[:h, :w]
    return (c * 255).astype(np.uint8)


def scene_z(env):
    """场景里可动物体（自由关节的 body）当前高度的中位数（camaug --zc：相机轨道中心取视轴与这个高度平面的交点）。
    默认的 Z_TABLE = 0.9 是桌面场景的桌面高度；LIBERO-Object / LIBERO-10 的地面场景 agentview 只有 0.65 / 0.96 m 高，
    与 z = 0.9 的交点落在相机身后 0.47 m / 身前 0.11 m → 轨道变成原地转头（常转出场景）、远近缩放方向反了或几乎不动。"""
    m, d = env.sim.model._model, env.sim.data._data
    b = [m.jnt_bodyid[j] for j in range(m.njnt) if m.jnt_type[j] == 0]  # mjJNT_FREE
    return float(np.median([d.xpos[i][2] for i in b])) if b else Z_TABLE


WS = dict(az=90, el=(-10, 60), dist=(0.8, 2.2), shift=0.25, roll=15, rot=10, fovy=(35, 65), margin=0.05, tries=200)
"""camaug --cam ws（11-07，B3）：原则 = "相机位于工作台周围、整个工作区可见的任意位姿"（与评测无关，参数不取自任何评测集）：
  轨道：绕场景中心（视轴与 scene_z 平面的交点）方位 ±90°、俯仰 −10°..+60°（正 = 抬高，相对默认相机；从场景中心看相机的仰角超过 85° 的拒绝，
  避免越过头顶翻转）、远近 0.8–2.2×；不重新对准的平移：每轴 ±25 cm、朝向不变；
  朝向：绕光轴滚转 ≤ 15°，外加原 DR 的随机轴小抖动 ≤ 10°；fovy 35–65°。前四项各以 1/2 概率出现（至少一项），fovy 每次都抽。
  可见性（拒绝采样）：场景里所有物体与固定件的根 body、机器人基座投影到画面，须都在相机前方、离四边 ≥ 5%；不满足就重抽（最多 200 次，仍不行则不扰动相机）。
  只换相机：纹理 / 光照 / 图像退化照原 sample() 抽（同一个 rng 流），相机参数另用 s["cam"]["seed"] 的独立 rng 抽，所以同种子下其余三类与原 DR 逐项相同。"""


WS2 = dict(az=150, el=(-10, 70), dist=(0.7, 2.5), shift=0.25, roll=20, rot=10, fovy=(30, 70), margin=0.05, tries=400, above=0.05, occ=0.25)
"""camaug --cam ws2（11-15，宽视角 canon 第 1 步，预注册 logs/pro.md 11-14 22:50）：原则同 WS，再放宽——方位 ±150°、俯仰 −10°..+70°、远近 0.7–2.5×、
  平移每轴 ±25 cm、滚转 ≤ 20°（随机轴小抖动仍 ≤ 10°）、fovy 30–70°。与 WS 不同：只换相机（贴图 / 光照 / 图像退化都不加）。
  可见性在 WS 的投影检查之外再加两条（冒烟预览里有相机钻到桌下、贴着机械臂的位姿，违背"整个工作区可见"的原则；logs/pro.md 11-15 11:45）：
  相机高于轨道中心平面（桌面 / scene_z）above = 5 cm；从相机向各物体 / 固定件根 body 打射线（mujoco.mj_ray），先撞上别的 body 的比例 ≤ occ = 25%。"""


def ws_points(env):  # 可见性约束用的点：物体、固定件的根 body 与机器人基座的世界坐标
    import mujoco
    m, d, e = env.sim.model._model, env.sim.data._data, env.env
    names = [o.root_body for o in list(getattr(e, "objects_dict", {}).values()) + list(getattr(e, "fixtures_dict", {}).values())] + ["robot0_base"]
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    return np.array([d.xpos[i] for i in ids if i >= 0])


def ws_occluded(env, p):  # --cam ws2：相机 p 到各物体 / 固定件根 body 的视线被别的 body 挡住的比例（不含机器人基座）
    import mujoco
    m, d, e = env.sim.model._model, env.sim.data._data, env.env
    names = [o.root_body for o in list(getattr(e, "objects_dict", {}).values()) + list(getattr(e, "fixtures_dict", {}).values())]
    ids = [i for i in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names) if i >= 0]
    if not ids:
        return 0.0
    k, g = 0, np.array([-1], np.int32)
    for i in ids:
        v = d.xpos[i] - p
        L = float(np.linalg.norm(v))
        x = mujoco.mj_ray(m, d, p.astype(np.float64), v / L, None, 1, -1, g)
        if x >= 0 and x < 0.97 * L and m.body_rootid[m.geom_bodyid[g[0]]] != m.body_rootid[i]:
            k += 1
    return k / len(ids)


def visible(pts, p, R, fovy, margin):  # 相机 (p, R)（MuJoCo：看 −z、上 +y）下各点都在画面内且离边 ≥ margin（方形画面）
    c = (pts - p) @ R  # 相机坐标
    if (c[:, 2] >= -1e-3).any():
        return False
    t = np.tan(np.deg2rad(fovy) / 2)
    uv = c[:, :2] / -c[:, 2:3] / t
    return bool((np.abs(uv) <= 1 - 2 * margin).all())


def cam_ws(env, c, zc):
    """--cam ws 的相机：c = {"seed": ...}（原 sample() 抽到的），就地写入实际用到的参数（存进 dr.json）→ (pos, R, fovy)。"""
    m = env.sim.model._model
    i = env.sim.model.camera_name2id("agentview")
    p0, R0 = m.cam_pos[i].copy(), _q2m(m.cam_quat[i])
    f0 = -R0[:, 2]
    ctr = p0 + ((Z_TABLE if zc is None else zc) - p0[2]) / f0[2] * f0
    pts, r, W = ws_points(env), np.random.default_rng([c["seed"], 11071]), (WS2 if c.get("ws2") else WS)
    for k in range(W["tries"]):
        on = r.random(4) < 0.5
        if not on.any():
            on[r.integers(4)] = True
        az, el = (float(r.uniform(-W["az"], W["az"])), float(r.uniform(*W["el"]))) if on[0] else (0.0, 0.0)
        dist = float(r.uniform(*W["dist"])) if on[1] else 1.0
        sh = r.uniform(-W["shift"], W["shift"], 3) if on[2] else np.zeros(3)
        roll, jit = (float(r.uniform(-W["roll"], W["roll"])), float(r.uniform(0, W["rot"]))) if on[3] else (0.0, 0.0)
        ax, fv = _unit(r), float(r.uniform(*W["fovy"]))
        A = _rot(R0[:, 0], np.deg2rad(-el)) @ _rot([0, 0, 1], np.deg2rad(az))  # 绕相机右轴转 −el = 抬高 el（原 sample() 的 el 正向是压低）
        v = A @ (p0 - ctr)
        if np.degrees(np.arcsin(v[2] / np.linalg.norm(v))) > 85:
            continue
        p, R = ctr + dist * v + sh, A @ R0
        R = _rot(ax, np.deg2rad(jit)) @ _rot(R[:, 2], np.deg2rad(roll)) @ R
        if visible(pts, p, R, fv, W["margin"]):
            if "above" in W:  # ws2：相机须高于轨道中心平面、视线遮挡比例不超过 occ
                if p[2] < ctr[2] + W["above"]:
                    continue
                oc = ws_occluded(env, p)
                if oc > W["occ"]:
                    continue
                c.update(occ=oc)
            c.update(az=az, el=el, dist=dist, shift=sh.tolist(), roll=roll, rot=jit, fovy=fv, tries=k + 1, ws=True)
            return p, R, fv
    c.update(tries=W["tries"], ws=True, fail=True)
    return p0, R0, float(m.cam_fovy[i])


def apply(env, s, zc=None):  # → 改过的贴图 id（复原时要重新上传）；zc：轨道中心所在平面高度（None = Z_TABLE，见 scene_z）
    m, dirty = env.sim.model._model, []
    if "cam" in s and s["cam"].get("ws"):  # --cam ws（见 WS）
        i = env.sim.model.camera_name2id("agentview")
        p, R, fv = cam_ws(env, s["cam"], zc)
        m.cam_pos[i], m.cam_quat[i], m.cam_fovy[i] = p, _m2q(R), fv
    elif "cam" in s:
        c, r = s["cam"], np.random.default_rng(s["cam"]["seed"])
        i = env.sim.model.camera_name2id("agentview")
        p, R = m.cam_pos[i].copy(), _q2m(m.cam_quat[i])
        f = -R[:, 2]
        ctr = p + ((Z_TABLE if zc is None else zc) - p[2]) / f[2] * f  # 视轴与桌面（或 zc 平面）的交点
        A = _rot(R[:, 0], np.deg2rad(c["el"])) @ _rot([0, 0, 1], np.deg2rad(c["az"]))  # 方位绕世界 z，俯仰绕相机右轴
        p, R = ctr + c["dist"] * (A @ (p - ctr)), A @ R
        R = _rot(_unit(r), np.deg2rad(c["rot"])) @ R
        m.cam_pos[i], m.cam_quat[i], m.cam_fovy[i] = p + c["shift"] * _unit(r), _m2q(R), c["fovy"]
    if "tex" in s:
        r = np.random.default_rng(s["tex"]["seed"])
        for name in s["tex"]["which"]:
            t = _tid(m, name)
            h, w, a = m.tex_height[t], m.tex_width[t], m.tex_adr[t]
            _buf(m)[a:a + h * w * 3] = texture(r, h, w).ravel()
            _upload(env, t)
            dirty.append(t)
            for j in np.nonzero(m.mat_texid == t)[0]:
                m.mat_texrepeat[j] = r.uniform(0.5, 4, 2)
                m.mat_rgba[j, :3] = r.uniform(0.6, 1.0, 3)
    if "obj" in s:  # 物体外观（canon v4）：换程序纹理和 / 或颜色，几何不变
        import mujoco
        r = np.random.default_rng(s["obj"]["seed"])
        for j in range(m.nmat):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, j) or ""
            if nm in SCENE or nm.startswith("robot0_") or r.random() < 0.3:
                continue
            t = m.mat_texid[j]
            if t >= 0 and r.random() < 0.5 and t not in dirty:
                h, w, a = m.tex_height[t], m.tex_width[t], m.tex_adr[t]
                _buf(m)[a:a + h * w * 3] = texture(r, h, w).ravel()
                _upload(env, t)
                dirty.append(t)
            m.mat_rgba[j, :3] = r.uniform(0.15, 1.0, 3)
    if "objc" in s:  # (pro) PRO Obj 修法：可动物体只换色（乘性染色，保留原贴图图案），几何不变
        import mujoco
        r = np.random.default_rng(s["objc"]["seed"])
        od = getattr(getattr(env, "env", env), "objects_dict", None)
        assert od, "objc 要 LIBERO env 的 objects_dict（可动物体）"
        pre = tuple(od.keys())
        bids = {b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(pre)}
        for j in sorted({int(m.geom_matid[g]) for g in range(m.ngeom) if m.geom_bodyid[g] in bids and m.geom_matid[g] >= 0}):
            if r.random() < 0.3:
                continue
            m.mat_rgba[j, :3] = r.uniform(0.15, 1.0, 3)
    if "light" in s:
        r, h = np.random.default_rng(s["light"]["seed"]), m.vis.headlight
        for j in range(m.nlight):
            col = np.exp(r.uniform(-0.7, 0.7, 3))
            if r.random() < 0.2:  # 近单色光
                col = np.where(np.arange(3) == r.integers(3), 1.0, r.uniform(0, 0.4, 3)) * 1.5
            m.light_diffuse[j] = m.light_diffuse[j] * col * np.exp(r.uniform(-0.7, 0.7))
            m.light_specular[j] = m.light_specular[j] * r.uniform(0, 3)
            m.light_ambient[j] = m.light_ambient[j] * np.exp(r.uniform(-1, 1))
            m.light_dir[j] = _rot(_unit(r), np.deg2rad(r.uniform(0, 50))) @ m.light_dir[j]
            m.light_pos[j] = m.light_pos[j] + r.normal(0, 1, 3) * [1, 1, 0.3]
            m.light_castshadow[j] = r.random() < 0.5
        g = np.exp(r.uniform(-0.7, 0.5))
        h.ambient[:], h.diffuse[:] = h.ambient * g, h.diffuse * g
    env.sim.forward()
    return dirty


def image(x, g):
    """x uint8 (N,H,W,3) cuda → 同形状 uint8；g = sample(...)["img"]。每条 demo 的退化参数固定，高斯噪声逐帧重采。"""
    import torch
    import torch.nn.functional as F
    gen = torch.Generator(device=x.device).manual_seed(g["seed"])
    y = x.permute(0, 3, 1, 2).float() / 255

    def conv(y, k):  # 逐通道卷积，反射填充
        k = (k / k.sum()).to(y)[None, None].expand(3, 1, *k.shape)
        return F.conv2d(F.pad(y, (k.shape[-1] // 2,) * 2 + (k.shape[-2] // 2,) * 2, mode="reflect"), k, groups=3)
    if "blur" in g:
        n = int(3 * g["blur"]) * 2 + 1
        e = torch.exp(-(torch.arange(n) - n // 2) ** 2 / (2 * g["blur"] ** 2))
        y = conv(y, e[:, None] * e[None])
    if "motion" in g:
        L, a = g["motion"]
        n = int(L) // 2 * 2 + 1
        k = torch.zeros(n, n)
        t = torch.linspace(-L / 2, L / 2, 4 * n)
        xs, ys = (n // 2 + t * np.cos(np.deg2rad(a))).round().long(), (n // 2 + t * np.sin(np.deg2rad(a))).round().long()
        k[ys.clamp(0, n - 1), xs.clamp(0, n - 1)] = 1
        y = conv(y, k)
    if "haze" in g:
        a, c, s = g["haze"]
        fld = torch.from_numpy(_noise2d(np.random.default_rng(s), y.shape[2], y.shape[3], 2, 3)).to(y)
        w = a * (0.5 + 0.5 * fld)
        y = y * (1 - w) + c * w
    if "color" in g:
        ct, b, gam, sat = g["color"]
        gray = y.mean(1, keepdim=True)
        y = gray + sat * (y - gray)
        y = ((y - 0.5) * ct + 0.5 + b).clamp(0, 1) ** gam
    if "zoom" in g:  # 缩放模糊：以中心为原点放大 1..zoom 的 8 份平均
        H = y.shape[2]
        acc = 0
        for f in np.linspace(1, g["zoom"], 8):
            c = int(round(H / f))
            o = (H - c) // 2
            acc = acc + F.interpolate(y[:, :, o:o + c, o:o + c], size=(H, H), mode="bilinear", align_corners=False)
        y = acc / 8
    if "glass" in g:  # 玻璃模糊：模糊 → 每个像素与 ±δ 内的随机像素交换位置（这里用随机位移采样）→ 再模糊
        sg, dl = g["glass"]
        n = int(3 * sg) * 2 + 1
        e = torch.exp(-(torch.arange(n) - n // 2) ** 2 / (2 * sg ** 2))
        y = conv(y, e[:, None] * e[None])
        H, W = y.shape[2:]
        iy, ix = torch.meshgrid(torch.arange(H, device=y.device), torch.arange(W, device=y.device), indexing="ij")
        dy = torch.randint(-dl, dl + 1, (H, W), generator=gen, device=y.device)
        dx = torch.randint(-dl, dl + 1, (H, W), generator=gen, device=y.device)
        y = y[:, :, (iy + dy).clamp(0, H - 1), (ix + dx).clamp(0, W - 1)]
        y = conv(y, e[:, None] * e[None])
    if "noise" in g:
        y = y + g["noise"] * torch.randn(y.shape, generator=gen, device=y.device)
    return (y.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
