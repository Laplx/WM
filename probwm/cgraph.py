"""闭环推理的 CUDA graph 捕获（closed_loop --graphs，默认关）：把 WM / IDM 速度场 / canon / prior / 编码器这些"确定性"的模块调用
按输入形状各捕获一张图，之后每次只把输入拷进静态缓冲再重放，省掉 B=1 时占大头的 kernel 启动 / Python 开销。
随机数（Euler 采样的初始噪声等）都在被包的函数之外、照常 eager 抽，所以重放执行的是同一串 kernel、同一组输入 → 输出与不开逐位相同
（scripts/fullbench_graphs.py 核对）。被包的函数必须只读输入、不改状态、不抽随机数；输出在返回前拷贝一份（下一次重放会覆盖静态缓冲）。
autocast：捕获和 eager 预热都在 cache_enabled=False 的嵌套 autocast 里（外层 autocast 的权重缓存在退出时会释放，不能被图引用）。"""
import torch
from torch.utils import _pytree as pt

from .model import KV

try:  # KV（list 子类，带 .heat）注册成 pytree 节点；heat 只支持 None（MS-far / 主线都不用 where）
    pt.register_pytree_node(KV, lambda k: (list(k), k.heat), lambda c, h: _kv(c, h))
except ValueError:
    pass


def _kv(c, h):
    k = KV(c)
    k.heat = h
    return k


def _ac():  # (autocast 开着?, 精度)，兼容新旧 API
    try:
        on = torch.is_autocast_enabled("cuda")
        return on, torch.get_autocast_dtype("cuda") if on else None
    except TypeError:
        on = torch.is_autocast_enabled()
        return on, torch.get_autocast_gpu_dtype() if on else None


class Graphed:
    def __init__(s, f, pool=None, max_graphs=32):
        s.f, s.g, s.pool, s.max = f, {}, pool, max_graphs

    def __call__(s, *args, **kw):
        leaves, spec = pt.tree_flatten((args, kw))
        ac, dt = _ac()
        key = (repr(spec), ac, dt,
               tuple((tuple(x.shape), x.stride(), x.dtype, x.device) if torch.is_tensor(x) else ("v", x) for x in leaves))
        if key not in s.g:
            if len(s.g) >= s.max or any(torch.is_tensor(x) and x.device.type != "cuda" for x in leaves):
                return s.f(*args, **kw)
            s.g[key] = s.capture(leaves, spec, ac, dt)
        g, sin, sout = s.g[key]
        for a, b in zip(sin, leaves):
            if torch.is_tensor(a):
                a.copy_(b)
        g.replay()
        return pt.tree_map(lambda x: x.clone() if torch.is_tensor(x) else x, sout)

    def capture(s, leaves, spec, ac, dt):
        # 静态输入保持与实参相同的步长（如 x[:, :T] 这种切片视图），保证捕获的 kernel 与 eager 时同一套
        sin = [torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device).copy_(x) if torch.is_tensor(x) else x for x in leaves]
        a, k = pt.tree_unflatten(sin, spec)
        ctx = (lambda: torch.autocast("cuda", dt, cache_enabled=False)) if ac else torch.no_grad
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st), torch.no_grad():
            for _ in range(2):
                with ctx():
                    s.f(*a, **k)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g, pool=s.pool):
            with ctx():
                sout = s.f(*a, **k)
        return g, sin, sout


def wrap(obj, name, pool):
    """obj.name = Graphed(obj.name)（实例属性覆盖；nn.Module 的 forward 同样适用）。"""
    f = getattr(obj, name)
    if not isinstance(f, Graphed):
        setattr(obj, name, Graphed(f, pool))
