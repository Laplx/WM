"""本机全局的仿真 env 信号量（文件锁：进程退出、崩溃、kill -9 都会自动释放）。所有线共用同一个锁目录（按 PROBWM_LOCKS 配置）。
11-07：几百个 env 进程同时建 / 关 EGL 渲染上下文会堵死 NVIDIA 驱动锁（别人的 nvidia-smi 卡在 D 状态）→ 建 / 关 env 必须排队。

  建 / 关 env（创建 / 销毁渲染上下文）的地方：
      from probwm import envlock
      with envlock.slot():            # 本机同一时刻最多 PROBWM_MAX_BUILD（默认 4）个在建 / 在关
          env = make_env(...); env.reset()
      ...
      with envlock.slot():
          env.close()
  每条线的 env 进程总数（在 env 子进程启动时调用一次，返回的槽持有到进程结束）：
      keep = envlock.hold("robocasa", 32)   # 本机最多 32 个 RoboCasa env 进程；满了就阻塞等
  check：等待期间周期调用的函数（如父进程没了就 raise SystemExit），可省。

看门狗（只管 build 槽）：持有者在锁文件里写"pid 起始时间"；等待者发现某个 build 槽被持有超过 PROBWM_STUCK_S（默认 600 s），
且持有进程（同用户）有线程卡在 NVIDIA 驱动里（wchan 为 nvkms_* / os_acquire_rwlock*），就 kill -9 它，并把该锁文件改名作废
（卡在内核里的进程要离开内核才会真正退出、释放旧锁；新文件是新 inode，等待者立刻可以接管这个槽），打印一行日志。"""
import fcntl
import os
import random
import signal
import time

DIR = os.environ.get("PROBWM_LOCKS", os.path.join(os.path.expanduser("~/.cache"), "probwm", "locks"))
MAX_BUILD = int(os.environ.get("PROBWM_MAX_BUILD", 4))
STUCK_S = float(os.environ.get("PROBWM_STUCK_S", 600))


def _stuck_in_driver(pid):  # 同用户、且有线程的 wchan 是 nvkms_* / os_acquire_rwlock*
    try:
        if os.stat(f"/proc/{pid}").st_uid != os.getuid():
            return False
        for t in os.listdir(f"/proc/{pid}/task"):
            w = open(f"/proc/{pid}/task/{t}/wchan").read().strip()
            if w.startswith("nvkms") or w.startswith("os_acquire_rwlock"):
                return True
    except Exception:
        return False
    return False


class Slot:
    def __init__(s, kind, n, check=None):
        s.kind, s.n, s.check, s.f = kind, n, check, None

    def path(s, i):
        return os.path.join(DIR, f"{s.kind}_{i}")

    def acquire(s):
        os.makedirs(DIR, exist_ok=True)
        while True:
            for i in random.sample(range(s.n), s.n):
                f = open(s.path(i), "a")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    f.close()
                    continue
                os.ftruncate(f.fileno(), 0)  # 写入持有者（看门狗用）
                f.write(f"{os.getpid()} {time.time():.0f}\n")
                f.flush()
                s.f = f
                return s
            if s.kind == "build":
                s.reap()
            if s.check is not None:
                s.check()
            time.sleep(0.2 + 0.5 * random.random())

    def reap(s):  # 看门狗：见模块说明
        now = time.time()
        for i in range(s.n):
            p = s.path(i)
            try:
                pid, t0 = open(p).read().split()[:2]
                pid, t0 = int(pid), float(t0)
            except Exception:
                continue
            if now - t0 < STUCK_S or pid == os.getpid() or not _stuck_in_driver(pid):
                continue
            with open(p, "a") as f:  # 确认这个槽确实还被占着（空出来了就不动它，下一轮正常去拿）
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    continue
                except OSError:
                    pass
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
            try:
                os.rename(p, f"{p}.stale.{pid}.{int(now)}")
            except Exception:
                continue  # 别的等待者已经处理了
            print(f"[envlock] {s.kind} 槽 {i} 被 pid {pid} 持有 {now - t0:.0f}s 且卡在 NVIDIA 驱动里：已 kill -9 并接管", flush=True)

    def release(s):
        if s.f is not None:
            try:
                os.ftruncate(s.f.fileno(), 0)  # 清掉持有者信息再放锁
            except OSError:
                pass
            s.f.close()
            s.f = None

    def __enter__(s):
        return s.acquire()

    def __exit__(s, *exc):
        s.release()


def slot(check=None):  # 建 / 关 env 的"在建"槽（本机所有线共用 MAX_BUILD 个）
    return Slot("build", MAX_BUILD, check)


def hold(kind, n, check=None):  # 占一个长期槽（kind 类共 n 个，如每线的 env 进程总数），返回已拿到的槽
    return Slot(kind, n, check).acquire()
