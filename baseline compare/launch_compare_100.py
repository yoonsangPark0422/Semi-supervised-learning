import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(
    r"C:\Users\FORYOUCOM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)
OUT = ROOT / "result"
OUT.mkdir(parents=True, exist_ok=True)

cmd = [
    str(PYTHON),
    str(ROOT / "compare_baselines.py"),
    "--out",
    str(OUT),
    "--methods",
    "crest,daso,abc,proposed",
    "--epochs",
    "100",
    "--eval-step",
    "1024",
    "--batch-size",
    "32",
    "--mu",
    "7",
    "--num-workers",
    "0",
]

stdout = open(OUT / "runner.stdout.log", "wb")
stderr = open(OUT / "runner.stderr.log", "wb")
creationflags = 0
if sys.platform == "win32":
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

process = subprocess.Popen(
    cmd,
    cwd=str(ROOT),
    stdout=stdout,
    stderr=stderr,
    stdin=subprocess.DEVNULL,
    creationflags=creationflags,
    close_fds=False,
)
(OUT / "runner.pid.txt").write_text(str(process.pid), encoding="utf-8")
print(process.pid)
