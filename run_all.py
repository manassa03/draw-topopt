"""Run every result and verification script and save its console output.

Usage, from the repository root:  python run_all.py
Writes results/logs/<script>.log for each script, results/logs/pytest.log, and
results/logs/environment.txt (Python and package versions, machine, mmapy defaults).
"""
import importlib.metadata as md
import inspect
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "results" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    "runs/enclosure.py",
    "runs/bumper.py",
    "verification/enclosure_handbuild.py",
    "verification/bumper_handbuild.py",
    "verification/cantilever_check.py",
    "verification/enclosure_cold_bottom.py",
    "verification/enclosure_mirror_warmstart.py",
    "verification/enclosure_from_handbuild.py",
    "verification/bumper_asymmetry.py",
    "verification/filter_radius.py",
    "verification/mesh_independence.py",
    "verification/matched_control.py",
    "verification/sensitivity_starvation.py",
    "verification/handbuild_sweep.py",
    "verification/no_skeleton.py",
    "verification/make_method_figures.py",
]

with open(LOGS / "environment.txt", "w") as f:
    f.write(f"python {sys.version}\n")
    f.write(f"platform {platform.platform()}\n")
    f.write(f"processor {platform.processor()}\n")
    f.write(f"cpu_count {os.cpu_count()}\n")
    for pkg in ("numpy", "scipy", "mmapy", "matplotlib", "pytest"):
        try:
            f.write(f"{pkg} {md.version(pkg)}\n")
        except md.PackageNotFoundError:
            f.write(f"{pkg} NOT INSTALLED\n")
    try:
        import mmapy
        f.write(f"mmapy.mmasub signature: {inspect.signature(mmapy.mmasub)}\n")
    except Exception as exc:  # noqa: BLE001
        f.write(f"could not read mmapy.mmasub signature: {exc}\n")


def run(cmd, log):
    print(f"running {' '.join(cmd[1:])}", flush=True)
    with open(log, "w") as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  -> {log.relative_to(ROOT)}  (exit code {code})", flush=True)
    return code


failed = []
if run([sys.executable, "-m", "pytest", "-v"], LOGS / "pytest.log") != 0:
    failed.append("pytest")
for s in SCRIPTS:
    if run([sys.executable, s], LOGS / (Path(s).stem + ".log")) != 0:
        failed.append(s)
print("\nALL DONE, no failures" if not failed else f"\nFAILED: {failed}")