"""One-command pipeline runner.

Usage:
    # 1) put the real dataset in data/raw/:  HDFS.log  and  anomaly_label.csv
    #    (or run with --synthetic to smoke-test on generated data)
    # 2) optional Tier 3:  export OPENAI_API_KEY=sk-...
    python run_all.py                 # real data in data/raw/
    python run_all.py --synthetic     # generate + run on synthetic scaffold
"""
import sys, subprocess, pathlib
SRC = pathlib.Path(__file__).resolve().parent / "src"

def step(mod):
    print(f"\n=== {mod} ===")
    subprocess.run([sys.executable, str(SRC / mod)], check=True)

if __name__ == "__main__":
    if "--synthetic" in sys.argv:
        step("make_synthetic.py")
        step("parsing.py")
        step("sessionize.py")
    else:
        # real HDFS_v1: build sessions from the official preprocessed traces
        step("load_preprocessed.py")
    step("features.py")
    step("benchmark.py")     # runs Tier 0-3 + ablation + figures
    print("\nDone. See results/results.csv and results/figures/.")
