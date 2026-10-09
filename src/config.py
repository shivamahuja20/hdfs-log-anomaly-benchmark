"""Central config: paths and hyperparameters shared by every stage.
Edit RAW_LOG / LABEL_CSV to point at the real HDFS_v1 files when available.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
for d in (RAW, INTERIM, PROCESSED, MODELS, RESULTS, FIGS):
    d.mkdir(parents=True, exist_ok=True)

# --- data source (switch these two lines to the real dataset) ---
RAW_LOG = RAW / "HDFS.log"              # real: the 11M-line HDFS.log
LABEL_CSV = RAW / "anomaly_label.csv"   # real: block_id,Label

# --- Drain3 parsing ---
DRAIN_DEPTH = 4
DRAIN_SIM_TH = 0.4

# --- split ---
SPLIT = (0.70, 0.10, 0.20)   # train / val / test
SEED = 42

# --- Tier 1 ---
PCA_VAR = 0.95
IFOREST_ESTIMATORS = 100
IFOREST_CONTAM = 0.03

# --- Tier 2 DeepLog ---
WINDOW_H = 10
TOPG = 9
LSTM_LAYERS = 2
LSTM_HIDDEN = 64
BATCH = 2048
EPOCHS = 12                  # higher-quality run (was 6); pushes F1 toward published
MAX_TRAIN_WINDOWS = 1000000  # more normal-pattern coverage (was 300k; None = all ~5.7M)

# --- Tier 3 LLM ---
LLM_MODEL = "gpt-4o-mini"
LLM_SAMPLE = 300      # stratified sample size for the API tier
FEWSHOT_K = 3         # k normal + k anomalous exemplars
