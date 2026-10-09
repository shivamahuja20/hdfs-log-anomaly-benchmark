"""Step 2 — parse raw HDFS.log into event templates with Drain3.
Input : data/raw/HDFS.log
Output: data/interim/structured_logs.parquet (LineId, block_ids, EventId, template)
        data/interim/templates.csv (EventId -> template)
Streams the file so it scales to the full 11M-line log.
"""
import re, csv
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction
from config import RAW_LOG, INTERIM, DRAIN_DEPTH, DRAIN_SIM_TH

BLOCK_RE = re.compile(r"(blk_-?\d+)")
# HDFS line: date time pid level component: message
LINE_RE = re.compile(r"^(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+([\w.$]+):\s+(.*)$")

def build_miner():
    cfg = TemplateMinerConfig()
    cfg.drain_depth = DRAIN_DEPTH
    cfg.drain_sim_th = DRAIN_SIM_TH
    # mask variable parts so ~29 stable templates emerge (block ids, ips, numbers)
    cfg.masking_instructions = [
        MaskingInstruction(r"blk_-?\d+", "BLK"),
        MaskingInstruction(r"/?\d+\.\d+\.\d+\.\d+(:\d+)?", "IP"),
        MaskingInstruction(r"\b\d+\b", "NUM"),
    ]
    cfg.mask_prefix, cfg.mask_suffix = "<", ">"
    return TemplateMiner(config=cfg)

def parse(limit=None):
    miner = build_miner()
    rows = []
    with open(RAW_LOG, "r", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            line = line.rstrip("\n")
            m = LINE_RE.match(line)
            msg = m.group(6) if m else line
            blocks = BLOCK_RE.findall(line)
            res = miner.add_log_message(msg)
            rows.append({
                "LineId": i,
                "block_ids": ",".join(sorted(set(blocks))),
                "EventId": f"E{res['cluster_id']}",
                "template": res["template_mined"],
            })
    df = pd.DataFrame(rows)
    df.to_parquet(INTERIM / "structured_logs.parquet", index=False)
    # template dictionary
    tpl = (df[["EventId", "template"]].drop_duplicates()
             .sort_values("EventId").reset_index(drop=True))
    tpl.to_csv(INTERIM / "templates.csv", index=False)
    return df, tpl

if __name__ == "__main__":
    df, tpl = parse()
    print(f"[parse] {len(df)} lines -> {len(tpl)} templates")
    print(tpl.to_string(index=False)[:2000])
