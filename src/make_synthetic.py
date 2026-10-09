"""DEV SCAFFOLD ONLY — generates a synthetic log file in the exact HDFS line
format so the whole pipeline can be validated end-to-end before the real
HDFS_v1 data is available. The FINAL benchmark must be run on the real dataset
(He et al., LogHub HDFS_v1). This is grounded in the real HDFS event grammar
(block lifecycle: allocate -> receiving -> received -> ack -> replicate -> serve/delete).
"""
import random
from config import RAW, SEED

random.seed(SEED)

# Real HDFS-style message templates (variables get filled with ids/sizes/ips).
TPL = {
    "ALLOC":  "BLOCK* NameSystem.allocateBlock: /user/root/rand{p}. {blk}",
    "RECVING":"Receiving block {blk} src: /{ip1}:{pt1} dest: /{ip2}:{pt2}",
    "RECVD":  "Received block {blk} of size {sz} from /{ip1}",
    "RESP":   "PacketResponder {r} for block {blk} terminating",
    "ADDSTORE":"BLOCK* NameSystem.addStoredBlock: blockMap updated: {ip2}:{pt2} is added to {blk} size {sz}",
    "VERIFY": "Verification succeeded for {blk}",
    "SERVED": "{blk} Served block to /{ip1}",
    "REPL":   "BLOCK* ask {ip1}:{pt1} to replicate {blk} to datanode(s) {ip2}:{pt2}",
    "DELETE": "Deleting block {blk} file /hadoop/dfs/data/current/{blk}",
    "INVAL":  "BLOCK* NameSystem.delete: {blk} is added to invalidSet of {ip1}:{pt1}",
    # anomaly-flavoured events
    "EXC":    "writeBlock {blk} received exception java.io.IOException: Connection reset",
    "TIMEOUT":"PendingReplicationMonitor timed out for {blk}",
    "REDUND": "Redundant addStoredBlock request received for {blk} on {ip2}:{pt2}",
}
LEVEL = {"EXC": "WARN", "TIMEOUT": "WARN"}

def normal_flow():
    """A realistic, VARIABLE normal block lifecycle: replication factor 2-3,
    occasional benign extra reads/verifications. This variation is what makes
    the detection task non-trivial (real HDFS normals are not identical)."""
    rep = random.choice([2, 3, 3, 3])            # replication factor (usually 3)
    flow = ["ALLOC"]
    flow += ["RECVING"] * rep
    flow += ["RECVD"] * rep
    flow += ["RESP"] * rep
    flow += ["ADDSTORE"] * rep
    flow += ["VERIFY"] * random.choice([1, 1, 2])  # occasional re-verify
    flow += ["SERVED"] * random.choice([0, 1, 1, 2])  # block may be read 0-2 times
    return flow

def rip():   return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
def rport(): return random.randint(10000, 60000)

def fill(kind, blk):
    return TPL[kind].format(blk=blk, p=random.randint(1,9999), sz=random.randint(1000,120000),
                            r=random.randint(1,3), ip1=rip(), ip2=rip(), pt1=rport(), pt2=rport())

def emit(lines, ts, kind, blk):
    lvl = LEVEL.get(kind, "INFO")
    comp = "dfs.DataNode$PacketResponder" if kind in ("RESP","RECVD") else \
           "dfs.FSNamesystem" if kind in ("ALLOC","ADDSTORE","INVAL","REPL") else "dfs.DataNode"
    date = "081109"
    tm = f"{ts//3600%24:02d}{ts//60%60:02d}{ts%60:02d}"
    pid = random.randint(1, 400)
    lines.append(f"{date} {tm} {pid} {lvl} {comp}: {fill(kind, blk)}")

def make(n_blocks=3000, anom_rate=0.03):
    lines, labels, ts = [], {}, 200000
    for i in range(n_blocks):
        blk = f"blk_{random.randint(-9_000_000_000, 9_000_000_000)}"
        is_anom = random.random() < anom_rate
        flow = normal_flow()
        if is_anom:
            mode = random.choice(["exception", "missing_ack", "timeout", "reorder", "redundant"])
            if mode == "exception":
                flow = flow[:4] + ["EXC"] + flow[4:]
            elif mode == "missing_ack":
                flow = [e for e in flow if e != "RESP"]           # acks never arrive
            elif mode == "timeout":
                flow = flow[:6] + ["TIMEOUT", "REPL"]
            elif mode == "reorder":
                flow = ["RECVD", "RECVD", "ALLOC", "RECVING", "RESP", "VERIFY"]
            elif mode == "redundant":
                flow = flow + ["REDUND"] * random.randint(3, 6)
        labels[blk] = "Anomaly" if is_anom else "Normal"
        for kind in flow:
            emit(lines, ts, kind, blk)
            ts += random.randint(0, 3)
        ts += random.randint(1, 8)
    # interleave lines roughly by time to mimic a real multi-block log stream
    random.shuffle(lines)  # weak interleave; block_id still recoverable per line
    return lines, labels

if __name__ == "__main__":
    lines, labels = make()
    (RAW / "HDFS.log").write_text("\n".join(lines) + "\n")
    with (RAW / "anomaly_label.csv").open("w") as f:
        f.write("BlockId,Label\n")
        for blk, lab in labels.items():
            f.write(f"{blk},{lab}\n")
    n_anom = sum(v == "Anomaly" for v in labels.values())
    print(f"[synthetic] {len(lines)} lines, {len(labels)} blocks, "
          f"{n_anom} anomalous ({100*n_anom/len(labels):.2f}%)")
