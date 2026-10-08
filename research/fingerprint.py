"""数据指纹：把「这一跑用的是哪批数据」变成可复查的哈希。

被 `research/record.py` 与 `research/train_nn.py` 共用——两处各写一套哈希算法，
就会出现「训练记录与数据记录对不上」的经典仪表问题。
"""
import glob
import hashlib
import os


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def dataset_fingerprint(pattern):
    """返回 (分片明细, 合并指纹, 总字节数)。

    合并指纹只与内容有关、与路径无关：对「按文件名排序后的分片 sha256 列表」再哈希。
    """
    files = sorted(glob.glob(pattern))
    shards = [{"file": os.path.basename(p), "sha256": sha256_file(p),
               "bytes": os.path.getsize(p)} for p in files]
    combined = hashlib.sha256("".join(s["sha256"] for s in shards).encode()).hexdigest()
    return shards, combined, sum(s["bytes"] for s in shards)
