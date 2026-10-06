"""Print the last [eval] / time-limit / done lines from a downloaded kernel log.
Usage: python kaggle/taillog.py build/s1.log"""
import sys

raw = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
hits = []
for marker in ("[eval] step", "time limit", "done. steps", "saved ckpt_best"):
    pos = 0
    while True:
        i = raw.find(marker, pos)
        if i < 0:
            break
        end = raw.find("\\n", i)
        hits.append((i, raw[i:end if end > 0 else i + 110][:110]))
        pos = i + 1
for _, line in sorted(hits)[-6:]:
    print(line)
