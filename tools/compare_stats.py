"""Compare Python vs C++ tracker stats JSON side by side."""
import json
import sys

p = json.load(open(sys.argv[1]))
c = json.load(open(sys.argv[2]))

print(f"{'metric':<26}{'python':<16}{'c++':<16}")
print("-" * 58)
for k in p["camera_model"]:
    print(f"{'cam.' + k:<26}{str(p['camera_model'][k]):<16}{str(c['camera_model'][k]):<16}")
for t in ("A", "B"):
    pt, ct = p["teams"].get(t, {}), c["teams"].get(t, {})
    for f in ("track_ids", "total_distance_m", "top_speed_kmh", "avg_space_control", "possession"):
        print(f"{'team ' + t + '.' + f:<26}{str(pt.get(f)):<16}{str(ct.get(f)):<16}")
for k in p["ball"]:
    print(f"{'ball.' + k:<26}{str(p['ball'][k]):<16}{str(c['ball'][k]):<16}")

for name, d in (("python", p), ("c++", c)):
    secs = [pl["seconds_tracked"] for pl in d["players"]]
    print(f"{name}: players>=1s={len(secs)}  mean_dur={sum(secs)/len(secs):.1f}s  "
          f"max_dur={max(secs):.1f}s  tracks>5s={sum(s>5 for s in secs)}")
