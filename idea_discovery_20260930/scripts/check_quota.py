"""Small Slurm metadata guard, safe on login node; invoke before GPU jobs."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from zoneinfo import ZoneInfo
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--gpu-hours", type=float, required=True)
p.add_argument("--cpu-hours", type=float, required=True)
p.add_argument("--memory-mb-hours", type=float, required=True)
args = p.parse_args()
month = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y-%m-01")
raw = subprocess.check_output(["sreport", "-t", "hours", "-T", "gres/gpu,cpu,mem", "cluster", "UserUtilizationByAccount", "start="+month, "end=now", "-P", "-n"], text=True)
by_resource = defaultdict(lambda: defaultdict(float))
for line in raw.splitlines():
    columns = line.split("|")
    if len(columns) >= 6 and columns[4] in ("gres/gpu", "cpu", "mem"):
        by_resource[columns[4]][columns[1]] += float(columns[5])
planned = {"gres/gpu": args.gpu_hours, "cpu": args.cpu_hours, "mem": args.memory_mb_hours}
report = {"calendar_month": month, "timezone": "Asia/Ho_Chi_Minh", "resources": {}, "allowed": True}
for resource, increment in planned.items():
    ranking = sorted(by_resource[resource].items(), key=lambda x: x[1], reverse=True)
    if len(ranking) < 5:
        raise RuntimeError("Cannot verify fifth-user quota threshold")
    fifth = ranking[4]
    current = by_resource[resource].get("nhatnc129", 0)
    okay = current + increment <= .5 * fifth[1]
    report["resources"][resource] = {"current": current, "planned_time_limit": increment, "fifth_user": fifth, "threshold_50pct": .5*fifth[1], "allowed": okay}
    report["allowed"] &= okay
print(json.dumps(report, indent=2))
if not report["allowed"]:
    raise SystemExit(2)
