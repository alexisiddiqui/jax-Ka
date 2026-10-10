"""Summarise an nsys capture of scripts/profile_train_step.py (2026-10-10).

Joins the nsys GPU kernel summary with the optimized HLO dumps (XLA names a fusion's kernel after the fusion
instruction, '.' -> '_'), so each kernel gets its JAX primitive, forward/backward direction and source line. Writes
report.json and prints the time split by kernel class, direction, source line and the top kernels.

  python scripts/profile_report.py OUT        (OUT holds profile.nsys-rep, hlo-*.txt, window-profiled.json)
"""
import csv
import io
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

INSTRUCTION = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=.*?metadata=\{([^}]*)\}")


def stats(rep, report):
    text = subprocess.run(["nsys", "stats", "--report", report, "--format", "csv", "--output", "-", str(rep)],
                          capture_output=True, text=True, check=True).stdout
    lines = text[text.index('"') if '"' in text else 0:].splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"Time') or l.startswith("Time"))
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def hlo_map(folder):
    mapping = {}
    for path in sorted(folder.glob("hlo-*.txt")):
        for line in path.read_text().splitlines():
            m = INSTRUCTION.match(line)
            if not m: continue
            name, meta = m.group(1), m.group(2)
            op = re.search(r'op_name="([^"]*)"', meta); src = re.search(r'source_file="([^"]*)"', meta); ln = re.search(r"source_line=(\d+)", meta)
            mapping.setdefault(name.replace(".", "_").replace("-", "_"), {
                "op_name": op.group(1) if op else "", "file": Path(src.group(1)).name if src else "", "line": int(ln.group(1)) if ln else 0,
                "path": src.group(1) if src else ""})
    return mapping


def kernel_class(name):
    n = name.lower()
    for key, label in (("indexed_attention", "triton indexed attention"), ("gemm", "gemm/matmul"), ("xmma", "gemm/matmul"),
                       ("cutlass", "gemm/matmul"), ("cublas", "gemm/matmul"), ("triton", "triton (xla)"), ("scatter", "scatter"),
                       ("gather", "gather"), ("reduce", "reduction"), ("transpose", "transpose/copy"), ("copy", "transpose/copy"),
                       ("concatenate", "concat"), ("loop", "elementwise loop"), ("input", "reduction"), ("memset", "memset")):
        if key in n: return label
    return "other"


def main():
    folder = Path(sys.argv[1]); rep = folder / "profile.nsys-rep"
    kernels = stats(rep, "cuda_gpu_kern_sum"); memory = stats(rep, "cuda_gpu_mem_time_sum")
    window = json.loads((folder / "window-profiled.json").read_text()); mapping = hlo_map(folder)
    rows = []; source_cache = {}
    for k in kernels:
        name = k["Name"]; total = float(k["Total Time (ns)"]) / 1e9; base = re.split(r"[(<\s]", name)[0]
        info = mapping.get(base) or mapping.get(re.sub(r"_\d+$", "", base)) or {}
        line_text = ""
        if info.get("path") and info.get("line"):
            try:
                lines = source_cache.setdefault(info["path"], Path(info["path"]).read_text().splitlines()); line_text = lines[info["line"] - 1].strip()
            except OSError: pass
        op = info.get("op_name", "")
        rows.append({"kernel": base[:80], "seconds": total, "instances": int(k["Instances"]), "avg_us": float(k["Avg (ns)"]) / 1e3,
                     "class": kernel_class(name), "direction": "backward" if "transpose(" in op else ("forward" if op else "unmapped"),
                     "primitive": op.split("/")[-1] if op else "", "source": f'{info.get("file", "")}:{info.get("line", "")}' if info else "",
                     "code": line_text[:110]})
    kernel_total = sum(r["seconds"] for r in rows); memcpy = sum(float(m["Total Time (ns)"]) / 1e9 for m in memory)
    launches = sum(r["instances"] for r in rows); wall = window["wall_seconds"]

    def split(key):
        d = defaultdict(float)
        for r in rows: d[r[key]] += r["seconds"]
        return sorted(({"key": k, "seconds": round(v, 4), "share": round(v / kernel_total, 4)} for k, v in d.items()), key=lambda x: -x["seconds"])
    by_source = defaultdict(lambda: {"seconds": 0.0, "code": "", "instances": 0})
    for r in rows:
        s = by_source[r["source"] or r["class"]]; s["seconds"] += r["seconds"]; s["code"] = s["code"] or r["code"]; s["instances"] += r["instances"]
    report = {"steps": window["steps"], "wall_seconds": wall, "kernel_seconds": kernel_total, "memcpy_seconds": memcpy,
              "gpu_busy_fraction": kernel_total / wall, "kernel_launches": launches, "launches_per_step": launches / window["steps"],
              "mean_kernel_us": kernel_total / launches * 1e6,
              "kernels_under_10us_share_of_launches": sum(r["instances"] for r in rows if r["avg_us"] < 10) / launches,
              "kernels_under_10us_share_of_time": sum(r["seconds"] for r in rows if r["avg_us"] < 10) / kernel_total,
              "by_class": split("class"), "by_direction": split("direction"),
              "by_source": sorted(({"source": k, **{kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()},
                                    "share": round(v["seconds"] / kernel_total, 4)} for k, v in by_source.items()), key=lambda x: -x["seconds"])[:25],
              "top_kernels": sorted(rows, key=lambda r: -r["seconds"])[:30]}
    (folder / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("by_source", "top_kernels")}, indent=1))
    for s in report["by_source"]: print(f'{s["share"]:6.1%} {s["seconds"]:8.3f}s {s["instances"]:7d}  {s["source"]:28s} {s["code"]}')
    for r in report["top_kernels"][:20]: print(f'{r["seconds"] / kernel_total:6.1%} {r["instances"]:6d} {r["avg_us"]:9.1f}us {r["class"]:24s} {r["direction"]:9s} {r["primitive"]:22s} {r["source"]:24s} {r["kernel"][:50]}')


if __name__ == "__main__":
    main()
