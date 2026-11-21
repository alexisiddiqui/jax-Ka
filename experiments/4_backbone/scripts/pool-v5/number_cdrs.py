"""abdata-v5 (2026-11-16): ANARCI IMGT numbering of the pool-v5 screen sequences (Ig H/K/L only).

  python number_cdrs.py IN_FASTA OUT_JSON NCPU
Writes {"numbered": {key: [domain]}, "failed": [keys without an Ig domain], "versions": ...}; a domain is
{chain_type, species, evalue, bitscore, start, end, cdr1/cdr2/cdr3: [[imgt position, insertion, aa], ...]} with IMGT
CDR1 27-38, CDR2 56-65, CDR3 105-117 (gap '-' positions dropped). Runs in the anarci-v1 env (HMMER on PATH).
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

IMGT = {"cdr1": (27, 38), "cdr2": (56, 65), "cdr3": (105, 117)}


def read_fasta(path):
    out = {}; key = None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"): key = line[1:]; out[key] = ""
        elif key: out[key] += line
    return out


def chunk(items):
    from anarci import anarci
    numbering, alignment, _ = anarci(items, scheme="imgt", allowed_species=None, allow={"H", "K", "L"}, assign_germline=False, ncpu=1)
    out = {}
    for (key, _), nums, alis in zip(items, numbering, alignment):
        domains = []
        for (num, start, end), ali in zip(nums or [], alis or []):
            d = {"chain_type": ali["chain_type"], "species": ali.get("species"), "evalue": float(ali.get("evalue", float("nan"))),
                 "bitscore": float(ali.get("bitscore", float("nan"))), "start": int(start), "end": int(end)}
            for c, (lo, hi) in IMGT.items():
                d[c] = [[int(p[0]), p[1].strip(), a] for p, a in num if lo <= p[0] <= hi and a != "-"]
            domains.append(d)
        out[key] = domains
    return out


def main():
    src, dest, ncpu = sys.argv[1], sys.argv[2], int(sys.argv[3]); t = time.time()
    seqs = read_fasta(src); items = sorted(seqs.items()); parts = [items[i:i + 50] for i in range(0, len(items), 50)]
    numbered = {}
    with ProcessPoolExecutor(ncpu) as pool:
        for k, res in enumerate(pool.map(chunk, parts)):
            numbered.update(res)
            if (k + 1) % 20 == 0: print(json.dumps({"done": len(numbered), "of": len(items), "s": round(time.time() - t)}), flush=True)
    import anarci, importlib.metadata as md, subprocess
    hmmer = subprocess.run(["hmmscan", "-h"], capture_output=True, text=True).stdout.splitlines()[1]
    out = {"numbered": {k: v for k, v in numbered.items() if v}, "failed": sorted(k for k, v in numbered.items() if not v),
           "sequences": len(items), "versions": {"anarci": md.version("anarci"), "hmmer": hmmer.strip("# ")},
           "scheme": "imgt", "allow": ["H", "K", "L"], "cdr_ranges": IMGT, "seconds": round(time.time() - t)}
    with open(dest + ".tmp", "w") as f: json.dump(out, f)
    import os; os.replace(dest + ".tmp", dest)
    print(json.dumps({k: v for k, v in out.items() if k not in ("numbered", "failed")} | {"numbered": len(out["numbered"]), "failed": len(out["failed"])}))


if __name__ == "__main__":
    main()
