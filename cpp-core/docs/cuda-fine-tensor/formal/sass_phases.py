"""Stall samples and instructions of a kernel's SASS csv per barrier-delimited phase (not formal): python sass_phases.py file.csv"""
import csv, collections, sys
rows = list(csv.reader(open(sys.argv[1], encoding="utf-8", errors="replace")))
h = rows[1]; ii = h.index("Instructions Executed"); si = h.index("Source"); wi = h.index("Warp Stall Sampling (All Samples)")
reg = 0; S = collections.Counter(); I = collections.Counter(); kinds = collections.defaultdict(collections.Counter); top = collections.defaultdict(list)
for r in rows[2:]:
    if len(r) <= wi or not r[wi].isdigit(): continue
    src = r[si].strip(); w = int(r[wi]); n = int(r[ii])
    S[reg] += w; I[reg] += n
    op = (src.split()[1] if src.startswith("@") else src.split()[0]).split(".")[0]
    kinds[reg][op] += n
    top[reg].append((w, src[:58], n))
    if "BAR.SYNC" in src: reg += 1
tot = sum(S.values()); ti = sum(I.values())
print("samples", tot, "instr", ti)
for k in sorted(S):
    print(k, "samples %6d %.1f%%  instr %10d %.1f%%" % (S[k], 100 * S[k] / tot, I[k], 100 * I[k] / ti), kinds[k].most_common(8))
    print("    top", sorted(top[k], reverse=True)[:6])
