"""Error-turn statistics (Qwen3.8 judge): clarification, correction and reformulation
rates at error turns, with 95% child-bootstrap intervals (1,000 resamples, seed 42).
Run from the repository root. Writes results/paper_error_turns.csv."""
import json, csv, collections, random, os
os.makedirs("results", exist_ok=True)
CLAR = {'open_request', 'restricted_request', 'candidate_check'}
man = {(r['listener'], r['condition'], r['item_id']): r for r in csv.DictReader(open('data/qwen_judge/qwen_judge_expand6k_manifest.csv'))}
per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0, 0]))  # clar, corrects, reform, n
for l in open('data/qwen_judge/codings_qwen_expand6k.jsonl'):
    if not l.strip(): continue
    c = json.loads(l)
    if c['failed']: continue
    k = (c['listener'], c['condition'], c['item_id']); m = man[k]
    if m['turn_type'] != 'error': continue
    v = per[(k[0], k[1])][m['child']]
    v[0] += c['category'] in CLAR; v[1] += bool(c['corrects_error']); v[2] += c['category'] == 'reformulation'; v[3] += 1
kids = sorted(set(m['child'] for m in man.values()))
random.seed(42)
boots = [[random.choice(kids) for _ in kids] for _ in range(1000)]
def agg(cell, ks):
    s = [0, 0, 0, 0]
    for ch in ks:
        v = per[cell].get(ch)
        if v: s = [a + b for a, b in zip(s, v)]
    return s
def ci(x): x = sorted(x); return x[25], x[974]
with open('results/paper_error_turns.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['listener', 'condition', 'n', 'metric', 'rate', 'lo', 'hi'])
    for cell in sorted(per):
        p = agg(cell, kids); bs = [agg(cell, k) for k in boots]
        for i, name in enumerate(['clarification', 'corrects_error', 'reformulation']):
            lo, hi = ci([b[i] / b[3] for b in bs])
            w.writerow([*cell, p[3], name, round(p[i] / p[3] * 100, 1), round(lo * 100, 1), round(hi * 100, 1)])
            print(cell[0][:14], cell[1], name, round(p[i] / p[3] * 100, 1), round(lo * 100, 1), round(hi * 100, 1))
