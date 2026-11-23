"""Second-judge check: Gemma 4 26B-A4B codes of the 6,000-reply sample (Gemma's own
replies excluded). Same child bootstrap (B=1000, seed 42, same child list) as
qwen_analysis.py, so Qwen- and Gemma-judged contrasts share resamples.
Run from the repository root: python3 analysis/gemma_judge_analysis.py
Writes results/paper_second_judge.csv (raw rates only).
"""
import json,csv,collections,random,os
os.makedirs("results", exist_ok=True)
from statistics import NormalDist
Z=NormalDist().inv_cdf
CLAR={'open_request','restricted_request','candidate_check'}
man={}
for r in csv.DictReader(open('data/qwen_judge/qwen_judge_expand6k_manifest.csv')):
    man[(r['listener'],r['condition'],r['item_id'])]=r
def load(path):
    rows={}
    for l in open(path):
        if not l.strip(): continue
        c=json.loads(l)
        if c.get('failed'): continue
        rows[(c['listener'],c['condition'],c['item_id'])]=c
    return rows
judges={'qwen':load('data/qwen_judge/codings_qwen_expand6k.jsonl'),
        'gemma':load('data/gemma_judge/gemma_judge_codes.jsonl')}
GEM='google/gemma-4-26b-a4b-qat'
data={j:collections.defaultdict(lambda: collections.defaultdict(lambda:[0,0,0,0])) for j in judges}
common=set(judges['gemma'])&set(judges['qwen'])
for j,rows in judges.items():
    for k in common:
        if k[0]==GEM: continue
        c=rows[k]; m=man[k]; t=m['turn_type']; cl=c['category'] in CLAR
        v=data[j][(k[0],k[1])][m['child']]
        if t=='unintelligible': v[0]+=cl; v[1]+=1
        elif t=='quiet_baseline': v[2]+=cl; v[3]+=1
cells=sorted(set((k[0],k[1]) for k in common if k[0]!=GEM))
def rate(k,n):
    return (k+0.5)/(n+1) if k<=0 or k>=n else k/n
def sdt(v):
    H=rate(v[0],v[1]);F=rate(v[2],v[3]); return Z(H)-Z(F),-0.5*(Z(H)+Z(F))
allkids=sorted(set(m['child'] for m in man.values()))
def agg(j,cell,kids):
    s=[0]*4
    for ch in kids:
        v=data[j][cell].get(ch)
        if v:
            for i in range(4): s[i]+=v[i]
    return s
point={j:{c:agg(j,c,allkids) for c in cells} for j in judges}
random.seed(42); boots=[]
for b in range(1000):
    ks=[random.choice(allkids) for _ in allkids]
    boots.append({j:{c:agg(j,c,ks) for c in cells} for j in judges})
def ci(xs):
    xs=sorted(xs); return xs[25],xs[974]
rows=[]
for j in judges:
    for c in cells:
        v=point[j][c]; d,cc=sdt(v); lo,hi=ci([sdt(b[j][c])[0] for b in boots])
        rows.append(['cell',j,c[0],c[1],'d',round(d,3),round(lo,3),round(hi,3),round(v[0]/v[1]*100,1),round(v[2]/v[3]*100,1),v[1],v[3]])
H=('human','human'); models=sorted(set(c[0] for c in cells if c[0]!='human'))
for j in judges:
    for m in models:
        for cond in ['plain','child','repair']:
            for idx,nm in [(0,'d'),(1,'c')]:
                p=sdt(point[j][(m,cond)])[idx]-sdt(point[j][H])[idx]
                lo,hi=ci([sdt(b[j][(m,cond)])[idx]-sdt(b[j][H])[idx] for b in boots])
                rows.append(['model_minus_adults',j,m,cond,nm,round(p,3),round(lo,3),round(hi,3),'','','',''])
        for idx,nm in [(0,'d'),(1,'c')]:
            p=sdt(point[j][(m,'repair')])[idx]-sdt(point[j][(m,'plain')])[idx]
            lo,hi=ci([sdt(b[j][(m,'repair')])[idx]-sdt(b[j][(m,'plain')])[idx] for b in boots])
            rows.append(['repair_minus_plain',j,m,'repair',nm,round(p,3),round(lo,3),round(hi,3),'','','',''])
with open('results/paper_second_judge.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['contrast','judge','listener','condition','metric','estimate','lo','hi','hit','fa','n_unint','n_quiet'])
    w.writerows(rows)
for r in rows: print(*r)
print('replies compared:',sum(1 for k in common if k[0]!=GEM))
