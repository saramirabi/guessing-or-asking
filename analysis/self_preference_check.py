"""Human self-preference check: one blind annotator (labels_annotator.csv) on 150 sample
replies (100 Qwen3.8, 50 other sources), compared with the Qwen3.8 and Gemma 4 judges.
Run from the repository root: python3 analysis/self_preference_check.py
Writes results/paper_self_preference.csv. Item bootstrap B=1000, seed 42.
"""
import csv,json,random,os
os.makedirs("results", exist_ok=True)
CLAR={'open_request','restricted_request','candidate_check'}
QW='qwen/qwen3.8-27b'
key={r['annotation_id']:r for r in csv.DictReader(open('data/human_check/self_preference_key.csv'))}
lab={r['annotation_id']:r for r in csv.DictReader(open('data/human_check/labels_annotator.csv',encoding='utf-8-sig')) if r['category']}
def load(p):
    o={}
    for l in open(p):
        if l.strip():
            c=json.loads(l)
            if not c.get('failed'): o[(c['listener'],c['condition'],c['item_id'])]=c
    return o
J={'qwen':load('data/qwen_judge/codings_qwen_expand6k.jsonl'),'gemma':load('data/gemma_judge/gemma_judge_codes.jsonl')}
items=[]
for a,k in key.items():
    if a not in lab: continue
    t=(k['listener'],k['condition'],k['item_id'])
    items.append(dict(id=a,qwen_src=k['listener']==QW,turn=k['turn_type'],h=lab[a]['category'],
        q=J['qwen'][t]['category'] if t in J['qwen'] else None,g=J['gemma'][t]['category'] if t in J['gemma'] else None,
        hce=lab[a]['corrects_error'].strip().lower()=='true',qce=bool(J['qwen'].get(t,{}).get('corrects_error'))))
def kappa(a,b):
    n=len(a); po=sum(x==y for x,y in zip(a,b))/n
    cats=set(a)|set(b); pe=sum((a.count(c)/n)*(b.count(c)/n) for c in cats)
    return (po-pe)/(1-pe) if pe<1 else float('nan')
def stats(rows,j):
    j={'qwen':'q','gemma':'g'}[j]
    rows=[r for r in rows if r[j] is not None]
    h=[r['h'] in CLAR for r in rows]; p=[r[j] in CLAR for r in rows]
    tp=sum(x and y for x,y in zip(h,p)); pos=sum(h); neg=len(h)-pos; tn=sum((not x) and (not y) for x,y in zip(h,p))
    return dict(n=len(rows),human_rate=pos/len(rows),judge_rate=sum(p)/len(rows),excess=(sum(p)-pos)/len(rows),
        se=tp/pos if pos else float('nan'),sp=tn/neg if neg else float('nan'),
        kclar=kappa(h,p),kcat=kappa([r['h'] for r in rows],[r[j] for r in rows]))
Q=[r for r in items if r['qwen_src']]; O=[r for r in items if not r['qwen_src']]
random.seed(42); B=1000
out=[]
for j in ('qwen','gemma'):
    sq,so=stats(Q,j),stats(O,j)
    for nm,st in (('qwen_replies',sq),('other_replies',so)):
        for m,v in st.items(): out.append([j,nm,m,round(v,4),'',''])
    bs={m:[] for m in ('excess','se','sp','kclar')}
    for _ in range(B):
        bq=stats([random.choice(Q) for _ in Q],j); bo=stats([random.choice(O) for _ in O],j)
        for m in bs:
            d=bq[m]-bo[m]
            if d==d: bs[m].append(d)
    for m in bs:
        x=sorted(bs[m]); out.append([j,'qwen_minus_other',m,round(sq[m]-so[m],4),round(x[int(.025*len(x))],4),round(x[int(.975*len(x))-1],4)])
for r in out: print(*r)
for nm,rows in (('qwen_replies',Q),('other_replies',O)):
    e=[r for r in rows if r['turn']=='error']
    print('corrects_error',nm,len(e),'human',sum(r['hce'] for r in e),'qwen judge',sum(r['qce'] for r in e),'agree',sum(r['hce']==r['qce'] for r in e))
with open('results/paper_self_preference.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['judge','subset','metric','estimate','lo','hi']); w.writerows(out)
