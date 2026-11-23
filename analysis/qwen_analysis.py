"""Qwen3.8-27B-judged analysis for the paper (6,000-reply expansion sample).
Run from the repository root: python3 analysis/qwen_analysis.py
Writes results/paper_cells.csv, paper_differences.csv, paper_categories.csv.
Rogan--Gladen uses the judge's held-out sensitivity/specificity (0.593/0.921).
"""
import json,csv,collections,random,math,sys,os
os.makedirs("results", exist_ok=True)
from statistics import NormalDist
Z=NormalDist().inv_cdf
SE,SP=0.593,0.921
CLAR={'open_request','restricted_request','candidate_check'}
man={}
for r in csv.DictReader(open('data/qwen_judge/qwen_judge_expand6k_manifest.csv')):
    man[(r['listener'],r['condition'],r['item_id'])]=r
rows={}
for l in open('data/qwen_judge/codings_qwen_expand6k.jsonl'):
    if not l.strip(): continue
    c=json.loads(l); k=(c['listener'],c['condition'],c['item_id'])
    if c['failed']: continue
    rows[k]=c
cells=sorted(set((k[0],k[1]) for k in man))
# per cell per child: counts
data=collections.defaultdict(lambda: collections.defaultdict(lambda:[0,0,0,0,0,0]))  # cell->child->[hu,nu,hc,nc,ce,ne]
cats=collections.defaultdict(collections.Counter)
for k,c in rows.items():
    m=man[k]; cell=(k[0],k[1]); t=m['turn_type']; cl=c['category'] in CLAR
    cats[cell][c['category']]+=1
    v=data[cell][m['child']]
    if t=='unintelligible': v[0]+=cl; v[1]+=1
    elif t=='quiet_baseline': v[2]+=cl; v[3]+=1
    elif t=='error': v[4]+=bool(c['corrects_error']); v[5]+=1
def rate(k,n,corr=False):
    p=k/n
    if corr: p=min(1,max(0,(p+SP-1)/(SE+SP-1))); k=p*n
    if k<=0 or k>=n: p=(k+0.5)/(n+1)
    return p
def sdt(v,corr=False):
    hu,nu,hc,nc=v[0],v[1],v[2],v[3]
    H=rate(hu,nu,corr);F=rate(hc,nc,corr)
    return Z(H)-Z(F), -0.5*(Z(H)+Z(F)), H, F
def agg(cell,kids):
    s=[0]*6
    for ch in kids:
        v=data[cell].get(ch)
        if v:
            for i in range(6): s[i]+=v[i]
    return s
allkids=sorted(set(m['child'] for m in man.values()))
point={cell:agg(cell,allkids) for cell in cells}
random.seed(42)
B=1000
boots=[]
for b in range(B):
    ks=[random.choice(allkids) for _ in allkids]
    boots.append({cell:agg(cell,ks) for cell in cells})
def ci(xs):
    xs=sorted(xs); return xs[int(.025*len(xs))], xs[int(.975*len(xs))-1]
out={}
for cell in cells:
    v=point[cell]
    d,c,H,F=sdt(v); dc,cc,Hc,Fc=sdt(v,True)
    bd=[sdt(bb[cell])[0] for bb in boots]; bc=[sdt(bb[cell])[1] for bb in boots]
    out[cell]=dict(H=H*100 if 0<v[0]<v[1] else v[0]/v[1]*100,F=v[2]/v[3]*100,d=d,c=c,dci=ci(bd),cci=ci(bc),dcor=dc,ccor=cc,Hc=Hc*100,Fc=Fc*100,nu=v[1],nc=v[3],ne=v[5],corr=v[4]/v[5]*100 if v[5] else None)
    o=out[cell]
    print(f"{cell[0][:28]:28s} {cell[1]:6s} H={v[0]/v[1]*100:5.1f} F={o['F']:5.1f} d={d:5.2f} [{o['dci'][0]:.2f},{o['dci'][1]:.2f}] c={c:5.2f} [{o['cci'][0]:.2f},{o['cci'][1]:.2f}] | corr H={o['Hc']:5.1f} F={o['Fc']:5.1f} d={dc:5.2f} c={cc:5.2f} | nU={v[1]} nC={v[3]} nE={v[5]} corrErr={o['corr']:.1f}")
def diff(a,b,idx,corr):
    p=sdt(point[a],corr)[idx]-sdt(point[b],corr)[idx]
    bs=[sdt(bb[a],corr)[idx]-sdt(bb[b],corr)[idx] for bb in boots]
    return p,ci(bs)
H=('human','human')
models=sorted(set(c[0] for c in cells if c[0]!='human'))
print('\n== model minus adults')
for m in models:
  for cond in ['plain','child','repair']:
    s=[]
    for idx,nm in [(0,'d'),(1,'c')]:
        for corr in [False,True]:
            p,(lo,hi)=diff((m,cond),H,idx,corr)
            s.append(f"{nm}{'C' if corr else 'R'}={p:+.2f}[{lo:+.2f},{hi:+.2f}]{'*' if lo>0 or hi<0 else ' '}")
    print(f"{m[:24]:24s} {cond:6s}", ' '.join(s))
print('\n== repair minus plain')
for m in models:
    s=[]
    for idx,nm in [(0,'d'),(1,'c')]:
        for corr in [False,True]:
            p,(lo,hi)=diff((m,'repair'),(m,'plain'),idx,corr)
            s.append(f"{nm}{'C' if corr else 'R'}={p:+.2f}[{lo:+.2f},{hi:+.2f}]{'*' if lo>0 or hi<0 else ' '}")
    print(f"{m[:24]:24s}", ' '.join(s))
print('\n== categories %')
order=['open_request','restricted_request','candidate_check','reformulation','echo_error','topic_question','guess_proceed','acknowledge','ignore_shift']
for cell in cells:
    n=sum(cats[cell].values())
    print(f"{cell[0][:24]:24s} {cell[1]:6s}", ' '.join(f"{cats[cell][o]/n*100:5.1f}" for o in order), n)
tot=collections.Counter()
for cell in cells:
    if cell[0]!='human': tot.update(cats[cell])

OUT='results/'
with open(OUT+'paper_cells.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['listener','condition','hit','fa','d','c','d_lo','d_hi','c_lo','c_hi','hit_corr','fa_corr','d_corr','c_corr','bound','n_unint','n_quiet','n_error','corrects_error'])
    for cell in cells:
        o=out[cell]; v=point[cell]
        hb=v[0]/v[1]; fb=v[2]/v[3]
        bound=any(x>=SE or x<=1-SP for x in (hb,fb))
        w.writerow([*cell,round(hb*100,1),round(fb*100,1),round(o['d'],3),round(o['c'],3),round(o['dci'][0],3),round(o['dci'][1],3),round(o['cci'][0],3),round(o['cci'][1],3),round(o['Hc'],1),round(o['Fc'],1),round(o['dcor'],3),round(o['ccor'],3),int(bound),v[1],v[3],v[5],round(o['corr'],1)])
with open(OUT+'paper_differences.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['contrast','listener','condition','metric','arm','estimate','lo','hi'])
    for m in models:
        for cond in ['plain','child','repair']:
            for idx,nm in [(0,'d'),(1,'c')]:
                for corr in [False,True]:
                    p,(lo,hi)=diff((m,cond),H,idx,corr); w.writerow(['model_minus_adults',m,cond,nm,'corrected' if corr else 'raw',round(p,3),round(lo,3),round(hi,3)])
        for idx,nm in [(0,'d'),(1,'c')]:
            for corr in [False,True]:
                p,(lo,hi)=diff((m,'repair'),(m,'plain'),idx,corr); w.writerow(['repair_minus_plain',m,'repair',nm,'corrected' if corr else 'raw',round(p,3),round(lo,3),round(hi,3)])
with open(OUT+'paper_categories.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['listener','condition',*order,'n'])
    for cell in cells:
        n=sum(cats[cell].values()); w.writerow([*cell,*[round(cats[cell][o]/n*100,1) for o in order],n])
