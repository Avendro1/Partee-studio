import json,glob,sys
from shapely.geometry import Polygon,LineString
from shapely.ops import unary_union, substring
pas=set(open('/tmp/claude-0/gh/p5/pass.txt').read().split())|set('royal-portrush-dunluce druids-glen portmarnock mount-juliet k-palmer-north waterville carne cork county-longford ballymena'.split())
tot=0;bad={}
for f in sorted(glob.glob('*.json')):
  d=json.load(open(f)); k=d['key']
  fw=[Polygon(list(zip(c[0::2],c[1::2]))).buffer(0) for t,c in d['feats'] if t=='fairway']
  U=unary_union(fw).buffer(8) if fw else None
  for h in d['holes']:
    if len(h['pts'])<2: continue
    L=LineString(h['pts']); par=int(str(h.get('par') or (3 if L.length<215 else 4))[0])
    if par==3: continue
    mid=substring(L,L.length*.2,L.length*.85)
    cov=mid.intersection(U).length/mid.length if U else 0
    tot+=1
    if cov<.5: bad.setdefault(k,[]).append((h.get('ref'),round(cov,2)))
print('par4/5 holes',tot,'flagged',sum(map(len,bad.values())))
for k,v in bad.items(): print('PASS' if k in pas else 'fail',k,v)
