# bcon.py KEY [apply] : bunkers whose brightness barely differs from the ground around them (tan meadow, dry rough) -> list / remove; overlay ev/KEY-bc.jpg
import sys,json,cv2,numpy as np; sys.path.insert(0,'/tmp/claude-0/xs'); import xsec
from shapely.geometry import Polygon
k=sys.argv[1]; d,(x0,y1),lab,tex,im=xsec.setup(k); L=lab[...,0]
P=lambda c:Polygon(list(zip(c[0::2],c[1::2]))).buffer(0)
def msk(g):
    m=np.zeros(L.shape,np.uint8)
    for p in getattr(g,'geoms',[g]):
        if p.geom_type=='Polygon' and not p.is_empty: cv2.fillPoly(m,[np.array([((x-x0)/.3,(y1-y)/.3) for x,y in p.exterior.coords],np.int32)],1)
    return m>0
bad=[]
for c in [c for kk,c in d['feats'] if kk=='bunker']:
    q=P(c); a=msk(q); r=msk(q.buffer(7).difference(q.buffer(1.5)))
    if a.sum()<5 or r.sum()<5: continue
    T=float(sys.argv[2]) if len(sys.argv)>2 else 12
    rr=q.minimum_rotated_rectangle; xs=list(rr.exterior.coords); import math; A_=math.dist(xs[0],xs[1]); B_=math.dist(xs[1],xs[2])
    thin=2*q.area/max(q.length,1e-3)<1.0 or max(A_,B_)/max(min(A_,B_),1e-3)>4.5
    if L[a].mean()-L[r].mean()<T or thin: bad.append(c)
print(len(bad),'low-contrast of',sum(1 for kk,c in d['feats'] if kk=='bunker'))
px=lambda c:np.array([((x-x0)/.3,(y1-y)/.3) for x,y in zip(c[0::2],c[1::2])],np.int32)
for c in bad: cv2.polylines(im,[px(c)],True,(255,0,255),4)
s=1600/max(im.shape[:2]); cv2.imwrite(f'ev/{k}-bc.jpg',cv2.resize(im,None,fx=s,fy=s))
json.dump(bad,open(f'ev/{k}-bc.json','w'))
