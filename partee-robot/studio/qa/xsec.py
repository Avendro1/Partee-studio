# xsec.py KEY REF [eval] : fairway from cross-sections of the cached satellite photo along the hole line
import sys, json, math, numpy as np, cv2
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
MPP=.3; import os; CT=float(os.environ.get('CT',14)); TT=float(os.environ.get('TT',1.8))
def setup(key):
    d=json.load(open(f'/home/user/partee-studio/partee-site/courses/{key}.json'))
    x0,y0,x1,y1=unary_union([LineString(h['pts']).buffer(110) for h in d['holes'] if len(h['pts'])>1]).bounds
    im=cv2.imread(f'/tmp/claude-0/osm/sat/{key}-0.3.png')
    lab=cv2.cvtColor(cv2.GaussianBlur(im,(0,0),1.5),cv2.COLOR_BGR2LAB).astype(np.float32)
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex=np.sqrt(np.maximum(cv2.GaussianBlur(g*g,(0,0),3)-cv2.GaussianBlur(g,(0,0),3)**2,0))
    return d,(x0,y1),lab,tex,im
def trace(d,o,lab,tex,h,a0=.12,a1=.93,half=45):
    x0,y1=o; L=LineString(h['pts']); n=max(4,int(L.length*(a1-a0)/4))
    def pix(x,y): return (x-x0)/MPP,(y1-y)/MPP
    Lf,Rt=[],[]
    for i in range(n):
        s=L.length*(a0+(a1-a0)*i/(n-1)); p=L.interpolate(s); q=L.interpolate(min(L.length,s+3)); r=L.interpolate(max(0,s-3))
        dx,dy=q.x-r.x,q.y-r.y; l=math.hypot(dx,dy) or 1; nx,ny=-dy/l,dx/l
        ts=np.arange(-half,half+.01,.6)
        X=np.array([pix(p.x+nx*t,p.y+ny*t) for t in ts],np.float32)
        mp=X[:,0].reshape(1,-1); mq=X[:,1].reshape(1,-1)
        c=cv2.remap(lab,mp,mq,cv2.INTER_LINEAR)[0]; tx=cv2.remap(tex,mp,mq,cv2.INTER_LINEAR)[0]
        Lf.append((p,nx,ny,ts,c,tx))
    # reference fairway colour/texture: median over the central 6 m of all sections
    C=np.array([c[np.abs(ts)<3].mean(0) for p,nx,ny,ts,c,tx in Lf]); Tm=np.median([tx[np.abs(ts)<3].mean() for *_,ts,c,tx in [(0,0,0,a[3],a[4],a[5]) for a in Lf]])
    ref=np.median(C,0)
    pts_l,pts_r,w=[],[],[]
    for p,nx,ny,ts,c,tx in Lf:
        ok=(np.linalg.norm(c-ref,axis=1)<CT)&(tx<max(Tm*TT,Tm+3))
        ok=cv2.morphologyEx(ok.astype(np.uint8)[None],cv2.MORPH_CLOSE,np.ones((1,5),np.uint8))[0].astype(bool)
        # nearest run of ok cells to the centre (allow 6 m offset)
        ci=np.argmin(np.abs(ts)); idx=[j for j in range(len(ts)) if ok[j] and abs(ts[j])<6]
        if not idx: w.append(None); continue
        j=min(idx,key=lambda j:abs(ts[j])); a=j
        while a>0 and ok[a-1]: a-=1
        b=j
        while b<len(ts)-1 and ok[b+1]: b+=1
        if ts[b]-ts[a]<8 or ts[b]-ts[a]>70: w.append(None); continue
        w.append((ts[a],ts[b]))
    ok=[x for x in w if x]
    if len(ok)<.6*len(w): return None,len(ok)/len(w)
    # smooth edges with a running median, skip gaps
    A=np.array([x[0] if x else np.nan for x in w]); B=np.array([x[1] if x else np.nan for x in w])
    def sm(v):
        out=v.copy()
        for i in range(len(v)):
            win=v[max(0,i-3):i+4]; win=win[~np.isnan(win)]
            out[i]=np.median(win) if len(win) else np.nan
        return out
    A,B=sm(A),sm(B); left,right=[],[]
    for (p,nx,ny,*_),a,b in zip(Lf,A,B):
        if np.isnan(a): continue
        left.append((p.x+nx*a,p.y+ny*a)); right.append((p.x+nx*b,p.y+ny*b))
    poly=Polygon(left+right[::-1]).buffer(1.5).buffer(-1.5)
    if poly.geom_type!='Polygon': poly=max(poly.geoms,key=lambda g:g.area)
    return poly.simplify(.8),len(ok)/len(w)
if __name__=='__main__':
    key=sys.argv[1]; d,o,lab,tex,im=setup(key)
    refs=sys.argv[2].split(',') if len(sys.argv)>2 and sys.argv[2]!='all' else [str(h['ref']) for h in d['holes']]
    ev='eval' in sys.argv
    F=unary_union([Polygon(list(zip(c[0::2],c[1::2]))).buffer(0) for k,c in d['feats'] if k=='fairway' and len(c)>=6])
    res={}
    for h in d['holes']:
        if str(h['ref']) not in refs or len(h['pts'])<2 or int(str(h.get('par') or 4)[0])<4: continue
        poly,cov=trace(d,o,lab,tex,h)
        if poly is None: print(h['ref'],'no trace',round(cov,2)); continue
        res[str(h['ref'])]=[round(v,1) for xy in poly.exterior.coords for v in xy]
        if ev:
            L=LineString(h['pts']); T=F.intersection(L.buffer(45))
            iou=poly.intersection(T).area/max(1,poly.union(T).area); print(h['ref'],'iou',round(iou,2),'cov',round(cov,2))
        else: print(h['ref'],'ok',round(cov,2),int(poly.area))
    json.dump(res,open(f'/tmp/claude-0/xs/{key}.json','w'))
