# sand.py KEY [dL] : bunkers = bright sand blobs in the cached photo near the holes -> ev/KEY-sand.json + ev/KEY-sand.jpg
import sys,json,cv2,numpy as np; sys.path.insert(0,'/tmp/claude-0/xs'); import xsec
from shapely.geometry import Polygon,LineString
k=sys.argv[1]; dL=float(sys.argv[2]) if len(sys.argv)>2 else 30
d,(x0,y1),lab,tex,im=xsec.setup(k); L=lab[...,0]; A=lab[...,1]-128; B=lab[...,2]-128
m=((L>np.median(L)+dL)&(B>4)&(B<45)&(np.abs(A)<14)).astype(np.uint8)
m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
H=[LineString(h['pts']) for h in d['holes'] if len(h['pts'])>1]
cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); out=[]; o=im.copy()
for c in cs:
    a=cv2.contourArea(c)*.09
    if a<12 or a>2500: continue
    q=Polygon([(x0+x*.3,y1-y*.3) for x,y in cv2.approxPolyDP(c,1.2,True)[:,0]]).buffer(0)
    if q.is_empty or q.geom_type!='Polygon' or 2*q.area/q.length<1.8 or (lambda r:max(r[1])/max(1e-3,min(r[1])))(cv2.minAreaRect(c))>3.5 or min(h.distance(q) for h in H)>45: continue
    out.append([round(t,1) for xy in q.exterior.coords for t in xy]); cv2.drawContours(o,[c],-1,(255,0,255),3)
json.dump(out,open(f'ev/{k}-sand.json','w')); print(len(out),'sand bunkers')
s=1600/max(o.shape[:2]); cv2.imwrite(f'ev/{k}-sand.jpg',cv2.resize(o,None,fx=s,fy=s))
