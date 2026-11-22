# bigbunk.py KEY [apply]: bunker polygons over 800 m2 are almost always rough mis-tagged as sand; keep only the sand-coloured pixels inside them
import sys, json, numpy as np, cv2
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
k = sys.argv[1]; MPP = .3; F = f'/home/user/partee-studio/partee-site/courses/{k}.json'
d = json.load(open(F))
x0, y0, x1, y1 = unary_union([LineString(h['pts']).buffer(110) for h in d['holes']]).bounds
im = cv2.imread(f'/tmp/claude-0/osm/sat/{k}-0.3.png')
px = lambda q: np.stack([(np.asarray(q)[:, 0] - x0) / MPP, (y1 - np.asarray(q)[:, 1]) / MPP], 1)
met = lambda p: np.stack([x0 + p[:, 0] * MPP, y1 - p[:, 1] * MPP], 1)
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(int); lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(int)
# sand: bright, low saturation or yellowish, not green
sand = ((lab[..., 0] > 165) & (lab[..., 1] > 122) & (hsv[..., 1] < 110)).astype(np.uint8)
sand = cv2.morphologyEx(sand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
out, nb, nn = [], 0, 0; ov = im.copy()
for kk, c in d['feats']:
    if kk != 'bunker': out.append([kk, c]); continue
    p = Polygon(list(zip(c[0::2], c[1::2]))).buffer(0)
    if p.area <= 800: out.append([kk, c]); continue
    nb += 1; m = np.zeros(im.shape[:2], np.uint8)
    for g in (p.geoms if hasattr(p, 'geoms') else [p]): cv2.fillPoly(m, [px(g.exterior.coords).astype(np.int32)], 1)
    cv2.polylines(ov, [px(p.exterior.coords).astype(np.int32)] if p.geom_type == 'Polygon' else [], True, (0, 0, 255), 3)
    cn, _ = cv2.findContours(sand & m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for q in cn:
        if cv2.contourArea(q) * MPP * MPP < 15: continue
        P = Polygon(met(cv2.approxPolyDP(q, 1, True)[:, 0].astype(float))).buffer(.6).buffer(0)
        if P.geom_type != 'Polygon' or P.area > 800: continue
        out.append(['bunker', [round(v, 1) for xy in P.exterior.coords for v in xy]]); nn += 1
        cv2.polylines(ov, [px(P.exterior.coords).astype(np.int32)], True, (255, 255, 0), 3)
print(k, 'big bunkers', nb, '-> sand pockets', nn)
cv2.imwrite(f'/tmp/claude-0/fw/bb-{k}.jpg', cv2.resize(ov, None, fx=.45, fy=.45))
if len(sys.argv) > 2 and nb: d['feats'] = out; json.dump(d, open(F, 'w'), separators=(',', ':'))
