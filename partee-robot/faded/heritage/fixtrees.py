# island trees + reed-fringe cleanup for one course, against the registered satellite (sat-*/sat.jpg + sat.json from canopy.py)
import sys, json, numpy as np, cv2
from PIL import Image
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
course, alljs, satdir, out = sys.argv[1:5]
d = json.load(open(course)); A = json.load(open(alljs)); b = json.load(open(f'{satdir}/sat.json'))
a = np.asarray(Image.open(f'{satdir}/sat.jpg').convert('RGB')); H, W = a.shape[:2]
sx = W / (b['x1'] - b['x0']); sy = H / (b['y1'] - b['y0']); mpp = 1 / sx
px = lambda x, y: ((x - b['x0']) * sx, (b['y1'] - y) * sy)
hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(np.float32); v, h, s = hsv[..., 2], hsv[..., 0], hsv[..., 1]
tex = cv2.GaussianBlur(np.abs(v - cv2.GaussianBlur(v, (0, 0), 1.5)), (0, 0), 3)
tree = ((v < 92) & (h > 30) & (h < 95) & (tex > 4) & (s > 45)).astype(np.uint8)
tree = cv2.morphologyEx(tree, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
wat = ((v < 48) & (tex < 6)).astype(np.uint8)
W_ = unary_union([Polygon(list(zip(xy[::2], xy[1::2]))).buffer(0) for k, xy in d['feats'] if k == 'water'])
# 1) islands: tree canopy on the photo inside a mapped lake, away from its shore (the lake polygon covers the island)
inner = W_.buffer(-4)
isl = np.zeros((H, W), np.uint8)
for p in getattr(inner, 'geoms', [inner]):
    if p.is_empty: continue
    cv2.fillPoly(isl, [np.array([px(*q) for q in p.exterior.coords], np.int32)], 1)
isl &= tree; isl = cv2.morphologyEx(isl, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(isl)
islands = []; new = []
for i in range(1, n):
    if st[i, cv2.CC_STAT_AREA] * mpp * mpp < 25: continue   # >= 25 m2 of canopy = a real island copse
    m = (lab == i).astype(np.uint8)
    yy_, xx_ = np.nonzero(m); cpt = Point(b['x0'] + xx_.mean() / sx, b['y1'] - yy_.mean() / sy)
    host = [q for q in getattr(W_, 'geoms', [W_]) if q.contains(cpt)]
    if not host or 2 * host[0].area / host[0].length < 12: continue   # islands only in real lakes (mean width > 12 m), never trees overhanging a stream
    cs, _ = cv2.findContours(cv2.dilate(m, np.ones((5, 5), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)[:, 0, :]
    islands.append([[round(b['x0'] + x / sx, 1), round(b['y1'] - y / sy, 1)] for x, y in c])
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5) * mpp; k = int(round(4 / mpp)) | 1
    mx = (dt == cv2.dilate(dt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))) & (m > 0) & (dt >= .6)
    for y, x in zip(*np.nonzero(mx)):
        new.append([round(b['x0'] + x / sx, 1), round(b['y1'] - y / sy, 1), round(float(np.clip(dt[y, x] * 1.3 + 1.2, 2.4, 6)) / .74, 1), 'round'])
# 2) reed fringe: a traced crown within 6 m of a lake whose photo footprint is mostly NOT canopy -> drop
shore = W_.buffer(6); drop = []
for i, (x, y, r, *_) in enumerate(A['crowns']):
    if not shore.contains(Point(x, y)): continue
    X, Y = px(x, y); R = max(2, r * .7 * sx)
    yy, xx = np.mgrid[int(Y - R):int(Y + R) + 1, int(X - R):int(X + R) + 1]
    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W) & ((yy - Y) ** 2 + (xx - X) ** 2 <= R * R)
    if ok.sum() == 0: continue
    if tree[yy[ok], xx[ok]].mean() < (.6 if r < 3.2 else .35): drop.append(i)
json.dump(dict(add=new, drop=drop, islands=islands), open(out, 'w'))
print(len(islands), 'islands', len(new), 'island crowns', len(drop), 'fringe crowns dropped of', len(A['crowns']))
