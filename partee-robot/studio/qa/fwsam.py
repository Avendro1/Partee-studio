# fwsam.py KEY REF : SAM fairway proposal along one hole's mid-line from the cached satellite photo; writes overlay + polygon json for eye-check
import sys, json, math, numpy as np, cv2, torch
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union, substring
from segment_anything import sam_model_registry, SamPredictor
key, ref = sys.argv[1], sys.argv[2]; MPP = .3
d = json.load(open(f'/home/user/partee-studio/partee-site/courses/{key}.json'))
x0, y0, x1, y1 = unary_union([LineString(h['pts']).buffer(110) for h in d['holes']]).bounds
im = cv2.imread(f'/tmp/claude-0/osm/sat/{key}-0.3.png'); W, H = int((x1 - x0) / MPP), int((y1 - y0) / MPP)
assert abs(im.shape[1] - W) < 3 and abs(im.shape[0] - H) < 3, (im.shape, W, H)
px = lambda q: np.stack([(np.asarray(q)[:, 0] - x0) / MPP, (y1 - np.asarray(q)[:, 1]) / MPP], 1)
met = lambda p: np.stack([x0 + p[:, 0] * MPP, y1 - p[:, 1] * MPP], 1)
h = [h for h in d['holes'] if str(h['ref']) == ref][0]; L = LineString(h['pts'])
b = L.buffer(70).bounds; bx0, by1 = max(0, int((b[0] - x0) / MPP)), min(H, int((y1 - b[1]) / MPP)); bx1, by0 = min(W, int((b[2] - x0) / MPP)), max(0, int((y1 - b[3]) / MPP))
crop = im[by0:by1, bx0:bx1]
torch.set_num_threads(4); P = SamPredictor(sam_model_registry['vit_b'](checkpoint='/tmp/claude-0/sam_vit_b.pth')); P.set_image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
pos = px([substring(L, L.length * f, L.length * f).coords[0] for f in (.3, .45, .6, .75)]) - [bx0, by0]
# negatives: existing trees/greens/bunkers not used; just points 45 m either side of the line midpoint
n = []
for f in (.4, .65):
    a = L.interpolate(L.length * f); c2 = L.interpolate(L.length * f + 5); dx, dy = c2.x - a.x, c2.y - a.y; l = math.hypot(dx, dy) or 1
    for s in (-1, 1): n.append((a.x - dy / l * 45 * s, a.y + dx / l * 45 * s))
neg = px(n) - [bx0, by0]
pts = np.concatenate([pos, neg]); lab = np.array([1] * len(pos) + [0] * len(neg))
ms, sc, _ = P.predict(point_coords=pts, point_labels=lab, multimask_output=True)
best = None
for m, s in zip(ms, sc):
    cn, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cn: continue
    cn = max(cn, key=cv2.contourArea); poly = Polygon(met(cn[:, 0].astype(float) + [bx0, by0])).buffer(0)
    poly = poly.intersection(L.buffer(20)).buffer(3).buffer(-3)
    if poly.geom_type != 'Polygon': poly = max(poly.geoms, key=lambda g: g.area) if not poly.is_empty else poly
    if poly.is_empty: continue
    w = poly.area / max(poly.intersection(L.buffer(80)).length if False else L.length, 1)
    cov = poly.buffer(5).intersection(substring(L, L.length*.2, L.length*.85)).length / (L.length*.65)
    print(f'mask score {s:.2f} area {poly.area:.0f} m2 mean width {poly.area / L.length:.0f} m cover {poly.buffer(5).intersection(substring(L, L.length*.2, L.length*.85)).length / (L.length*.65):.2f}')
    sausage = not poly.is_empty and poly.area > .85 * L.buffer(20).area
    if not poly.is_empty and not sausage and 12 <= poly.area / L.length <= 60 and cov >= .7 and (best is None or s > best[0]): best = (s, poly, m)
ov = crop.copy()
if best:
    s, poly, m = best; m = np.zeros(crop.shape[:2], np.uint8); cv2.fillPoly(m, [(px(np.array(poly.exterior.coords)) - [bx0, by0]).astype(np.int32)], 1); ov[m > 0] = (ov[m > 0] * .5 + np.array([0, 255, 255]) * .5).astype(np.uint8)
    json.dump([round(v, 1) for xy in poly.simplify(1.2).exterior.coords for v in xy], open(f'/tmp/claude-0/fw/{key}-{ref}.json', 'w'))
for q in pos: cv2.circle(ov, tuple(q.astype(int)), 6, (0, 255, 0), -1)
for q in neg: cv2.circle(ov, tuple(q.astype(int)), 6, (0, 0, 255), -1)
cv2.polylines(ov, [(px(h['pts']) - [bx0, by0]).astype(np.int32)], False, (255, 255, 255), 2)
cv2.imwrite(f'/tmp/claude-0/fw/{key}-{ref}.jpg', np.hstack([crop, ov]))
print('best', None if not best else round(best[0], 2))
