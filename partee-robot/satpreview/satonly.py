# satonly.py KEY NAME PLACE BOUNDARY.geojson OUT_DIR : whole-course preview from the satellite photo only (no hole data).
# A pixel classifier is taught on checked courses we already have, then labels the new course inside its boundary.
import sys, json, math, os, urllib.request, numpy as np, cv2
from shapely.geometry import Polygon, LineString, Point, shape
from shapely.ops import unary_union
from sklearn.ensemble import RandomForestClassifier
key, name, place, bnd, out = sys.argv[1:6]; MPP = .3
TRAIN = os.environ.get('TRAIN', 'county-longford birr tullamore mount-juliet castle elmgreen').split()
CL = ['rough', 'fairway', 'green', 'bunker', 'water', 'trees']

def features(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32); g = lab[..., 0]
    sdv = lambda x, k: np.sqrt(np.maximum(cv2.blur(x * x, (k, k)) - cv2.blur(x, (k, k)) ** 2, 0))
    f = [lab[..., i] for i in range(3)] + [hsv[..., i] for i in range(3)]
    for k in (5, 15, 41): f += [cv2.blur(lab[..., i], (k, k)) for i in range(3)]
    for k in (5, 11, 25): f.append(sdv(g, k))
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1), cv2.CV_32F)); f += [cv2.blur(lap, (9, 9)), cv2.blur(lap, (25, 25))]
    return np.stack(f, -1)

def frame_of(d):
    z = unary_union([LineString(h['pts']).buffer(110) for h in d['holes'] if len(h['pts']) > 1]); return z.bounds

def rast(polys, x0, y1, H, W, ero=0):
    m = np.zeros((H, W), np.uint8)
    for q in polys:
        q = np.asarray(q, float); cv2.fillPoly(m, [np.stack([(q[:, 0] - x0) / MPP, (y1 - q[:, 1]) / MPP], 1).astype(np.int32)], 1)
    if ero: m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ero / MPP) * 2 + 1,) * 2))
    return m > 0

MODEL = os.environ.get('MODEL')
# ---- teach on checked courses
Xs, ys = [], []; rng = np.random.default_rng(0)
for k in ([] if MODEL else TRAIN):
    d = json.load(open(f'/home/user/partee-studio/partee-site/courses/{k}.json')); im = cv2.imread(f'/tmp/claude-0/osm/sat/{k}-0.3.png')
    if im is None: continue
    x0, y0, x1, y1 = frame_of(d); H, W = im.shape[:2]; F = {}
    for t, c in d['feats']: F.setdefault(t, []).append(np.array(c, float).reshape(-1, 2))
    lbl = np.full((H, W), -1, np.int8)
    near = rast([np.array(LineString(h['pts']).buffer(60).exterior.coords) for h in d['holes'] if len(h['pts']) > 1], x0, y1, H, W)
    lbl[near] = 0
    edge = cv2.dilate(rast(sum([F.get(t, []) for t in ('fairway', 'green', 'bunker', 'water', 'tee', 'building', 'road', 'path', 'parking')], []), x0, y1, H, W).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    lbl[edge & (lbl == 0)] = -1
    lbl[rast(F.get('fairway', []), x0, y1, H, W, 3)] = 1
    lbl[rast(F.get('green', []), x0, y1, H, W, 1.5)] = 2
    lbl[rast(F.get('bunker', []), x0, y1, H, W, .6)] = 3
    wy = [w for w in F.get('water', []) if Polygon(w).buffer(0).area > 300]
    lbl[rast(wy, x0, y1, H, W, 2)] = 4
    tr = np.zeros((H, W), np.uint8)
    for cx, cy, r in [c[:3] for c in d.get('crowns', [])]: cv2.circle(tr, (int((cx - x0) / MPP), int((y1 - cy) / MPP)), max(1, int(r * .6 / MPP)), 1, -1)
    lbl[(tr > 0) & (lbl <= 0)] = 5
    X = features(im)
    for c in range(len(CL)):
        ii = np.flatnonzero(lbl.ravel() == c)
        if len(ii): ii = rng.choice(ii, min(12000, len(ii)), replace=False); Xs.append(X.reshape(-1, X.shape[-1])[ii]); ys.append(np.full(len(ii), c))
    print('taught', k, flush=True)
if not MODEL:
    X_ = np.concatenate(Xs); y_ = np.concatenate(ys); print({CL[c]: int((y_ == c).sum()) for c in range(len(CL))}, flush=True)
    rf = RandomForestClassifier(n_estimators=80, max_depth=20, min_samples_leaf=5, n_jobs=4, random_state=0).fit(X_, y_)

# ---- new course: boundary -> local metres -> photo
g = json.load(open(bnd)); g = g[0]['geojson'] if isinstance(g, list) else g
B = shape(g); lat, lon = B.centroid.y, B.centroid.x; c = math.cos(math.radians(lat))
loc = lambda lo, la: ((lo - lon) * 111320 * c, (la - lat) * 110574)
ring = [loc(*p) for p in (B.exterior.coords if B.geom_type == 'Polygon' else max(B.geoms, key=lambda q: q.area).exterior.coords)]
Bm = Polygon(ring).buffer(0); x0, y0, x1, y1 = Bm.buffer(110).bounds
ll = lambda x, y: (lon + x / (111320 * c), lat + y / 110574)
merc = lambda lo, la: (lo * 20037508.34 / 180, math.log(math.tan(math.radians(90 + la) / 2)) * 6378137)
W, H = int((x1 - x0) / MPP), int((y1 - y0) / MPP); cache = f'/tmp/claude-0/osm/sat/{key}-0.3.png'
if os.path.exists(cache): im = cv2.imread(cache)
else:
    im = np.zeros((H, W, 3), np.uint8); T = 1800
    for ty in range(0, H, T):
        for tx in range(0, W, T):
            w, hh = min(T, W - tx), min(T, H - ty)
            (a, b), (cc, dd) = merc(*ll(x0 + tx * MPP, y1 - (ty + hh) * MPP)), merc(*ll(x0 + (tx + w) * MPP, y1 - ty * MPP))
            u = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={a},{b},{cc},{dd}&bboxSR=3857&imageSR=3857&size={w},{hh}&format=jpg&f=image'
            im[ty:ty + hh, tx:tx + w] = cv2.resize(cv2.imdecode(np.frombuffer(urllib.request.urlopen(u, timeout=120).read(), np.uint8), cv2.IMREAD_COLOR), (w, hh))
    cv2.imwrite(cache, im)
H, W = im.shape[:2]; print('photo', W, H, flush=True)
inb = rast([np.array(Bm.buffer(10).exterior.coords)], x0, y1, H, W)
if MODEL:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unet')); from predict import classify
    cls = classify(im, MPP).astype(np.int8)
else:
    X = features(im).reshape(-1, 20); cls = np.full(H * W, 0, np.int8); ii = np.flatnonzero(inb.ravel())
    for s in range(0, len(ii), 1_500_000): cls[ii[s:s + 1_500_000]] = rf.predict(X[ii[s:s + 1_500_000]])
    cls = cls.reshape(H, W)
cls[~inb] = -1
met = lambda p: np.stack([x0 + p[:, 0] * MPP, y1 - p[:, 1] * MPP], 1)
k_ = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(r / MPP) * 2 + 1,) * 2)
def clean(m, r): return cv2.morphologyEx(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k_(r)), cv2.MORPH_CLOSE, k_(r))
def shapes(m, amin, amax=1e9, simp=.6):
    cn, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); o = []
    for q in cn:
        a = cv2.contourArea(q) * MPP * MPP
        if a < amin or a > amax or len(q) < 4: continue
        p = Polygon(met(cv2.approxPolyDP(q, simp / MPP, True)[:, 0].astype(float))).buffer(0)
        if p.geom_type == 'Polygon' and not p.is_empty: o.append(p)
    return o
feats = []
_gr = clean(((cls == 0) | (cls == 1)) & inb, 1.5)   # all open grass; the classifier can't split fairway from rough, so fairway = the middle of each open corridor
_fw = cv2.dilate(cv2.erode(_gr, k_(float(os.environ.get('FWE', '11')))), k_(5)) & _gr
if MODEL: _fw = (cls == 1) & inb   # the model tells fairway from rough itself
fw = shapes(clean(_fw, 2.5), 600 if MODEL else 1500)
te = shapes(clean((cls == 6) & inb, .8), 25, 900) if MODEL else []
gr = [p for p in shapes(clean(cls == 2, .8), 120, 1600) if 4 * math.pi * p.area / p.length ** 2 > .45]
bk = [p for p in shapes(clean(cls == 3, .4), 12, 600)]
# water only where the photo is smooth and dark (same rule as the checked courses)
lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); sd = np.sqrt(np.maximum(cv2.blur(lab[..., 0] ** 2, (15, 15)) - cv2.blur(lab[..., 0], (15, 15)) ** 2, 0))
wm = clean((cls == 4) & (lab[..., 0] < 95) & (sd < 7), 1.5)
wa = shapes(wm, 400)
for t, L in (('fairway', fw), ('green', gr), ('bunker', bk), ('water', wa), ('tee', te)):
    for p in L: feats.append([t, [round(v, 1) for xy in p.exterior.coords for v in xy]])
# trees: canopy pixels -> crowns
can = clean(((cls == 5) | ((cls == 4) & ~wm.astype(bool))) & inb, .9); f = 2; small = cv2.resize(can, (W // f, H // f), interpolation=cv2.INTER_NEAREST)
crowns = []; rem = small.copy()
for _ in range(20000):
    dt = cv2.distanceTransform(rem, cv2.DIST_L2, 5) * MPP * f
    if dt.max() < 1.8: break
    for _r in range(400):
        yy, xx = np.unravel_index(np.argmax(dt), dt.shape); r = float(dt[yy, xx])
        if r < 1.8: break
        r = min(r, 7.5); crowns.append([round(x0 + xx * f * MPP, 1), round(y1 - yy * f * MPP, 1), round(r, 1), 'round'])
        cv2.circle(rem, (xx, yy), int(r * .8 / (MPP * f)), 0, -1); cv2.circle(dt, (xx, yy), int(r * 1.6 / (MPP * f)), 0, -1)
# stripes need a direction per fairway: its long axis stands in for the hole line
holes = []
for p in sorted(fw, key=lambda p: -p.area):
    if p.area < 2500: continue
    xy = list(p.minimum_rotated_rectangle.exterior.coords); e = [(xy[i], xy[i + 1]) for i in range(2)]
    (a, b) = max(e, key=lambda s: math.dist(*s)); (cc, dd) = min(e, key=lambda s: math.dist(*s))
    m1 = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2); m2 = (m1[0] + (xy[2][0] - xy[1][0]) if e.index((a, b)) == 0 else m1[0], 0)
    o = ((cc[0] + dd[0]) / 2 - (a[0] + b[0]) / 2, (cc[1] + dd[1]) / 2 - (a[1] + b[1]) / 2)
    ctr = (p.centroid.x, p.centroid.y); v = (b[0] - a[0], b[1] - a[1]); n = math.hypot(*v) or 1; hl = n / 2 * .9
    holes.append({'ref': len(holes) + 1, 'pts': [[ctr[0] - v[0] / n * hl, ctr[1] - v[1] / n * hl], [ctr[0] + v[0] / n * hl, ctr[1] + v[1] / n * hl]]})
pins = [[round(p.centroid.x, 1), round(p.centroid.y, 1)] for p in gr]
os.makedirs(out, exist_ok=True)
json.dump(dict(key=key, name=name, place=place, holes=holes, feats=feats, crowns=crowns, pins=pins, links=False,
               sat={'path': cache, 'x0': x0, 'y1': y1, 'mpp': MPP}, boundary=[[round(a, 1), round(b, 1)] for a, b in ring]), open(f'{out}/all.json', 'w'))
print(key, 'fairways', len(fw), 'greens', len(gr), 'bunkers', len(bk), 'water', len(wa), 'crowns', len(crowns), 'stripe lines', len(holes))
dbg = im.copy(); col = {1: (120, 230, 160), 2: (60, 255, 60), 3: (200, 240, 255), 4: (255, 120, 0), 5: (20, 90, 20)}
for kk, cc_ in col.items(): dbg[cls == kk] = (dbg[cls == kk] * .45 + np.array(cc_) * .55).astype(np.uint8)
cv2.imwrite(f'{out}/classes.jpg', cv2.resize(dbg, None, fx=.35, fy=.35))
