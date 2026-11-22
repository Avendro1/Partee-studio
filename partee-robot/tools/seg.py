# seg.py KEY : photo-first course model. Every pixel of the satellite photo is labelled (rough, fairway, green, bunker,
# water, trees, heather), edges follow the photo (superpixels), and the course JSON shapes are rebuilt from that.
# Map data is used only to TEACH the classifier (eroded, so edges don't matter) and for holes, tees, paths, buildings.
import sys, json, math, os, urllib.request, numpy as np, cv2
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from sklearn.ensemble import RandomForestClassifier
from skimage.segmentation import slic
key = sys.argv[1]; MPP = .3
CJ = f'/home/user/avendro-web-v2/partee-site/courses/{key}.json'; d = json.load(open(CJ)); c = math.cos(math.radians(d['lat']))
ll = lambda x, y: (d['lon'] + x / (111320 * c), d['lat'] + y / 110574)
merc = lambda lo, la: (lo * 20037508.34 / 180, math.log(math.tan(math.radians(90 + la) / 2)) * 6378137)
zone = unary_union([LineString(h['pts']).buffer(110) for h in d['holes']]); x0, y0, x1, y1 = zone.bounds
W = int((x1 - x0) / MPP); H = int((y1 - y0) / MPP); cache = f'/tmp/claude-0/osm/sat/{key}-{MPP}.png'; os.makedirs(os.path.dirname(cache), exist_ok=True)
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
px = lambda q: np.stack([(np.asarray(q)[:, 0] - x0) / MPP, (y1 - np.asarray(q)[:, 1]) / MPP], 1)
met = lambda p: np.stack([x0 + p[:, 0] * MPP, y1 - p[:, 1] * MPP], 1)
TJ = f'/tmp/claude-0/osm/teacher/{key}.json'; td = json.load(open(TJ)) if os.path.exists(TJ) else d   # teacher shapes (photo_fix output)
F = {}
for k, cs in td['feats']: F.setdefault(k, []).append(np.array(cs, float).reshape(-1, 2))
def rast(polys, dil=0):
    m = np.zeros((H, W), np.uint8)
    for q in polys: cv2.fillPoly(m, [px(q).astype(np.int32)], 1)
    if dil > 0: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dil / MPP) * 2 + 1,) * 2))
    if dil < 0: m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(-dil / MPP) * 2 + 1,) * 2))
    return m > 0
inz = np.zeros((H, W), np.uint8)
for g in (zone.geoms if hasattr(zone, 'geoms') else [zone]): cv2.fillPoly(inz, [px(np.array(g.exterior.coords)).astype(np.int32)], 1)
inz = inz > 0
# ---- features
lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
g = lab[..., 0]
def sdv(x, k): mu = cv2.blur(x, (k, k)); return np.sqrt(np.maximum(cv2.blur(x * x, (k, k)) - mu * mu, 0))
feats = [lab[..., i] for i in range(3)] + [hsv[..., i] for i in range(3)]
for k in (5, 15): feats += [cv2.blur(lab[..., i], (k, k)) for i in range(3)]
for k in (5, 11, 25): feats.append(sdv(g, k))
lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1), cv2.CV_32F)); feats += [cv2.blur(lap, (9, 9)), cv2.blur(lap, (25, 25))]
X = np.stack(feats, -1).reshape(-1, len(feats))
# ---- training labels from the (already photo-checked) shapes, eroded so only clear interiors teach
CL = ['rough', 'fairway', 'green', 'bunker', 'water', 'trees', 'heather', 'hard']
lbl = np.full((H, W), -1, np.int8)
hard = rast(F.get('building', []) + F.get('path', []) + F.get('parking', []) + F.get('road', []))
lbl[inz] = 0
lbl[rast(F.get('fairway', []), -3)] = 1
lbl[rast(F.get('green', []), -1.5)] = 2
lbl[rast(F.get('bunker', []), -.6)] = 3
lbl[rast([q for q in F.get('water', []) if Polygon(q).buffer(0).area > 150], -2)] = 4
lbl[rast(F.get('wood', []), -1.5)] = 5
lbl[rast(F.get('building', []), -1)] = 7
# rough is everything else, but don't teach rough near any shape edge
edge = rast(F.get('fairway', []) + F.get('green', []) + F.get('bunker', []) + F.get('water', []) + F.get('wood', []) + F.get('tee', []), 4) & (lbl == 0)
lbl[edge] = -1; lbl[hard & (lbl != 7)] = -1
# heather / long golden grass: warm, dry-looking pixels in rough teach their own class
warm = (lbl == 0) & (lab[..., 2] - 128 > np.percentile(lab[..., 2][lbl == 0] - 128, 90)) & (lab[..., 1] - 128 > -8)
lbl[warm] = 6
rng = np.random.default_rng(0); idx = []
for k in range(len(CL)):
    ii = np.flatnonzero(lbl.ravel() == k)
    if len(ii): idx.append(rng.choice(ii, size=min(25000, len(ii)), replace=False))
idx = np.concatenate(idx); print('train', {CL[k]: int((lbl.ravel()[idx] == k).sum()) for k in range(len(CL))}, flush=True)
rf = RandomForestClassifier(n_estimators=60, max_depth=18, min_samples_leaf=5, n_jobs=4, random_state=0).fit(X[idx], lbl.ravel()[idx])
prob = np.zeros((H * W, len(CL)), np.float32)
for s in range(0, H * W, 2_000_000): prob[s:s + 2_000_000][:, rf.classes_] = rf.predict_proba(X[s:s + 2_000_000])
prob = prob.reshape(H, W, len(CL)); print('predicted', flush=True)
# ---- superpixels: edges from the photo, label = mean class probability inside
sp = slic(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), n_segments=int(H * W * MPP * MPP / 6), compactness=8, start_label=0, channel_axis=-1)
n = sp.max() + 1; acc = np.zeros((n, len(CL)), np.float32)
for k in range(len(CL)): acc[:, k] = np.bincount(sp.ravel(), prob[..., k].ravel(), n)
best = acc.argmax(1); acc2 = acc.copy(); acc2[:, 4] = -1; second = acc2.argmax(1)
lakes = rast([q for q in F.get('water', []) if Polygon(q).buffer(0).area > 60], 4)
# water only where the map knows there is water (tree shadows are dark too); elsewhere take the next best label
spw = np.bincount(sp.ravel(), lakes.ravel().astype(np.float32), n) / np.maximum(1, np.bincount(sp.ravel(), None, n))
best = np.where((best == 4) & (spw < .6), second, best)
cls = best[sp]; cls[~inz] = 0
# bunkers can't sit on roads, car parks or roofs
nb, lb_ = cv2.connectedComponents((cls == 3).astype(np.uint8)); bad = np.unique(lb_[rast(F.get('building', []) + F.get('path', []), 1.5) & (cls == 3)])
cls[np.isin(lb_, bad[bad > 0])] = 0
# ...nor next to anything the photo shows as hard surface (unmapped drives, patios, house roofs)
hp = cv2.dilate((cls == 7).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))).astype(bool)
hp = hp | cv2.dilate(rast(F.get('building', [])).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))).astype(bool)
nb, lb_ = cv2.connectedComponents((cls == 3).astype(np.uint8)); bad = np.unique(lb_[hp & (cls == 3)]); cls[np.isin(lb_, bad[bad > 0])] = 0
# ...and real sand is warm (yellow/tan). Grey patches are tarmac, car parks, roofs or concrete.
_b = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 2].astype(np.float32) - 128
nb, lb_ = cv2.connectedComponents((cls == 3).astype(np.uint8))
if nb > 1:
    medb = np.array([np.median(_b[lb_ == i]) if i else 99 for i in range(nb)]); cls[(lb_ > 0) & (medb[lb_] < 11)] = 0
# ...and brighter than the grass (dead bracken, bare earth and dirt tracks are warm but darker)
_L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * 100 / 255; _gL = float(np.median(_L[(cls == 0) | (cls == 1)]))
nb, lb_ = cv2.connectedComponents((cls == 3).astype(np.uint8))
if nb > 1:
    medl = np.array([np.median(_L[lb_ == i]) if i else 999 for i in range(nb)]); cls[(lb_ > 0) & (medl[lb_] < _gL + 20)] = 0
cls[hard & (cls != 4)] = 7
# photo sand rescue: any clear sand patch (bright, warm, unsaturated, away from hard surfaces) is a bunker - same test the QA uses
_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); _Lq = _lab[..., 0] * 100 / 255; _Bq = _lab[..., 2] - 128
_S = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32); _gq = float(np.median(_Lq[(cls == 0) | (cls == 1)]))
_hd = cv2.dilate(((cls == 7) | hard).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))).astype(bool) | cv2.dilate(rast(F.get('building', [])).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))).astype(bool)
_sand = cv2.morphologyEx(((_Lq > _gq + 22) & (_S < 90) & (_Bq > 11) & ~_hd & (cls != 4)).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
n_, l_, st_, _ = cv2.connectedComponentsWithStats(_sand)
for i in range(1, n_):
    if st_[i, 4] * MPP * MPP >= 5: cls[(l_ == i) & inz] = 3
# pot bunkers (links): small, round, much brighter than their surroundings; colour varies (cream at RCD, grey-pink at Lahinch)
_g = lambda a, s_: cv2.GaussianBlur(a, (0, 0), s_)
_Lr = _lab[..., 0]; _Ar = _lab[..., 1] - 128; _vl = im.sum(2) > 30
_pot = (_g(_Lr, 1.2) - _g(_Lr, 6) > 14) & (_g(_Ar, 1.5) - _g(_Ar, 8) > -2) & (_g(_Lr, 1.2) > np.median(_Lr[_vl & inz]) + 30) & _vl & inz
_tr = cv2.dilate((cls == 5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
_pot &= ~_hd & ~_tr & (cls != 4)
n_, l_, st_, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(_pot.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
_kp = np.zeros(n_, bool)
for i in range(1, n_):
    a_ = st_[i, 4] * MPP * MPP; w_, h_ = st_[i, 2], st_[i, 3]
    _kp[i] = 2.5 < a_ < 120 and max(w_, h_) / max(1, min(w_, h_)) < 3 and st_[i, 4] / (w_ * h_) > .45
cls[_kp[l_]] = 3; _np = int(_kp.sum())
print('pot bunkers', _np, flush=True)
# lake islands: anything inside a lake that is clearly brighter than the water is land (reeds, trees, grass)
nw_, lw_ = cv2.connectedComponents((cls == 4).astype(np.uint8))
for i in range(1, nw_):
    mw = lw_ == i
    if mw.sum() * MPP * MPP < 200: continue
    wl = float(np.median(_Lq[mw])); isl_ = mw & (_Lq > wl + 14)
    isl_ = cv2.morphologyEx(isl_.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)
    cls[isl_] = 6
print('labelled', {CL[k]: round(float((cls == k).sum() * MPP * MPP / 1e4), 2) for k in range(len(CL))}, 'ha', flush=True)
np.save(f'/tmp/claude-0/osm/sat/{key}-cls.npy', cls.astype(np.uint8))
# ---- shapes
HOLES = []
def shapes(m, amin, simp=.35, keep_holes=False):
    cs, hier = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE); out = []
    if hier is None: return out
    for i, cn in enumerate(cs):
        if hier[0][i][3] != -1 or len(cn) < 4: continue
        holes = [met(cs[j][:, 0].astype(float)) for j in range(len(cs)) if hier[0][j][3] == i and len(cs[j]) >= 4]
        P = Polygon(met(cn[:, 0].astype(float))).buffer(0)
        for h_ in holes: P = P.difference(Polygon(h_).buffer(0))
        P = P.simplify(simp)
        for gg in (P.geoms if hasattr(P, 'geoms') else [P]):
            if gg.geom_type == 'Polygon' and gg.area >= amin: out.append(np.array(gg.exterior.coords)[:-1])   # holes filled by drawing order
            if keep_holes and gg.geom_type == 'Polygon' and gg.area >= amin:
                HOLES.extend(np.array(i.coords)[:-1] for i in gg.interiors if Polygon(i).area >= 12)
    return out
k3 = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(r / MPP) * 2 + 1,) * 2)
def clean(m, r=.6): return cv2.morphologyEx(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k3(r)), cv2.MORPH_CLOSE, k3(r))
NF = {k: v for k, v in F.items() if k in ('tee', 'building', 'path', 'rough', 'parking', 'road')}
# fairways: the mapped outline is the backbone (shadows and winter colour fool the pixel classifier on whole holes);
# the photo moves its edges by up to ~6 m, and only takes over where nothing is mapped
_tf = rast(F.get('fairway', [])) > 0
if _tf.any():
    _sf = clean(cls == 1, 1.5) > 0; _not = np.isin(cls, (2, 3, 4, 5, 7))
    _near = cv2.dilate(_tf.astype(np.uint8), k3(6)).astype(bool)
    _cov = cv2.dilate(_tf.astype(np.uint8), k3(40)).astype(bool)
    _fw = (_tf & ~(_not & ~cv2.dilate(_sf.astype(np.uint8), k3(1)).astype(bool) & cv2.erode(_not.astype(np.uint8), k3(1)).astype(bool))) | (_sf & _near) | (_sf & ~_cov)
    NF['fairway'] = shapes(clean(_fw, 1.5), 250)
else:
    NF['fairway'] = shapes(clean(cls == 1, 1.5), 250)
_tg = [Polygon(q).buffer(0) for q in F.get('green', [])]          # greens: keep the photo-snapped (SAM) outlines
NF['green'] = [q for q in F.get('green', [])] + [q for q in shapes(clean(cls == 2, .8), 250) if all(Polygon(q).buffer(0).distance(t) > 15 for t in _tg) and 4 * math.pi * Polygon(q).area / Polygon(q).length ** 2 > .45 and min(Polygon(q).distance(Point(h['pts'][-1])) for h in d['holes'] if h['pts']) < 35]   # extra greens only where a hole ends
NF['bunker'] = shapes(clean(cls == 3, .3), 5, .2)
NF['water'] = shapes(clean(cls == 4, .8), 25, keep_holes=True) + [q for q in F.get('water', []) if Polygon(q).buffer(0).area / max(1, Polygon(q).buffer(0).length) < 2.5]
trees = clean(cls == 5, .6); NF['wood'] = shapes(trees, 10, .5)
NF['heath'] = shapes(clean(cls == 6, 1.5), 150, .6)
NF['island'] = list(HOLES)
_known = rast(F.get('building', []) + F.get('path', []) + F.get('parking', []) + F.get('road', []), 2)
for q in shapes(clean((cls == 7) & ~_known & inz, .8), 120, .5):
    P_ = Polygon(q).buffer(0); rr = P_.minimum_rotated_rectangle
    if P_.area / max(1, rr.area) > .6 and 4 * math.pi * P_.area / P_.length ** 2 > .35: NF.setdefault('building', []).append(np.array(rr.exterior.coords)[:-1])
_bl = [Polygon(q).buffer(12) for q in NF.get('building', [])]
NF['bunker'] = [q for q in NF['bunker'] if not any(b.intersects(Polygon(q)) for b in _bl)]
# sanity: a pale winter/clifftop photo can turn whole rough areas into 'sand'. A new bunker no map knows about must be bunker-sized
_mb = [Polygon(q).buffer(2) for q in F.get('bunker', [])]
NF['bunker'] = [q for q in NF['bunker'] if Polygon(q).buffer(0).area < 400 or any(m.intersects(Polygon(q).buffer(0)) for m in _mb)]
_ma = sum(Polygon(q).buffer(0).area for q in F.get('bunker', [])); _na = sum(Polygon(q).buffer(0).area for q in NF['bunker'])
if len(F.get('bunker', [])) >= 20 and _na > 3 * _ma:   # photo found 3x more sand than the mapped bunkers: trust the map
    NF['bunker'] = [q for q in NF['bunker'] if any(m.intersects(Polygon(q).buffer(0)) for m in _mb)]; print('bunker sanity: kept mapped bunkers only', flush=True)   # no 'bunkers' in yards next to houses   # land inside lakes (drawn over the water)
print({k: len(v) for k, v in NF.items()}, flush=True)
# crowns from the tree class
f = 2; small = cv2.resize(trees, (W // f, H // f), interpolation=cv2.INTER_NEAREST); rem = small.copy()
R = cv2.distanceTransform(rem, cv2.DIST_L2, 5) * MPP * f; crowns = []
while True:
    j = int(np.argmax(R)); r = float(R.flat[j])
    if r < 1.5: break
    yy, xx = divmod(j, R.shape[1]); r = min(r, 8.5) * 1.15
    crowns.append([round(float(x0 + xx * f * MPP), 1), round(float(y1 - yy * f * MPP), 1), round(r, 1)])
    cv2.circle(rem, (xx, yy), int(r * .78 / (MPP * f)), 0, -1)
    a_, b_, c_, d_ = max(0, yy - 60), min(R.shape[0], yy + 61), max(0, xx - 60), min(R.shape[1], xx + 61)
    R[a_:b_, c_:d_] = cv2.distanceTransform(rem[a_:b_, c_:d_], cv2.DIST_L2, 5) * MPP * f * (rem[a_:b_, c_:d_] > 0)
order = {'wood': 0, 'heath': 0.5, 'sand': 1, 'water': 2, 'island': 2.2, 'rough': 3, 'path': 3.5, 'fairway': 4, 'building': 8, 'parking': 3.6, 'road': 3.55, 'tee': 5, 'green': 6, 'bunker': 7}
d['feats'] = [[k, [round(float(v), 1) for v in np.asarray(q).ravel()]] for k in sorted(NF, key=lambda k: order.get(k, 5)) for q in NF[k]]
d['crowns'] = crowns; d['photo_model'] = 'seg 2026-11-22'
open(CJ, 'w').write(json.dumps(d, separators=(',', ':')))
pal = np.array([[90, 150, 110], [80, 200, 140], [120, 235, 160], [210, 235, 245], [140, 80, 40], [30, 80, 30], [60, 170, 210], [120, 120, 120]], np.uint8)
vis = cv2.addWeighted(im, .45, pal[cls], .55, 0); vis[~inz] = im[~inz] // 2
cv2.imwrite(f'/tmp/claude-0/osm/fw/seg-{key}.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80]); print('crowns', len(crowns), 'done')
