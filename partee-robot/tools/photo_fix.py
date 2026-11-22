# photo_fix.py KEY : rebuild a course's outlines from the satellite photo itself so everything sits exactly where it is.
#  - trees: every crown found in the photo (canopy mask -> circle packing), stored as 'crowns' [x, y, r] + wood polygons
#  - bunkers + greens: each mapped one snapped to its true edge with Segment Anything; bunkers the map misses are added
#  - water: lakes trimmed/extended to the dark water actually in the photo
# Run with the SAM venv: /tmp/claude-0/samenv/bin/python photo_fix.py KEY
import os, sys, json, math, urllib.request, numpy as np, cv2, torch
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
if os.environ.get('SAM', 'base') == 'mobile':   # MobileSAM: same API, ~6x faster encoder
    from mobile_sam import sam_model_registry, SamPredictor; _SAMT, _SAMCK = 'vit_t', '/tmp/claude-0/mobile_sam.pt'
else:
    from segment_anything import sam_model_registry, SamPredictor; _SAMT, _SAMCK = 'vit_b', '/tmp/claude-0/sam_vit_b.pth'
key = sys.argv[1]; MPP = .3
CJ = f'/home/user/avendro-web-v2/partee-site/courses/{key}.json'; d = json.load(open(CJ)); c = math.cos(math.radians(d['lat']))
ll = lambda x, y: (d['lon'] + x / (111320 * c), d['lat'] + y / 110574)
merc = lambda lo, la: (lo * 20037508.34 / 180, math.log(math.tan(math.radians(90 + la) / 2)) * 6378137)
zone = unary_union([LineString(h['pts']).buffer(110) for h in d['holes']])
x0, y0, x1, y1 = zone.bounds
W = int((x1 - x0) / MPP); H = int((y1 - y0) / MPP); T = 1800
cache = f'/tmp/claude-0/osm/sat/{key}-{MPP}.png'; os.makedirs(os.path.dirname(cache), exist_ok=True)
if os.path.exists(cache) and cv2.imread(cache).shape[:2] == (H, W): im = cv2.imread(cache)
else:   # fetch the photo tiles in parallel, then cache them for seg.py / qa_sat.py
    from concurrent.futures import ThreadPoolExecutor
    im = np.zeros((H, W, 3), np.uint8)
    REL = d.get('sat_release')   # Esri Wayback release id: use an older, cloud-free capture where the current one is cloudy/hazy
    def _wb(a, b, cc, dd, w, hh, z=int(d.get('sat_zoom', 19))):   # old releases can be hazy at z19 but crisp at z18   # mosaic Wayback z19 tiles over a 3857 bbox, resampled to w x hh
        R0 = 20037508.342789244; ts = 2 * R0 / 2 ** z
        c0, c1 = int((a + R0) // ts), int((cc + R0) // ts); r0, r1 = int((R0 - dd) // ts), int((R0 - b) // ts)
        mo = np.zeros(((r1 - r0 + 1) * 256, (c1 - c0 + 1) * 256, 3), np.uint8)
        for r in range(r0, r1 + 1):
            for c_ in range(c0, c1 + 1):
                u = f'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/{REL}/{z}/{r}/{c_}'
                for k in range(4):
                    try: mo[(r - r0) * 256:(r - r0 + 1) * 256, (c_ - c0) * 256:(c_ - c0 + 1) * 256] = cv2.imdecode(np.frombuffer(urllib.request.urlopen(u, timeout=60).read(), np.uint8), cv2.IMREAD_COLOR); break
                    except Exception:
                        if k == 3: raise
        px0 = (a + R0) / ts - c0; py0 = (R0 - dd) / ts - r0; sx = (cc - a) / ts * 256 / w; sy = (dd - b) / ts * 256 / hh
        M = np.float32([[sx, 0, px0 * 256], [0, sy, py0 * 256]])
        return cv2.warpAffine(mo, M, (w, hh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    def _get(t):
        tx, ty = t; w, hh = min(T, W - tx), min(T, H - ty)
        (a, b), (cc, dd) = merc(*ll(x0 + tx * MPP, y1 - (ty + hh) * MPP)), merc(*ll(x0 + (tx + w) * MPP, y1 - ty * MPP))
        if REL: im[ty:ty + hh, tx:tx + w] = _wb(a, b, cc, dd, w, hh); return
        u = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={a},{b},{cc},{dd}&bboxSR=3857&imageSR=3857&size={w},{hh}&format=jpg&f=image'
        for k in range(4):
            try: im[ty:ty + hh, tx:tx + w] = cv2.resize(cv2.imdecode(np.frombuffer(urllib.request.urlopen(u, timeout=120).read(), np.uint8), cv2.IMREAD_COLOR), (w, hh)); return
            except Exception:
                if k == 3: raise
    with ThreadPoolExecutor(8) as ex: list(ex.map(_get, [(tx, ty) for ty in range(0, H, T) for tx in range(0, W, T)]))
    cv2.imwrite(cache, im)
print('photo', W, H, flush=True)
if os.environ.get('PHOTO_ONLY'): sys.exit(0)   # render-only reruns just need the cached photo
px = lambda q: np.stack([(np.asarray(q)[:, 0] - x0) / MPP, (y1 - np.asarray(q)[:, 1]) / MPP], 1)
met = lambda p: np.stack([x0 + p[:, 0] * MPP, y1 - p[:, 1] * MPP], 1)
def rast(polys, dil=0):
    m = np.zeros((H, W), np.uint8)
    for q in polys: cv2.fillPoly(m, [px(q).astype(np.int32)], 1)
    if dil: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(abs(dil) / MPP) * 2 + 1,) * 2)) if dil > 0 else cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(-dil / MPP) * 2 + 1,) * 2))
    return m
F = {}
for k, cs in d['feats']: F.setdefault(k, []).append(np.array(cs, float).reshape(-1, 2))
lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
L = lab[..., 0]; A = lab[..., 1] - 128; B = lab[..., 2] - 128
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
mu = cv2.blur(gray, (7, 7)); sd = np.sqrt(np.maximum(cv2.blur(gray * gray, (7, 7)) - mu * mu, 0)); sd = cv2.blur(sd, (9, 9))
inz = np.zeros((H, W), np.uint8); cv2.fillPoly(inz, [px(np.array(zone.exterior.coords)).astype(np.int32)], 1) if zone.geom_type == 'Polygon' else None
inz = inz > 0 if inz.any() else np.ones((H, W), bool)
V = hsv[..., 2]
turf = rast(F.get('fairway', []) + F.get('green', []) + F.get('tee', [])) > 0
grassV = np.median(V[turf]) if turf.any() else np.median(V[inz]); grassL = np.median(L[turf]) if turf.any() else np.median(L[inz])
# ---------- SAM: snap greens + bunkers, then find missing bunkers
torch.set_num_threads(4); sam = sam_model_registry[_SAMT](checkpoint=_SAMCK); P = SamPredictor(sam)
S, O = 1024, 256; tiles = [(tx, ty, min(W, tx + S), min(H, ty + S)) for ty in range(0, max(1, H - O), S - O) for tx in range(0, max(1, W - O), S - O)]
sandc = (L > grassL + 30) & (A > -8) & (B > -4) & (B < 45) & inz
sandc = cv2.morphologyEx(cv2.morphologyEx(sandc.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
known = rast(F.get('bunker', []), 1.5) | rast(F.get('building', []) + F.get('path', []), 1.0) | rast(F.get('water', []), 1.0)
nl, lb, st, cen = cv2.connectedComponentsWithStats(sandc * (1 - known))
cand = []
for i in range(1, nl):
    a = st[i, 4] * MPP * MPP
    if 8 <= a <= 900:
        bw, bh = st[i, 2], st[i, 3]
        if max(bw, bh) / max(1, min(bw, bh)) <= 5: cand.append((i, cen[i]))
done = {}; newb = []
def snap(m0, pt, box):
    ms, sc, _ = P.predict(point_coords=np.array([pt]), point_labels=np.array([1]), box=np.array(box), multimask_output=True)
    best = None
    for m, s in zip(ms, sc):
        inter = (m & m0).sum(); iou = inter / max(1, (m | m0).sum()); ar = m.sum() / max(1, m0.sum())
        if iou > .35 and .4 < ar < 2.5 and (best is None or s * iou > best[0]): best = (s * iou, m)
    return None if best is None else best[1]
def contour(m, tx, ty):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs: return None
    cn = max(cs, key=cv2.contourArea)[:, 0].astype(float) + [tx, ty]
    if len(cn) < 4: return None
    q = met(cn[::2]); Pq = Polygon(q).buffer(0).simplify(.25)
    return None if Pq.is_empty or Pq.geom_type != 'Polygon' else np.array(Pq.exterior.coords)[:-1]
for ti, (tx, ty, tx1, ty1) in enumerate(tiles):
    jobs = []
    for k in ('green', 'bunker'):
        for j, q in enumerate(F.get(k, [])):
            if (k, j) in done: continue
            p = px(q); b0 = p.min(0); b1 = p.max(0)
            if b0[0] >= tx and b0[1] >= ty and b1[0] <= tx1 and b1[1] <= ty1: jobs.append((k, j, p))
    cj = [(i, cc_) for i, cc_ in cand if tx + 40 <= cc_[0] < tx1 - 40 and ty + 40 <= cc_[1] < ty1 - 40 and ('n', i) not in done]
    if not jobs and not cj: continue
    P.set_image(cv2.cvtColor(im[ty:ty1, tx:tx1], cv2.COLOR_BGR2RGB))
    for k, j, p in jobs:
        pl = p - [tx, ty]; m0 = np.zeros((ty1 - ty, tx1 - tx), np.uint8); cv2.fillPoly(m0, [pl.astype(np.int32)], 1)
        b0 = pl.min(0); b1 = pl.max(0); pad = max(6, .15 * max(b1 - b0)); rp = Polygon(pl).buffer(0).representative_point()
        m = snap(m0 > 0, [rp.x, rp.y], [b0[0] - pad, b0[1] - pad, b1[0] + pad, b1[1] + pad])
        q = contour(m, tx, ty) if m is not None else None
        if k == 'bunker' and q is not None:     # snapped shape must be sand in the photo
            mm = rast([q]) > 0
            if mm.sum() and np.median(L[mm]) < grassL + 20: q = None
        done[(k, j)] = q
    for i, cc_ in cj:
        m0 = (lb[ty:ty1, tx:tx1] == i); ys, xs = np.nonzero(m0)
        if not len(xs): continue
        pad = 8; m = snap(m0, [cc_[0] - tx, cc_[1] - ty], [xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad])
        done[('n', i)] = True
        if m is None: continue
        q = contour(m, tx, ty)
        if q is None: continue
        mm = rast([q]) > 0; ring = (rast([q], 4) > 0) & ~(rast([q], 1.5) > 0)
        if not mm.sum() or not ring.sum(): continue
        a = Polygon(q).area; conv = a / max(1e-6, Polygon(q).convex_hull.area)
        greenring = ((A[ring] < -6) | (hsv[..., 0][ring] > 30) & (hsv[..., 0][ring] < 90)).mean()
        if np.median(L[mm]) > grassL + 28 and 8 <= a <= 900 and conv > .6 and greenring > .55: newb.append(q)
    print('tile', ti + 1, '/', len(tiles), flush=True)
G2 = [done.get(('green', j)) if done.get(('green', j)) is not None else q for j, q in enumerate(F.get('green', []))]
B2 = [done[('bunker', j)] for j, q in enumerate(F.get('bunker', [])) if done.get(('bunker', j)) is not None] + newb
print('greens', len(G2), 'bunkers kept', len(B2) - len(newb), 'of', len(F.get('bunker', [])), 'new', len(newb), flush=True)
# ---------- water: dark, smooth water actually in the photo, near a mapped lake
W2 = []  # NOTE 22 Nov: dark-water refit shrank Druids lakes; disabled below until fixed
for q in F.get('water', []):
    Pq = Polygon(q).buffer(0)
    if True or Pq.area < 80 or Pq.area / max(1, Pq.length) < 2.5: W2.append(q); continue   # refit off for now      # streams: keep
    reg = rast([q], 8) > 0; dark = reg & (L < min(70, np.percentile(L[rast([q]) > 0], 60) + 10)) & (sd < 14)
    dark = cv2.morphologyEx(cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m0 = rast([q]) > 0; iou = (dark.astype(bool) & m0).sum() / max(1, (dark.astype(bool) | m0).sum())
    cs, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if iou > .45 and cs:
        cn = max(cs, key=cv2.contourArea)[:, 0].astype(float); Pn = Polygon(met(cn)).buffer(0).simplify(.4)
        if Pn.geom_type == 'Polygon': W2.append(np.array(Pn.exterior.coords)[:-1]); continue
    W2.append(q)
# ---------- trees: canopy from the photo, then individual crowns
open_ = rast(F.get('fairway', []) + F.get('green', []) + F.get('tee', []), -1.0) | rast(B2 + W2 + F.get('building', []) + F.get('path', []) + F.get('road', []) + F.get('parking', []), 1.0)
vref = np.percentile(V[inz], 50); sref = np.percentile(sd[inz], 50)
can = ((V < vref * 1.08) & (sd > max(7, sref * 1.5)) & (hsv[..., 0] > 25) & (hsv[..., 0] < 100)) | ((V < vref * .6) & (sd > 6))
# winter photos: bare deciduous canopy is brown/mauve (redder than grass), and dense woods read darker than turf
grassA = np.median(A[turf]) if turf.any() else np.median(A[inz])
LINKS = grassA > -7 or bool(d.get('coastal_links'))   # links turf is straw-yellow; its brown patches are heather/dune grass, not bare trees
tex = cv2.blur(sd, (15, 15)) > sref * 1.3   # canopy is textured; rough, heather and dune grass are smooth at this scale
bare = ((((A > grassA + 9) & (L < grassL + 25) & (L > grassL * .45)) | ((L < grassL - 18) & (L > grassL * .3))) & tex & (not LINKS)).astype(np.uint8)
bare = cv2.morphologyEx(bare, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
bare = cv2.morphologyEx(bare, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
can = (can | (bare > 0)).astype(np.uint8)
can = cv2.morphologyEx(can, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
_inv = (1 - can).astype(np.uint8); _n, _lb, _st, _ = cv2.connectedComponentsWithStats(_inv); _sm = np.zeros(_n, bool); _sm[1:] = _st[1:, 4] * MPP * MPP < 40
can = (can > 0) | _sm[_lb]
can &= inz & (open_ == 0)
can = cv2.morphologyEx(can.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
can = cv2.morphologyEx(can, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
nl, lb, st, _ = cv2.connectedComponentsWithStats(can)
keep = np.zeros(nl, bool); keep[1:] = st[1:, 4] * MPP * MPP >= 12; can = keep[lb].astype(np.uint8)
f = 2; small = cv2.resize(can, (W // f, H // f), interpolation=cv2.INTER_NEAREST); dt = cv2.distanceTransform(small, cv2.DIST_L2, 5) * MPP * f
crowns = []; rem = small.copy(); R = cv2.distanceTransform(rem, cv2.DIST_L2, 5) * MPP * f
while True:
    j = int(np.argmax(R)); r = R.flat[j]
    if r < 1.6: break
    yy, xx = divmod(j, R.shape[1]); r = min(r, 8.5) * 1.15
    crowns.append([round(float(x0 + xx * f * MPP), 1), round(float(y1 - yy * f * MPP), 1), round(float(r), 1)])
    cv2.circle(rem, (xx, yy), int(r * .78 / (MPP * f)), 0, -1)
    y_0, y_1, x_0, x_1 = max(0, yy - 60), min(R.shape[0], yy + 61), max(0, xx - 60), min(R.shape[1], xx + 61)
    R[y_0:y_1, x_0:x_1] = cv2.distanceTransform(rem[y_0:y_1, x_0:x_1], cv2.DIST_L2, 5)[:, :] * MPP * f * (rem[y_0:y_1, x_0:x_1] > 0)
print('crowns', len(crowns), flush=True)
cs, _ = cv2.findContours(can, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
Wd = []
for cn in cs:
    if cv2.contourArea(cn) * MPP * MPP < 12: continue
    Pq = Polygon(met(cv2.approxPolyDP(cn, 1.0 / MPP, True)[:, 0].astype(float))).buffer(0)
    for g in (Pq.geoms if hasattr(Pq, 'geoms') else [Pq]):
        if g.geom_type == 'Polygon' and g.area >= 12: Wd.append(np.array(g.exterior.coords)[:-1])
# ---------- write
order = {'wood': 0, 'heath': 0, 'sand': 1, 'water': 2, 'rough': 3, 'path': 3.5, 'fairway': 4, 'building': 8, 'tee': 5, 'green': 6, 'bunker': 7}
NF = {k: v for k, v in F.items() if k not in ('wood', 'bunker', 'green', 'water')}
NF['wood'] = Wd; NF['bunker'] = B2; NF['green'] = G2; NF['water'] = W2
d['feats'] = [[k, [round(float(v), 1) for v in np.asarray(q).ravel()]] for k in sorted(NF, key=lambda k: order.get(k, 5)) for q in NF[k]]
d['crowns'] = crowns; d['links'] = bool(LINKS); d['photo_fixed'] = '2026-11-22'
s_ = json.dumps(d, separators=(',', ':')); open(CJ, 'w').write(s_)
chk = im.copy()
for x, y, r in crowns: cv2.circle(chk, tuple(px(np.array([[x, y]]))[0].astype(int)), int(r / MPP), (255, 0, 255), 1)
for q in B2: cv2.polylines(chk, [px(q).astype(np.int32)], True, (255, 255, 255), 2)
for q in newb: cv2.polylines(chk, [px(q).astype(np.int32)], True, (0, 255, 255), 2)
for q in G2: cv2.polylines(chk, [px(q).astype(np.int32)], True, (0, 255, 0), 2)
for q in W2: cv2.polylines(chk, [px(q).astype(np.int32)], True, (255, 128, 0), 2)
cv2.imwrite(f'/tmp/claude-0/osm/fw/photofix-{key}.jpg', chk, [cv2.IMWRITE_JPEG_QUALITY, 80]); print('done')
