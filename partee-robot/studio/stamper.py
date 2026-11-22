# stamper.py : fast 2D "studio" renderer. course-style JSON (local metres) -> print image.
import json, math, sys, time, glob, numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union
T0 = time.time()
src, out = sys.argv[1], sys.argv[2]; PPM = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
d = json.load(open(src)); rng = np.random.default_rng(7)
def P(f): c = f[1]; return Polygon(list(zip(c[0::2], c[1::2]))).buffer(0)
polys = {}
for f in d['feats']: polys.setdefault(f[0], []).append(P(f))
LINES = [np.array(h['pts'], float) for h in d['holes'] if len(h['pts']) > 1]; line = LINES[0]
def _close(ps, r, hole):
    g = unary_union(ps).buffer(r, 12).buffer(-r, 12); out_ = []
    for p in getattr(g, 'geoms', [g]):
        if p.geom_type == 'Polygon' and p.area > 3: out_.append(Polygon(p.exterior, [i for i in p.interiors if Polygon(i).area > hole]))
    return out_
if 'own' in d and len(d['holes']) == 1:   # only this hole's own fairway/green/tees; neighbours read as rough, like the club maps
    ownp = [P(['x', c]) for c in d['own']]
    for k_ in ('fairway', 'green', 'tee'):
        polys[k_] = [p for p in polys.get(k_, []) if any(p.intersection(o).area > .5 * min(p.area, o.area) for o in ownp)]
for k_, r_, h_ in (('fairway', 7, 400), ('green', 1.5, 1e9), ('bunker', .4, 1e9)):
    if polys.get(k_): polys[k_] = _close(polys[k_], r_, h_)
if polys.get('fairway'): polys['fairway'] = [q for p in polys['fairway'] for q in getattr(p.buffer(-3, 16).buffer(3, 16), 'geoms', [p.buffer(-3, 16).buffer(3, 16)]) if q.area > 30]   # no steps or notches
polys['tee'] = [p.buffer(-1.2).buffer(1.2, join_style=1) for p in polys.get('tee', [])]
def smooth_line(pts, it=4):
    p = np.asarray(pts, float)
    for _ in range(it):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]): q += [a * .75 + b * .25, a * .25 + b * .75]
        q.append(p[-1]); p = np.array(q)
    return p
allg = unary_union([p for v in polys.values() for p in v] + [Point(c[0], c[1]).buffer(c[2]) for c in d['crowns']])
x0, y0, x1, y1 = allg.buffer(25).bounds
W, H = int((x1 - x0) * PPM), int((y1 - y0) * PPM)
def px(x, y): return ((np.asarray(x) - x0) * PPM, (y1 - np.asarray(y)) * PPM)
def mask(ps, buf=0):
    im = Image.new('L', (W, H), 0); dr = ImageDraw.Draw(im)
    for p in ps:
        p = p.buffer(buf) if buf else p
        for g in getattr(p, 'geoms', [p]):
            if g.is_empty: continue
            xs, ys = px(*g.exterior.xy); dr.polygon(list(zip(xs, ys)), fill=255)
            for h in g.interiors: xs, ys = px(*h.xy); dr.polygon(list(zip(xs, ys)), fill=0)
    return np.asarray(im, np.float32) / 255
def blur(a, m): return ndi.gaussian_filter(a, max(.01, m * PPM))
def _oct(r, sig):
    f = max(1, int(sig / 1.5)); h_, w_ = H // f + 2, W // f + 2
    a = ndi.gaussian_filter(r.standard_normal((h_, w_)).astype(np.float32), max(.5, sig / f))
    if f > 1: a = ndi.zoom(a, f, order=1)
    return a[:H, :W]
def noise(sc, seed, oct=3):
    r = np.random.default_rng(seed); n = np.zeros((H, W), np.float32); amp = 1
    for o in range(oct):
        n += amp * _oct(r, max(.5, sc * PPM / 2 ** o)); amp *= .55
    return (n - n.mean()) / (n.std() + 1e-6)
def hexc(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255
def shift(a, dx, dy): return ndi.shift(a, (dy * PPM, dx * PPM), order=1, mode='nearest')   # +dx right, +dy down (metres)
img = np.zeros((H, W, 3), np.float32)
def _bb(m):
    r_ = np.flatnonzero(m.any(1)); c_ = np.flatnonzero(m.any(0))
    return None if not len(r_) else (slice(r_[0], r_[-1] + 1), slice(c_[0], c_[-1] + 1))
def fill(m, col, a=1.):
    m = np.broadcast_to(m, (H, W)); b = _bb(m > 1e-3)
    if b is None: return
    mm = np.clip(m[b] * a, 0, 1)[..., None]; c = col[b] if np.ndim(col) == 3 else np.asarray(col, np.float32)[None, None]
    img[b] = img[b] * (1 - mm) + c * mm
def mul(m, k):  # darken/lighten by mask
    m = np.broadcast_to(m, (H, W)); b = _bb(m > 1e-3)
    if b is None: return
    img[b] = img[b] * (1 + (k - 1) * np.clip(m[b], 0, 1)[..., None])
_G = []
def crop_on(bounds, pad):
    global H, W, x0, y1, img, wx, wy
    bx0, by0, bx1, by1 = bounds
    r0 = max(0, int((y1 - by1 - pad) * PPM)); r1 = min(H, int((y1 - by0 + pad) * PPM) + 1)
    c0 = max(0, int((bx0 - pad - x0) * PPM)); c1 = min(W, int((bx1 + pad - x0) * PPM) + 1)
    if r1 <= r0 + 2 or c1 <= c0 + 2: return False
    _G.append((H, W, x0, y1, img, wx, wy))
    img = img[r0:r1, c0:c1]; wx = wx[r0:r1, c0:c1]; wy = wy[r0:r1, c0:c1]
    x0 = x0 + c0 / PPM; y1 = y1 - r0 / PPM; H, W = r1 - r0, c1 - c0
    return True
def crop_off():
    global H, W, x0, y1, img, wx, wy
    H, W, x0, y1, img, wx, wy = _G.pop()
# --- along-line coordinate for mowing bands
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); wx = x0 + xx / PPM; wy = y1 - yy / PPM
best = np.full((H, W), 1e9, np.float32); tal = np.zeros((H, W), np.float32); acc = 0
segs = []
for ln_ in LINES:
    sl_ = smooth_line(ln_); a0 = 0
    for p_, q_ in zip(sl_[:-1], sl_[1:]): segs.append((p_, q_, a0)); a0 += float(np.hypot(*(q_ - p_)))
for (ax, ay), (bx, by), acc in segs:
    P_ = 120; r0 = max(0, int((y1 - max(ay, by) - P_) * PPM)); r1 = min(H, int((y1 - min(ay, by) + P_) * PPM) + 1)
    c0 = max(0, int((min(ax, bx) - P_ - x0) * PPM)); c1 = min(W, int((max(ax, bx) + P_ - x0) * PPM) + 1)
    if r1 <= r0 or c1 <= c0: continue
    sw, sy_, bs_, ts_ = wx[r0:r1, c0:c1], wy[r0:r1, c0:c1], best[r0:r1, c0:c1], tal[r0:r1, c0:c1]
    vx, vy = bx - ax, by - ay; L2 = max(1e-6, vx * vx + vy * vy); t = np.clip(((sw - ax) * vx + (sy_ - ay) * vy) / L2, 0, 1)
    dd = (sw - ax - t * vx) ** 2 + (sy_ - ay - t * vy) ** 2; k = dd < bs_; bs_[k] = dd[k]; ts_[k] = acc + t[k] * math.sqrt(L2)
# --- PALETTE (tune here)
C = {k: hexc(v) for k, v in dict(rough='#64813A', rough_d='#567135', semi='#74923F', fair='#A1BE5B', fair2='#91AF51',
     green='#9EBF5C', green2='#91B354', collar='#789B44', sand='#F7F1E1', sand_s='#CDBF9E', water='#4E93A3', water_d='#22596B',
     shore='#6E9A57', heath='#8E8A4E', heath_d='#6B5E36', tarmac='#7C7C78', path='#CFC8B6', path_e='#A9A28F', roof='#9A958C', roof2='#7F7A72',
     deck='#9C7A55', paper='#F6F8F9').items()}
def canopy():
    m = np.zeros((H, W), np.float32)
    for c in d['crowns']:
        X, Y = px(c[0], c[1]); R = c[2] * PPM; sl = (slice(max(0, int(Y - R)), min(H, int(Y + R) + 1)), slice(max(0, int(X - R)), min(W, int(X + R) + 1)))
        gy_, gx_ = np.mgrid[sl]; m[sl] = np.maximum(m[sl], (np.hypot(gx_ - X, gy_ - Y) < R).astype(np.float32))
    return m
CAN = canopy()
# 1 rough with texture, darker in deep rough
fair_m = mask(polys.get('fairway', [])); green_m = mask(polys.get('green', [])); tee_m = mask(polys.get('tee', []))
play = np.clip(fair_m + green_m + tee_m, 0, 1)
dplay = ndi.distance_transform_edt(play < .5) / PPM
img[:] = C['rough']; fill(np.clip((dplay - 6) / 25, 0, 1), C['rough_d'], .8)
if d.get('links'):   # links: deep rough is tan marram / dune grass, mottled, as in the photo (Shane 22 Nov: must look like the satellite)
    dn = noise(9, 61, 2); dn2 = noise(2.5, 62, 1); dune = np.clip((dplay - 5) / 14, 0, 1) * np.clip(.75 + .35 * dn, 0, 1)
    fill(dune, hexc('#9C9358'), .85); fill(dune * np.clip(-dn2, 0, 1), hexc('#7E7445'), .6); fill(dune * np.clip(dn2 - .3, 0, 1), hexc('#B3A873'), .5)
n1, n2, n3 = noise(6, 1), noise(1.2, 2, 2), noise(.25, 3, 1)
mul(np.ones((H, W)), 1); g_ = np.random.default_rng(33).standard_normal((H, W)).astype(np.float32)
grain = ndi.gaussian_filter(g_, (max(.6, .35 * PPM), max(.4, .06 * PPM))); grain /= grain.std() + 1e-6
grain2 = ndi.gaussian_filter(g_, (max(.4, .05 * PPM), max(.6, .3 * PPM))); grain2 /= grain2.std() + 1e-6
img *= (1 + .035 * n1 + .035 * n2 + .03 * grain + .02 * grain2)[..., None]
# heather
hm = blur(mask(polys.get('heath', [])), .6); hn = noise(.8, 4, 2)
hm = blur(mask(polys.get('heath', [])), 2.5); fill(hm, C['heath'], .7); tuft = np.clip(noise(.22, 16, 1) - .4, 0, 1); fill(hm * tuft, np.array([.55, .45, .30]), .45); fill(hm * np.clip(hn * .8 + .2, 0, 1), C['heath_d'], .6); fill(hm * np.clip(-hn - .6, 0, 1), C['rough'], .7)
# first cut ring
semi = blur(mask(polys.get('fairway', []) + polys.get('green', []), 2.4), .07); fill(semi, C['semi'], 1)
fill(np.clip(blur(mask(polys.get('fairway', []), 6), 2) - semi, 0, 1), C['semi'], .22)
# fairway: bands across line of play, crisp edge, soft inner rim + rough casts small shadow NW edge
band = (np.floor(tal / 13) % 2).astype(np.float32); band = blur(band, .15)
fw = blur(fair_m, .06); fill(fw, C['fair']); fill(fw * band, C['fair2'], .9)
img *= (1 + fw[..., None] * (.035 * noise(3, 5, 2) + .02 * n3)[..., None])
rim = np.clip(fw - blur(fair_m, 1.2) * 1.0, 0, 1); mul(rim * fw, .93)
edge_sh = np.clip(fw - shift(fw, -.9, .9), 0, 1); mul(blur(edge_sh, .4), .86)
# tees
tm = blur(tee_m, .08); fill(blur(mask(polys.get('tee', []), .8), .1) - tm * 0, C['semi'], 1); fill(tm, C['fair']); fill(tm * band, C['fair2'], .7)
mul(blur(np.clip(tm - shift(tm, -.5, .5), 0, 1), .2), .85); mul(blur(np.clip(tm - shift(tm, .5, -.5), 0, 1), .2), 1.08)
# green: collar, cross-mow, gentle dome light
gm = blur(green_m, .04); col = blur(mask(polys.get('green', []), 1.8), .04)
fill(col, C['collar']); fill(gm, C['green'])
dg = (np.floor((wx * .8 + wy * .6) / 2.2) % 2).astype(np.float32); fill(gm * blur(dg, .08), C['green2'], .6)
mul(blur(np.clip(col - gm, 0, 1), .03) * np.clip(gm * 0 + 1, 0, 1) * (blur(np.clip(col-gm,0,1),.03) > .3), 1)
mul(np.clip(blur(gm, .25) - gm, 0, 1) * 0 + np.clip(gm - blur(green_m * 0 + ndi.binary_erosion(green_m > .5, iterations=2).astype(np.float32), .05), 0, 1), .88)
dome = blur(green_m, 6); mul(gm * np.clip(shift(dome, 2, 2) - dome + .5, 0, 1), 1.10)
mul(blur(np.clip(gm - shift(gm, -.4, .4), 0, 1), .25), .9)
# water: depth gradient, bank, shadow from NW bank, glints
wm = mask(polys.get('water', [])); wb = blur(wm, .05)
bank = blur(mask(polys.get('water', []), 1.0), .2); fill(bank, C['shore'], .9)
depth = np.clip(ndi.distance_transform_edt(wm > .5) / PPM / 14, 0, 1)
wcol = C['water'][None, None] * (1 - depth[..., None]) + C['water_d'][None, None] * depth[..., None]
fill(wb, wcol); rip = np.sin((wy + 2 * noise(4, 15, 1)) * 2.6 * math.pi) * .5 + .5; img *= (1 + wb * (.04 * noise(4, 8, 2) + .025 * rip * depth))[..., None]
fill(blur(np.clip(wb - blur(ndi.binary_erosion(wm > .5, iterations=int(PPM * .5)).astype(np.float32), .05), 0, 1), .05), np.array([.80, .86, .80]), .45)
mul(blur(np.clip(wb - shift(wb, -1.6, 1.6), 0, 1), .8), .72)
gl = np.clip(noise(3, 9, 2) - 1.4, 0, 1) * wb * depth; fill(blur(gl, 1.5), np.array([.75, .88, .9]), .25)
if polys.get('island'):   # land inside lakes: grass with a soft bank
    im_ = mask(polys['island']); ib = blur(im_, .08); fill(blur(mask(polys['island'], .9), .2), C['shore'], .9); fill(ib, C['rough'])
    img[:] *= (1 + ib * (.035 * n1 + .03 * grain))[..., None]; wm = np.clip(wm - im_, 0, 1); wb = np.clip(wb - ib, 0, 1)
refl = blur(shift(CAN, 1.5, 3.0), 1.5) * wb; mul(np.clip(refl * 1.6, 0, 1), .72)
rl = Image.new('RGBA', (W, H), (0, 0, 0, 0)); rd = ImageDraw.Draw(rl); rr_ = np.random.default_rng(21)
for p in polys.get('water', []):
    if p.area < 300: continue
    ring = p.exterior; Lr = ring.length; ph = rr_.uniform(0, 6)
    for s_ in np.arange(0, Lr, .45):
        if math.sin(s_ / 9 + ph) + .6 * math.sin(s_ / 3.1 + 2 * ph) < .35: continue
        q = ring.interpolate(s_); q2 = ring.interpolate(s_ + .5); nx0, ny0 = q2.y - q.y, -(q2.x - q.x); nl = math.hypot(nx0, ny0) or 1
        inw = 1 if p.contains(Point(q.x + nx0 / nl * .5, q.y + ny0 / nl * .5)) else -1
        for k in range(7):
            o = rr_.uniform(-.3, 1.6) * inw; bx_, by_ = q.x + nx0 / nl * o, q.y + ny0 / nl * o; a = rr_.uniform(0, 6.28); ln_ = rr_.uniform(.5, 1.2)
            X1, Y1 = px(bx_, by_); X2, Y2 = px(bx_ + math.cos(a) * ln_, by_ + math.sin(a) * ln_)
            cc = (int(rr_.uniform(70, 125)), int(rr_.uniform(100, 140)), int(rr_.uniform(40, 60)), 235)
            rd.line([(X1, Y1), (X2, Y2)], fill=cc, width=max(2, int(PPM * .3)))
ra = np.asarray(rl, np.float32) / 255; img[:] = img * (1 - ra[..., 3:]) + ra[..., :3] * ra[..., 3:]
# bunkers: grass lip, sand, NW lip shadow, SE lit wall, rake grain
bm = mask(polys.get('bunker', [])); bb = blur(bm, .035)
lip = blur(mask(polys.get('bunker', []), .9), .12); mul(np.clip(lip - bb, 0, 1), .78); fill(np.clip(lip - bb, 0, 1) * blur(np.clip(shift(bm, -.5, -.5) - bm, 0, 1), .2), np.array([.62, .74, .36]), .6)
fill(bb, C['sand']); img *= (1 + bb * (.025 * noise(.18, 10, 1) + .02 * noise(1.5, 11, 1)))[..., None]
sh = blur(np.clip(bb - shift(bb, 1.5, 1.5), 0, 1), .5); fill(np.clip(sh * 1.3, 0, 1) * bb, C['sand_s'], 1)
mul(blur(np.clip(shift(bb, .9, .9) - bb, 0, 1), .35), .8)
sh2 = blur(np.clip(bb - shift(bb, -.6, -.6), 0, 1), .3); mul(sh2 * bb, 1.04)
# paths, car park, bridge
pm = blur(mask(polys.get('path', [])), .08); fill(blur(mask(polys.get('path', []), .4), .1), C['path_e'], .8); fill(pm, C['path'])
img *= (1 + pm * .03 * noise(.3, 12, 1))[..., None]; mul(blur(np.clip(pm - shift(pm, -.3, .3), 0, 1), .1), .9)
for p in polys.get('path', []):
    if p.intersects(unary_union(polys.get('water', [Point(1e6, 1e6).buffer(1)]))) and p.area < 400:   # bridge deck
        dm = blur(mask([p]), .05); fill(dm, C['deck']); pl = (np.floor(wy / .6) % 2).astype(np.float32); mul(dm * pl, .9)
        mul(blur(np.clip(shift(dm, .8, .8) - dm, 0, 1), .3), .6)
# public roads and drives: tarmac with a soft verge line
if polys.get('road'):
    rm_ = blur(mask(polys['road']), .06); fill(blur(mask(polys['road'], .5), .1), C['path_e'], .6); fill(rm_, C['tarmac']); img[:] *= (1 + rm_ * .03 * noise(.4, 17, 1))[..., None]
# car park: tarmac, white bay lines, kerb
for p in polys.get('parking', []):
  if not crop_on(p.bounds, 6): continue
  if True:
    km = blur(mask([p], .8), .08); fill(km, C['path_e']); cm = blur(mask([p]), .05); fill(cm, C['tarmac']); img[:] *= (1 + cm * .04 * noise(.4, 14, 1))[..., None]
    bx0, by0, bx1, by1 = p.bounds; ln = np.zeros((H, W), np.float32)
    for row in (by0 + 6, by1 - 6):
        m_ = (np.abs(wy - row) < 5) & (np.abs(((wx - bx0) % 2.6) - 1.3) > 1.2) & (wx > bx0 + 3) & (wx < bx1 - 3); ln[m_] = 1
    fill(blur(ln, .03) * cm, np.array([.93, .93, .9]), .9)
    cols = [(.85, .85, .87), (.15, .16, .18), (.55, .58, .62), (.62, .1, .1), (.2, .28, .45), (.93, .93, .93), (.3, .32, .3)]
    for row in (by0 + 6, by1 - 6):
        for xc in np.arange(bx0 + 4.6, bx1 - 3, 2.6):
            if rng.random() < .45: continue
            cm2 = ((np.abs(wx - xc) < .9) & (np.abs(wy - row) < 2.2)).astype(np.float32); cm2 = blur(cm2, .05)
            mul(blur(np.clip(shift(cm2, .5, .5) - cm2, 0, 1), .2), .55); cc = np.array(cols[rng.integers(len(cols))])
            fill(cm2, cc); fill(cm2 * (np.abs(wy - row) < .8), cc * .55 + .1, .8); fill(cm2 * (np.abs(wy - row) < .7) * (wx < xc), np.array([1, 1, 1.]), .12)
  crop_off()
# buildings: drop shadow, two-tone roof (gable), parapet line
for b in polys.get('building', []):
  if not crop_on(b.bounds, 8): continue
  if True:
    bm_ = mask([b]); mul(blur(np.clip(shift(bm_, 3.5, 3.5) - bm_, 0, 1), .6), .55)
    mr = b.minimum_rotated_rectangle; cx, cy = b.centroid.x, b.centroid.y; c = np.array(mr.exterior.coords)
    e = c[1] - c[0] if np.linalg.norm(c[1] - c[0]) > np.linalg.norm(c[2] - c[1]) else c[2] - c[1]; e /= np.linalg.norm(e)
    bx0_, by0_, bx1_, by1_ = b.bounds; sl = (slice(max(0, int((y1 - by1_) * PPM) - 2), int((y1 - by0_) * PPM) + 2), slice(max(0, int((bx0_ - x0) * PPM) - 2), int((bx1_ - x0) * PPM) + 2))
    # straight-skeleton roof for (near-)orthogonal buildings = L-infinity distance in the building's own frame
    ang = math.atan2(e[1], e[0]); ca, sa = math.cos(-ang), math.sin(-ang); cs = np.array(b.exterior.coords) - [cx, cy]
    lc = np.c_[cs[:, 0] * ca - cs[:, 1] * sa, cs[:, 0] * sa + cs[:, 1] * ca]; res = .1
    lx0, ly0 = lc.min(0) - 1; lw, lh = (lc.max(0) - lc.min(0) + 2) / res
    lim = Image.new('L', (int(lw) + 1, int(lh) + 1), 0); ImageDraw.Draw(lim).polygon([((u - lx0) / res, (v - ly0) / res) for u, v in lc], fill=1)
    cdt = ndi.distance_transform_cdt(np.asarray(lim), metric='chessboard').astype(np.float32)
    gy_, gx_ = np.gradient(cdt); ax_ = np.abs(gx_) >= np.abs(gy_)
    lnx = np.where(ax_, -np.sign(gx_), 0); lny = np.where(ax_, 0, -np.sign(gy_))      # outward = downhill
    px_, py_ = wx[sl] - cx, wy[sl] - cy; u = px_ * ca - py_ * sa; v = px_ * sa + py_ * ca
    iu = np.clip(((u - lx0) / res).astype(int), 0, cdt.shape[1] - 1); iv = np.clip(((v - ly0) / res).astype(int), 0, cdt.shape[0] - 1)
    a_, b_ = lnx[iv, iu], lny[iv, iu]; c2, s2 = math.cos(ang), math.sin(ang)
    nx_, ny_ = a_ * c2 - b_ * s2, a_ * s2 + b_ * c2; best_ = cdt[iv, iu] * res
    lit = .5 + .5 * (nx_ * -.7 + ny_ * .7)          # outward normal facing NW (world: -x, +y) is lit
    roofc = C['roof'][None, None] * (.78 + .38 * lit)[..., None] * (1 + .03 * np.sin(best_ * 2 * math.pi / .35))[..., None]
    mm = bm_[sl][..., None]; img[sl] = img[sl] * (1 - mm) + roofc * mm
    fill(blur(np.clip(bm_ - ndi.binary_erosion(bm_ > .5, iterations=1), 0, 1), .03), C['roof2'] * .7, .7)
  crop_off()
# --- terrain: course DEM + shaped greens, tees, bunker faces, small mounds in the rough
if 'elev' in d:
    e = d['elev']; Z0 = np.array(e['q'], np.float32).reshape(e['h'], e['w']) / 4 + e['zmin']
    ro = -d.get('rot', 0); ex_ = wx * math.cos(ro) - wy * math.sin(ro); ey_ = wx * math.sin(ro) + wy * math.cos(ro)
    Z = ndi.map_coordinates(Z0, [(e['y1'] - ey_) / e['step'], (ex_ - e['x0']) / e['step']], order=3, mode='nearest')
    Z += .9 * blur(green_m, 2.5) + .5 * blur(green_m, .8) + .8 * blur(tee_m, .7)
    bun = mask(polys.get('bunker', [])); Z += -.7 * blur(bun, .5) + .55 * blur(np.clip(mask(polys.get('bunker', []), 3) - bun, 0, 1), 1)
    Z += .3 * noise(9, 31, 1) * (1 - blur(play, 3)) + .05 * noise(4, 32, 1)
    Z = np.where(wm > .5, ndi.grey_erosion(Z, size=3).min() if False else Z, Z)
    gyz, gxz = np.gradient(Z, 1 / PPM); nz = 1 / np.sqrt(1 + gxz ** 2 + gyz ** 2)
    Ld = np.array([-.55, .55, .63]); Ld /= np.linalg.norm(Ld)           # from NW: world -x, +y  (image row grows south: gy is d/drow)
    shade = (-gxz * Ld[0] + gyz * Ld[1] + Ld[2]) * nz; flat = Ld[2]
    hs = np.clip((shade - flat) * 1.4, -.3, .25) * (1 - wb * .9)
    img[:] = img * (1 + hs[..., None])
# --- trees: AO on grass, shadows, sprites
sys.path.insert(0, '/tmp/claude-0/osm/stamp'); import trees2d; LIB = trees2d.library()
def conifer(seed, S=384):
    r = np.random.default_rng(seed); yy_, xx_ = np.mgrid[0:S, 0:S] - S / 2 + .5; rr = np.hypot(xx_, yy_) / (S * .37); th = np.arctan2(yy_, xx_)
    n = r.integers(22, 30); spike = .82 + .18 * np.abs(np.sin(n / 2 * th + r.uniform(0, 6))) ** .6 + .05 * np.sin(5 * th + r.uniform(0, 6))
    a = np.clip((spike - rr) * 18, 0, 1)
    sh = np.clip(1 - .55 * rr + .25 * (-(xx_ + yy_) / S) , 0, 1.2)
    tex = 1 + .18 * np.sin(n * th) * rr + .1 * ndi.gaussian_filter(r.standard_normal((S, S)), 2)
    base = np.array([.16, .30, .17]) * r.uniform(.85, 1.15)
    rgb = base[None, None] * (sh * tex)[..., None]; return np.dstack([np.clip(rgb, 0, 1), a]).astype(np.float32)
cr = sorted(d['crowns'], key=lambda c: -c[1])
TA = np.zeros((H, W), np.float32)   # real leafy alpha of every drawn tree (for the outline)
tm_ = np.zeros((H, W), np.float32); shd = np.zeros((H, W), np.float32)
for c in cr:
    X, Y = px(c[0], c[1]); R = c[2] * PPM; yy0, xx0 = int(Y - R * 2.6), int(X - R * 2.6)
    sl = (slice(max(0, yy0), min(H, int(Y + R * 2.6))), slice(max(0, xx0), min(W, int(X + R * 2.6))))
    gy_, gx_ = np.mgrid[sl]; tm_[sl] = np.maximum(tm_[sl], np.clip(1.15 - np.hypot(gx_ - X, gy_ - Y) / R, 0, 1))
    g_ = len(c) > 3 and c[3] == 'gorse'; o = R * (.35 if g_ else 1.05); shd[sl] = np.maximum(shd[sl], np.clip(1.1 - np.hypot(gx_ - X - o, gy_ - Y - o) / (R * .98), 0, 1))
mul(blur(tm_, 4) * 1.4, .88)                     # ambient occlusion under canopy edges
mul(blur(shd, .6) * (1 - tm_ * .0), .62)          # cast shadows
dens = blur((tm_ > .05).astype(np.float32), 2.5); floor = blur(np.clip((dens - .45) * 2.5, 0, 1), 1)
fill(floor, hexc('#3A5226'), .35)
for i, c in enumerate(cr):
    typ = c[3] if len(c) > 3 else 'round'; spr = LIB[typ][rng.integers(len(LIB[typ]))]
    R = c[2] * PPM; S = max(4, int(2 * R / .80))
    im = Image.fromarray((spr * 255).astype(np.uint8)).rotate(0).resize((S, S), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255; X, Y = px(c[0], c[1]); ox, oy = int(X - S / 2), int(Y - S / 2)
    ys, xs = slice(max(0, oy), min(H, oy + S)), slice(max(0, ox), min(W, ox + S)); a = a[ys.start - oy:ys.stop - oy, xs.start - ox:xs.stop - ox]
    jit = np.array([rng.uniform(.95, 1.1), rng.uniform(.97, 1.1), rng.uniform(.9, 1.05)]) * rng.uniform(1.02, 1.2)
    al = a[..., 3:4]; img[ys, xs] = img[ys, xs] * (1 - al) + np.clip(a[..., :3] * jit, 0, 1) * al
    TA[ys, xs] = np.maximum(TA[ys, xs], a[..., 3])
# flag
for fx, fy in d['pins']:
    X, Y = px(fx, fy)
    dr = Image.new('L', (W, H), 0); g = ImageDraw.Draw(dr); g.line([(X, Y), (X + 7 * PPM * .6, Y + 7 * PPM * .6)], fill=255, width=max(2, int(PPM * .35)))
    mul(blur(np.asarray(dr, np.float32) / 255, .1), .55)
    fl = Image.new('L', (W, H), 0); g = ImageDraw.Draw(fl); g.ellipse([X - PPM * .7, Y - PPM * .7, X + PPM * .7, Y + PPM * .7], fill=255)
    g.polygon([(X, Y - PPM * .3), (X + PPM * 3.2, Y - PPM * 1.2), (X + PPM * .2, Y - PPM * 2.4)], fill=0)
    fill(np.asarray(fl, np.float32) / 255, np.array([1, 1, 1.]))
    ff = Image.new('L', (W, H), 0); g = ImageDraw.Draw(ff); g.polygon([(X, Y), (X + PPM * 3.4, Y - PPM * 1.0), (X + PPM * .4, Y - PPM * 2.6)], fill=255)
    fill(blur(np.asarray(ff, np.float32) / 255, .05), np.array([.82, .12, .12]))
# optional hole numbers (customer tick box): small white discs beside each tee
import os
if os.environ.get('NUMS') == '1' and len(LINES) > 1:
    from PIL import ImageFont
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); od = ImageDraw.Draw(ov); rpx = max(9, int(5.5 * PPM))
    fnt = ImageFont.truetype('/home/user/avendro-web-v2/partee-robot/fonts/jost500.ttf', int(rpx * 1.15))
    for h in d['holes']:
        if len(h['pts']) < 2: continue
        (ax, ay), (bx, by) = h['pts'][0], h['pts'][1]; Ln = math.hypot(bx - ax, by - ay) or 1
        X, Y = px(ax - (bx - ax) / Ln * 14, ay - (by - ay) / Ln * 14)
        od.ellipse([X - rpx + 2, Y - rpx + 4, X + rpx + 2, Y + rpx + 4], fill=(0, 0, 0, 70))
        od.ellipse([X - rpx, Y - rpx, X + rpx, Y + rpx], fill=(255, 255, 255, 245), outline=(40, 70, 45, 255), width=max(1, rpx // 7))
        od.text((X, Y + 1), str(h['ref']), font=fnt, fill=(35, 60, 40, 255), anchor='mm')
    oa = np.asarray(ov, np.float32) / 255; img[:] = img * (1 - oa[..., 3:]) + oa[..., :3] * oa[..., 3:]
# grade: gentle S-curve + warm highlights
img[:] = np.clip(img, 0, 1); lum = img.mean(-1, keepdims=True); img[:] = lum + (img - lum) * .9
img[:] = img + .10 * (img - img ** 2) * (img - .4) * 4; img[:] *= np.array([1.015, 1.0, .965]) * 1.05
# island: hole corridor + features, organic edge, feathered on paper
if 'own' in d:
    core = unary_union([LineString(smooth_line(l_, 6)).buffer(62 if len(LINES) == 1 else 30) for l_ in LINES] + [P(['x', c]).buffer(30) for c in d['own']]).buffer(30).buffer(-30)
    if len(LINES) > 1:   # whole course: pull in the clubhouse, car parks and buildings next to the holes
        _hl = unary_union([LineString(l_) for l_ in LINES])
        # only around the clubhouse (1st tee, 9th/18th green, 10th tee): housing estates and factories beside the fairways stay off the print
        _an = [LINES[0][0], LINES[-1][-1]] + ([LINES[9][0], LINES[8][-1]] if len(LINES) >= 10 else [])
        _cz = unary_union([Point(a).buffer(260) for a in _an])
        _ex = [p.buffer(15) for k in ('building', 'parking') for p in polys.get(k, []) if p.intersects(_cz) and p.area < 6000 and (p.distance(_hl) < 90 or (k == 'building' and p.area > 600 and p.distance(_hl) < 220))]
        if _ex: core = unary_union([core] + _ex).buffer(45).buffer(-45)
else:
    core = unary_union([LineString(line).buffer(85)] + [p.buffer(14) for k in ('green', 'fairway', 'tee', 'building', 'heath', 'parking') for p in polys.get(k, [])] + [p.buffer(8) for p in polys.get('water', []) if p.area > 2000] + [p.buffer(10) for p in polys.get('path', []) if p.area > 1500])
core = unary_union([Polygon(g.exterior) for g in getattr(core, 'geoms', [core]) if g.geom_type == 'Polygon'])   # no holes inside the course
cm_ = mask([core]); sd = (ndi.distance_transform_edt(cm_ > .5) - ndi.distance_transform_edt(cm_ <= .5)) / PPM
wob = 5 * noise(25, 51, 1) + 2 * noise(8, 52, 1) + .8 * noise(3, 53, 1); edge_ = sd + wob
grass = blur((edge_ > 0).astype(np.float32), 1.6); grass = np.clip((grass - .15) / .7, 0, 1) ** 1.3
treea = np.zeros((H, W), np.float32)                                 # whole trees on the edge break the outline
for c in d['crowns']:
    X, Y = px(c[0], c[1]); xi, yi = int(X), int(Y)
    if not (0 <= xi < W and 0 <= yi < H) or not (-7 < edge_[yi, xi] < 3): continue
    if X - c[2] * PPM * 1.3 < 0 or Y - c[2] * PPM * 1.3 < 0 or X + c[2] * PPM * 1.3 >= W or Y + c[2] * PPM * 1.3 >= H: continue   # never show a tree cut by the frame
    R = c[2] * PPM * 1.45; sl = (slice(max(0, int(Y - R)), min(H, int(Y + R) + 1)), slice(max(0, int(X - R)), min(W, int(X + R) + 1)))
    gy_, gx_ = np.mgrid[sl]; treea[sl] = np.maximum(treea[sl], np.clip((R - np.hypot(gx_ - X, gy_ - Y)) / (PPM * .35), 0, 1))
treea = np.clip(treea * 4, 0, 1) * TA   # follow each tree's own leafy outline, not a circle
isl = np.maximum(grass, treea)
lab, nl = ndi.label(isl > .3); keep = np.zeros(nl + 1, bool)
for i_, sz in enumerate(ndi.sum(np.ones_like(isl), lab, range(1, nl + 1)), 1): keep[i_] = sz > .08 * (isl > .3).sum()
isl *= ndi.binary_dilation(keep[lab], iterations=int(PPM * 2))
mul(np.clip(1 - grass, 0, 1) * (isl > 0) * (1 - treea), .9)
ys, xs = np.where(isl > .02); S_ = max(ys.max() - ys.min(), xs.max() - xs.min()); M_ = max(60, int(S_ * .055))
paper_ = np.ones((H, W, 1), np.float32)
if os.environ.get('LIFT', '1') == '1':   # style A (Shane 22 Nov): the course sits lifted off the paper with a soft shadow + fine paper grain
    import cv2
    def _sh(dy, sig, op): return cv2.GaussianBlur(np.roll(isl, int(dy), 0), (0, 0), sig) * op
    paper_ = ((1 - _sh(S_ * .012, S_ * .018, .30)) * (1 - _sh(S_ * .003, S_ * .004, .25)))[..., None]
    rg = np.random.default_rng(7); gr = cv2.GaussianBlur(rg.normal(0, 1, (H, W)).astype(np.float32), (0, 0), max(.8, PPM * .35)) * .012
    paper_ = paper_ * (1 + gr[..., None])
outimg = C['paper'][None, None] * paper_ * (1 - isl[..., None]) + img * isl[..., None]
outimg *= (1 + isl[..., None] * (.006 * noise(.12, 41, 1))[..., None])
outimg = outimg[max(0, ys.min() - M_):ys.max() + M_, max(0, xs.min() - M_):xs.max() + M_]
Image.fromarray((np.clip(outimg, 0, 1) * 255).astype(np.uint8)).save(out, quality=92)
_cr = (slice(max(0, ys.min() - M_), ys.max() + M_), slice(max(0, xs.min() - M_), xs.max() + M_))
if os.environ.get('ALPHA') == '1':   # marketing: island on transparent background + edge-to-edge version, same pixel frame
    rgba = np.dstack([np.clip(img[_cr], 0, 1), np.clip(isl[_cr], 0, 1)])
    Image.fromarray((rgba * 255).astype(np.uint8), 'RGBA').save(out.rsplit('.', 1)[0] + '-island.png')
    Image.fromarray((np.clip(img[_cr], 0, 1) * 255).astype(np.uint8)).save(out.rsplit('.', 1)[0] + '-fullbleed.png')
    json.dump({'x0': int(_cr[1].start), 'y0': int(_cr[0].start), 'pins_px': [list(px(*p_)) for p_ in d.get('pins', [])], 'tee_px': [list(px(*h_['pts'][0])) for h_ in d['holes']]}, open(out.rsplit('.', 1)[0] + '-frame.json', 'w'))
if os.environ.get('SAT') == '1' and 'sat' in d:   # the satellite photo in exactly the same frame, for the swipe check
    import cv2
    sa = d['sat']; sim = cv2.imread(sa['path']); ro = -d.get('rot', 0)
    ex_ = wx * math.cos(ro) - wy * math.sin(ro); ey_ = wx * math.sin(ro) + wy * math.cos(ro)
    sv = cv2.remap(sim, ((ex_ - sa['x0']) / sa['mpp']).astype(np.float32), ((sa['y1'] - ey_) / sa['mpp']).astype(np.float32), cv2.INTER_LINEAR)
    sv = sv[max(0, ys.min() - M_):ys.max() + M_, max(0, xs.min() - M_):xs.max() + M_]
    cv2.imwrite(out.replace('.jpg', '-sat.jpg'), sv, [cv2.IMWRITE_JPEG_QUALITY, 88])
print('done', out, outimg.shape, f'{time.time() - T0:.1f}s')
