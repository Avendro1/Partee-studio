# Estate style: an engraved estate map in one deep-green ink on ivory, drawn from the traced vectors (GEODUMP from plan.py)
import sys, json, numpy as np, cv2
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.ops import unary_union
geo, fin, nop, out = sys.argv[1:5]
G = json.load(open(geo)); W, H = G['W'], G['H']; S = 2
PAPER = np.array((244, 238, 224), np.float32); INK = np.array((18, 53, 42), np.float32)
def polys(k): return [Polygon(r[0], r[1:]).buffer(0) for r in G[k] if len(r[0]) > 2]
def mask(geoms, grow=0):
    m = Image.new('L', (W * S, H * S), 0); d = ImageDraw.Draw(m)
    for g in geoms:
        g = g.buffer(grow) if grow else g
        for p in ([g] if g.geom_type == 'Polygon' else list(getattr(g, 'geoms', []))):
            if p.is_empty or p.geom_type != 'Polygon': continue
            d.polygon([(x * S, y * S) for x, y in p.exterior.coords], fill=255)
            for i in p.interiors: d.polygon([(x * S, y * S) for x, y in i.coords], fill=0)
    return np.asarray(m, np.float32) / 255
def outline(geoms, w):
    m = Image.new('L', (W * S, H * S), 0); d = ImageDraw.Draw(m)
    for g in geoms:
        for p in ([g] if g.geom_type == 'Polygon' else list(getattr(g, 'geoms', []))):
            if p.is_empty or p.geom_type != 'Polygon': continue
            for r in [p.exterior] + list(p.interiors): d.line([(x * S, y * S) for x, y in r.coords], fill=255, width=int(w * S), joint='curve')
    return np.asarray(m, np.float32) / 255
yy, xx = np.mgrid[0:H * S, 0:W * S].astype(np.float32)
def hatch(ang, sp, w):
    u = (xx * np.cos(ang) + yy * np.sin(ang)) / sp; dd = np.abs(u - np.round(u)) * sp
    return np.clip(w / 2 - dd + .5, 0, 1)
# base: the finished print in duotone (title, hole strip, surroundings keep their layout)
f = np.asarray(Image.open(fin).convert('RGB').resize((W * S, H * S), Image.LANCZOS), np.float32)
L = (f @ np.array([.3, .59, .11], np.float32)) / 255
t = np.clip((.93 - L) / .78, 0, 1) ** 1.1
cov = t.copy()
land, fair, green, tee, bunk, water = (polys(k) for k in ('land', 'fair', 'green', 'tee', 'bunk', 'water'))
mL = mask(land)
cov = cov * (1 - mL)
_lb = max(y for r in G['land'] for x, y in r[0]) * S
_sb = (yy > _lb + 40 * S) & (yy < H * S * .855)
_sh = cv2.morphologyEx(((t > .3) & _sb).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9 * S, 9 * S), np.uint8))
_n, _lab, _st, _ = cv2.connectedComponentsWithStats(_sh); _k = np.zeros(_n, np.float32); _k[1:] = _st[1:, 4] > 900 * S * S
_sh = cv2.GaussianBlur(_k[_lab], (0, 0), .8)
cov = np.where(_sb, np.maximum(cov, _sh * .92), cov)
# rough: fine diagonal engraving
cov += mL * hatch(np.pi / 4, 7.0, 1.05) * .55
mF = mask(fair + tee); mG = mask(green); mB = mask(bunk); mW = mask(water)
turf = np.clip(mF + mG, 0, 1)
cov = cov * (1 - turf)                                   # fairways + greens: clean paper
cov += mG * .10 + mF * hatch(0, 9.0, .7) * .12            # faint mowing lines on fairways, light tint on greens
cov = cov * (1 - mB) + mB * 0                             # bunkers: paper, stippled below
rng = np.random.default_rng(7); st = (rng.random((H * S, W * S)) < .055).astype(np.float32)
st = cv2.dilate(st, np.ones((2, 2), np.uint8)); cov += mB * st * .75
# water: close horizontal lines, rippled rings outside the bank
cov = cov * (1 - mW) + mW * np.maximum(hatch(np.pi / 2, 4.2, 1.6) * .85, .18)
rip = np.zeros_like(cov)
for k, g in enumerate((3.2, 6.4)):
    rip = np.maximum(rip, outline([p.buffer(g) for p in water], .55) * (.55 - k * .2))
cov = np.maximum(cov, rip * mL * (1 - turf))
# trees: solid engraved crowns, cast shadow to SE in cross hatch, small lit nick to NW
sh = Image.new('L', (W * S, H * S), 0); sd = ImageDraw.Draw(sh); tr = Image.new('L', (W * S, H * S), 0); td = ImageDraw.Draw(tr); hl = Image.new('L', (W * S, H * S), 0); hd = ImageDraw.Draw(hl)
for x, y, r in G['crowns']:
    X, Y, R = x * S, y * S, max(1.6, r) * S; o = R * .55
    sd.ellipse([X - R + o, Y - R + o, X + R + o, Y + R + o], fill=255)
for x, y, r in sorted(G['crowns'], key=lambda q: q[1]):
    X, Y, R = x * S, y * S, max(1.6, r) * S
    td.ellipse([X - R, Y - R, X + R, Y + R], fill=255); hd.ellipse([X - R, Y - R, X + R, Y + R], fill=0)
    h = R * .32; hd.ellipse([X - R * .38 - h, Y - R * .38 - h, X - R * .38 + h, Y - R * .38 + h], fill=255)
# traced woodland / scrub with no single crowns: small engraved bushes scattered through the real outline
from shapely.prepared import prep; from shapely.geometry import Point
rs = np.random.default_rng(9); pxm = G['px_m']; rg = Image.new('L', (W * S, H * S), 0); rd = ImageDraw.Draw(rg)
for q in polys('scrub'):
    if q.area < 4: continue
    pq = prep(q); bx0, by0, bx1, by1 = q.bounds; n_ = int(q.area / (34 * pxm * pxm))
    for x, y in [(x, y) for x, y in zip(rs.uniform(bx0, bx1, n_ * 3), rs.uniform(by0, by1, n_ * 3)) if pq.contains(Point(x, y))][:n_]:
        X, Y, R = x * S, y * S, max(1.2, rs.uniform(2.0, 3.2) * pxm) * S
        R *= .8; rd.ellipse([X - R, Y - R, X + R, Y + R], fill=255)
sh, tr, hl = (np.asarray(m, np.float32) / 255 for m in (sh, tr, hl))
cov = np.maximum(cov, sh * (1 - tr) * hatch(-np.pi / 4, 4.0, 1.4) * .8)
cov = np.maximum(cov, np.asarray(rg, np.float32) / 255 * .9)
cov = cov * (1 - tr) + tr * (.96 - hl * .5)
# ink outlines: course edge, fairways, greens, tees, bunkers, water
for g, w, a in ((land, 2.6, 1), (fair + tee, 1.3, .95), (green, 1.6, 1), (bunk, 1.0, .9), (water, 1.8, 1)):
    cov = np.maximum(cov, outline(g, w) * a)
cov = np.clip(cov, 0, 1)
# hole markers + routing from the finished print (pixels that differ from the no-markers render)
n = np.asarray(Image.open(nop).convert('RGB').resize((W * S, H * S), Image.LANCZOS), np.float32)
dm = (np.abs(f - n).max(2) > 26).astype(np.uint8); dm = cv2.dilate(dm, np.ones((5, 5), np.uint8)); dm = cv2.morphologyEx(dm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
nc, lab, stt, _ = cv2.connectedComponentsWithStats(dm); keep = np.zeros(nc, np.uint8); keep[1:] = stt[1:, 4] > 2500 * S * S / 4   # tee markers only (pins dropped)
dm = keep[lab].astype(np.float32)
cov = cov * (1 - dm) + t * dm
o = PAPER * (1 - cov[..., None]) + INK * cov[..., None]
im = Image.fromarray(np.clip(o, 0, 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
a = np.asarray(im, np.float32); g = cv2.GaussianBlur(np.random.default_rng(3).normal(0, 1, (H, W)).astype(np.float32), (0, 0), .7) * 3.2
Image.fromarray(np.clip(a + g[..., None], 0, 255).astype(np.uint8)).save(out, quality=93)
print(out)
