# plan.py IN.json(course_input output) COURSE.json OUT.jpg   env SIZE=WxH, MSG
# Partee "Course Plan" (24 Nov 2026, Shane: map looks like an AI satellite picture / Word document; wants it to look expensive and hard to copy).
# An architect's ink plan drawn from our traced data: real elevation contours, scalloped tree clumps from the real crowns, stippled sand,
# water lines, line of play with numbered tees, plus a strip of all 18 holes drawn on their own. Nothing invented: every mark is a traced feature.
import json, sys, os, math, re
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
from shapely import affinity
from shapely.prepared import prep
HERE = os.path.dirname(os.path.abspath(__file__)); FD = HERE + '/../../fonts/'
F = lambda n, s: ImageFont.truetype(FD + n + '.ttf', max(6, int(s)))
d = json.load(open(sys.argv[1])); c = json.load(open(sys.argv[2])); out = sys.argv[3]
if os.environ.get("TREEFIX"):   # 27 Nov: per-course tree fixes checked on the satellite (fixtrees.py): add island copses, drop reed 'crowns' on lake banks
    _tf = json.load(open(os.environ["TREEFIX"])); _dr = set(_tf.get("drop", []))
    d["crowns"] = [q for i, q in enumerate(d.get("crowns", [])) if i not in _dr] + _tf.get("add", [])
if os.environ.get("EXTRA_CROWNS"):
    from scipy.spatial import cKDTree as _KD
    _ex = json.load(open(os.environ["EXTRA_CROWNS"])); _cr = d.get("crowns", [])
    if _cr: _kd = _KD([(q[0], q[1]) for q in _cr]); _ex = [q for q in _ex if _kd.query((q[0], q[1]))[0] > 3.5]   # no double trees where the trace already has one
    d["crowns"] = _cr + _ex   # 27 Nov heritage: real canopy from satellite (canopy.py), so tree density matches the ground
PV = bool(c.get("preview"))   # satellite preview (26 Nov): holes not routed yet, so no line of play, flags or hole strip
PAPER = (240, 234, 219); INK = (36, 46, 38); INK2 = (110, 108, 94); RED = (150, 70, 48)
LANDF = (232, 228, 208); FAIR = (196, 210, 160); GRN = (170, 196, 128); TREEF = (134, 150, 114); WATERF = (196, 210, 206); WATERL = (92, 124, 132); SANDF = (246, 240, 222); SANDD = (150, 130, 96); BLD = (214, 206, 188)
# HERITAGE (27 Nov, Shane: 'get this exact look'): same traced drawing, full daylight colour like a hand-painted estate map:
# vivid striped fairways, deep shaded tree canopies, blue water, cream hole markers in ink, double-rule border, deep green title.
HER = os.environ.get('PLAN_STYLE') == 'heritage'
if HER: PAPER = (238, 224, 206); INK = (41, 59, 29); INK2 = (104, 98, 78); LANDF = (180, 172, 124); FAIR = (110, 126, 50); GRN = (124, 142, 58); TREEF = (66, 84, 40); WATERF = (76, 118, 126); WATERL = (52, 80, 86); SANDF = (232, 220, 192); SANDD = (176, 156, 116); BLD = (226, 214, 196)   # sampled from Shane's reference 27 Nov
def polys(g):
    for p in getattr(g, 'geoms', [g]):
        if p.geom_type == 'Polygon' and not p.is_empty: yield p
        elif hasattr(p, 'geoms'): yield from polys(p)
FE = {}
for k, cc in d['feats']:
    if len(cc) < 6: continue
    p = Polygon(list(zip(cc[0::2], cc[1::2]))).buffer(0)
    if not p.is_empty and not (k == 'bunker' and p.area > 3000): FE.setdefault(k, []).append(p)
HL = [h for h in d['holes'] if len(h['pts']) > 1]
HOLE = os.environ.get('HOLE')   # single-hole print: just this hole's corridor, tee at the bottom, no strip
if HOLE: HL = [h for h in HL if str(h['ref']) == HOLE]
hl = unary_union([LineString(h['pts']) for h in HL]); near = hl.buffer(60)
HW = max(65, min(150, hl.length * .3)) if HOLE else 0   # single hole: show a real-width slice of the course around it
if HOLE: near = unary_union([LineString(h['pts']) for h in d['holes'] if len(h['pts']) > 1]).buffer(60).intersection(hl.buffer(HW + 10))
def clean(k, r, minA):
    g = unary_union([p for p in FE.get(k, []) if p.intersects(near)])
    if g.is_empty: return g
    g = g.buffer(r, 16).buffer(-2 * r, 16).buffer(r, 16)
    return unary_union([Polygon(p.exterior, [i for i in p.interiors if Polygon(i).area > 300]).simplify(.4) for p in polys(g) if p.area > minA])
fair, tee, green, bunk, water = clean('fairway', 4, 100), clean('tee', 1, 12), clean('green', 1.5, 40), clean('bunker', .6, 8), clean('water', 2.5, 150)
bld = unary_union([p for p in FE.get('building', []) if p.intersects(near)])
land = unary_union([hl.buffer(42, 32)] + [g.buffer(14, 16) for g in (fair, green, tee) if not g.is_empty]).buffer(40, 32).buffer(-40, 32).buffer(-10, 32).buffer(10, 32).simplify(.8)
land = unary_union([Polygon(p.exterior) for p in polys(land) if p.area > 5000])
if HOLE:   # the course as it really is, within a rounded slice around this hole (neighbouring holes cut by the slice)
    _all = unary_union([LineString(h['pts']) for h in d['holes'] if len(h['pts']) > 1])
    _cl = unary_union([_all.buffer(42, 32)] + [g.buffer(14, 16) for g in (fair, green, tee) if not g.is_empty]).buffer(40, 32).buffer(-40, 32)
    land = _cl.intersection(hl.buffer(HW, 48)).buffer(-6, 32).buffer(6, 32).simplify(.8)
    land = unary_union([Polygon(p.exterior) for p in polys(land) if p.area > 2000])
fair, tee, green, bunk, water, bld = [g.intersection(land.buffer(-2)) for g in (fair, tee, green, bunk, water, bld)]
if HOLE:   # only this hole is drawn: its own fairway/green/tees and the bunkers that guard it; every other hole's turf becomes rough
    _lb = hl.buffer(30, 16)
    def _own(g, k, dmax):   # keep the traced pieces that belong to this hole (raw shapes, so a neighbour's fairway that touches isn't dragged in)
        raw = unary_union([q for q in FE.get(k, []) if q.distance(hl) < dmax and (q.intersection(_lb).area / max(q.area, 1) > .35 or q.area < 1500)])
        return g.intersection(raw.buffer(2, 8)) if not raw.is_empty else Polygon()
    fair, tee, green = _own(fair, 'fairway', 6), _own(tee, 'tee', 25), _own(green, 'green', 15)
    bunk = unary_union([q for q in polys(bunk) if q.distance(hl) < 32 or q.distance(unary_union([fair, green])) < 10])
LINKS = bool(c.get('links') or c.get('coastal_links') or c.get('no_trees'))
if os.environ.get('TREEFIX') and _tf.get('islands'): water = water.difference(unary_union([Polygon(q).buffer(1.5) for q in _tf['islands']]))   # wooded islands are land, not lake
if LINKS and not HER: LANDF = (224, 221, 190); FAIR = (196, 218, 146); GRN = (140, 188, 98)
if LINKS and HER: LANDF = (198, 196, 146)   # links: tawny dune-grass ground, so the course doesn't fade into the paper
crowns = [] if c.get('no_trees') else [(x, y, r) for x, y, r, *k in d.get('crowns', []) if (k[0] if k else 'round') != 'gorse']
gorse = [] if c.get('no_trees') else [(x, y, r) for x, y, r, *k in d.get('crowns', []) if k and k[0] == 'gorse']
inner = prep(land.buffer(-2)); playg = unary_union([g for g in (fair, green, tee, bunk, water) if not g.is_empty]).buffer(1); pplay = prep(playg)
crowns = [(x, y, r) for x, y, r in crowns if inner.contains(Point(x, y)) and not pplay.contains(Point(x, y))]
woods = unary_union([Point(x, y).buffer(r * .62, 8) for x, y, r in crowns]).difference(playg).intersection(land) if crowns else Polygon()
# ---- page
sz = [int(v) for v in os.environ.get('SIZE', '3543x4724').split('x')]
def _layout(lo):
    W_, H_ = (max(sz), min(sz)) if lo else (min(sz), max(sz)); U_ = min(W_, H_) / 100; MG_ = min(W_, H_) * .075
    ST_ = (H_ * .655, H_ * .855) if lo else (H_ * .655, H_ * .868)
    if HOLE or PV: ST_ = (H_ * .855, H_ * .855)
    return W_, H_, U_, MG_, ST_, W_ - 2 * MG_, ST_[0] - U_ * 3 - MG_
# rotate the plan (like architects do) when that lets a long, thin course fill the page; north arrow follows
best = None; lc = land.centroid
for lo in ((False,) if HOLE else (True, False)):
    W_, H_, U_, MG_, ST_, aw, ah = _layout(lo)
    for th in range(-90, 91, 3):
        b = affinity.rotate(land, th, origin=lc).bounds; sc = min(aw / (b[2] - b[0]), ah / (b[3] - b[1]))
        if th == 0: sc *= 1.18   # prefer north-up unless rotating gains a lot
        if best is None or sc > best[0]: best = (sc, lo, th)
_, land_o, ROT = best
print('auto rot', ROT)
if HOLE:
    _p = HL[0]['pts']; ROT = 90 - math.degrees(math.atan2(_p[-1][1] - _p[0][1], _p[-1][0] - _p[0][0]))
if os.environ.get('ROT'): ROT = float(os.environ['ROT'])
W, H, U, MG, STRIP, _aw, _ah = _layout(land_o)
SS = int(os.environ.get('PLAN_SS', '2' if min(W, H) < 3000 else '1'))   # supersample; print-size files (3600+ px) are already 300 dpi and SS=2 there needs >14 GB
top_, bot_ = MG, STRIP[0] - U * 3
_rl = affinity.rotate(land, ROT, origin=lc); x0, y0, x1, y1 = _rl.bounds
s = min((W - 2 * MG) / (x1 - x0), (bot_ - top_) / (y1 - y0))
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; ox, oy = W / 2, (top_ + bot_) / 2
im = Image.new('RGB', (W * SS, H * SS), PAPER); dr = ImageDraw.Draw(im)
_ca, _sa = math.cos(math.radians(ROT)), math.sin(math.radians(ROT)); _lx, _ly = lc.x, lc.y
def mk(sx, ox_, oy_, cx_, cy_):
    return lambda x, y: ((ox_ + (x - cx_) * sx) * SS, (oy_ - (y - cy_) * sx) * SS)
def T(x, y):
    dx, dy = x - _lx, y - _ly; xr, yr = _lx + dx * _ca - dy * _sa, _ly + dx * _sa + dy * _ca
    return ((ox + (xr - cx) * s) * SS, (oy - (yr - cy) * s) * SS)
if os.environ.get('TDUMP'):   # world (metres) -> poster px, for drawing shots/flag on top
    _a = [T(0, 0), T(100, 0), T(0, 100)]; _a = [(x / SS, y / SS) for x, y in _a]
    json.dump({'m': [(_a[1][0] - _a[0][0]) / 100, (_a[2][0] - _a[0][0]) / 100, _a[0][0], (_a[1][1] - _a[0][1]) / 100, (_a[2][1] - _a[0][1]) / 100, _a[0][1]], 'W': W, 'H': H}, open(os.environ['TDUMP'], 'w'))
def grid_affine(step_):   # DEM grid pixel (col,row) -> canvas, honouring rotation
    a0 = T(e['x0'] - e['step'] / 2, e['y1'] + e['step'] / 2); a1 = T(e['x0'] - e['step'] / 2 + step_ * 100, e['y1'] + e['step'] / 2); a2 = T(e['x0'] - e['step'] / 2, e['y1'] + e['step'] / 2 - step_ * 100)
    return cv2.getAffineTransform(np.float32([[0, 0], [100, 0], [0, 100]]), np.float32([a0, a1, a2]))
LW = lambda k: max(1, int(U * k * SS))
def fill(g, col, T_, hole=LANDF):
    for p in polys(g):
        dr.polygon([T_(*q) for q in p.exterior.coords], fill=col)
        for i in p.interiors: dr.polygon([T_(*q) for q in i.coords], fill=hole)
def line(g, col, w, T_):
    for p in polys(g):
        for r_ in [p.exterior] + list(p.interiors): dr.line([T_(*q) for q in r_.coords], fill=col, width=w, joint='curve')
def stipple(g, T_, sx, dens=.9, col=SANDD):
    rng = np.random.default_rng(1)
    for p in polys(g):
        a = p.area; n = int(a * dens); bx0, by0, bx1, by1 = p.bounds; pp = prep(p)
        xs = rng.uniform(bx0, bx1, n * 2); ys = rng.uniform(by0, by1, n * 2); k = 0
        rr = max(.5, U * .03) * SS
        for x, y in zip(xs, ys):
            if k >= n: break
            if pp.contains(Point(x, y)):
                X, Y = T_(x, y); dr.ellipse([X - rr, Y - rr, X + rr, Y + rr], fill=col); k += 1
def waterlines(g, T_, sx, w):
    for i, off in enumerate((-1.6, -3.4, -5.6)):
        gg = g.buffer(off / max(sx, .2) * 2.2, 8)
        if gg.is_empty: break
        line(gg, WATERL + (), max(1, w - (1 if i else 0)), T_)
# SEA (24 Nov, Shane: "beside the sea... shade in the sea"): real OSM coastline/tidal flats from CTX=ctx.json (fetch.py), drawn only in
# the margin around the course, fading out from the shore, with old-chart water-lines. Course size/position unchanged.
CTX = os.environ.get('CTX')
if CTX and os.path.exists(CTX):
    cx_ = json.load(open(CTX)); P_ = lambda k: unary_union([Polygon(list(zip(q[0::2], q[1::2]))).buffer(0) for q in cx_.get(k, []) if len(q) >= 6])
    sea, tid, bch = P_("sea"), unary_union([P_("tidal"), P_("marsh")]), unary_union([P_("beach"), P_("sand")])
    reach = land.buffer(900)
    sea, tid, bch = [g.intersection(reach).difference(land) for g in (sea, tid, bch)]
    def rmask(g):
        m = Image.new('L', im.size, 0); md = ImageDraw.Draw(m)
        for p_ in polys(g):
            md.polygon([T(*q) for q in p_.exterior.coords], fill=255)
            for i_ in p_.interiors: md.polygon([T(*q) for q in i_.coords], fill=0)
        return np.asarray(m, np.float32) / 255
    ms, mt, mb = rmask(sea), rmask(tid), rmask(bch)
    wet = np.clip(ms + mt, 0, 1)
    fr = np.zeros_like(ms); fr[int(U * 1.5 * SS):int(bot_ * SS + U * 1.2 * SS), int(U * 1.5 * SS):int((W - U * 1.5) * SS)] = 1
    fr = cv2.GaussianBlur(fr, (0, 0), U * 1.5 * SS)
    shore = (wet < .5).astype(np.uint8)
    dsea = cv2.distanceTransform(1 - (wet > .5).astype(np.uint8) if False else (wet > .5).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)   # px from shore, inside the water
    lm = rmask(land); dland = cv2.distanceTransform((lm < .5).astype(np.uint8), cv2.DIST_L2, 5)
    band = U * 22 * SS
    fade = np.clip(1 - dland / band, 0, 1) ** .8 * fr
    SEA_D = np.array((92, 140, 160), np.float32); SEA_L = np.array((176, 204, 206), np.float32)
    k_ = np.clip(dsea / (U * 2.2 * SS), 0, 1)[..., None]
    seacol = SEA_L * (1 - k_) + SEA_D * k_          # shallow near the shore, deeper offshore
    a_ = np.asarray(im, np.float32)
    al = (ms * fade)[..., None] * .95; a_ = a_ * (1 - al) + seacol * al
    alt = (mt * (1 - ms) * fade)[..., None] * .9; a_ = a_ * (1 - alt) + np.array((178, 200, 196), np.float32) * alt
    alb = (mb * (1 - wet) * fade)[..., None] * .95; a_ = a_ * (1 - alb) + np.array((232, 214, 168), np.float32) * alb
    yy0 = np.arange(ms.shape[0])[:, None]; xx0 = np.arange(ms.shape[1])[None, :]
    hat = ((yy0 % int(U * .6 * SS)) < max(1, LW(.04))) & (((xx0 + (yy0 // int(U * .6 * SS)) * 37) % int(U * 2.2 * SS)) < int(U * 1.3 * SS))
    th_ = hat.astype(np.float32) * mt * (1 - ms) * fade
    a_ = a_ * (1 - th_[..., None] * .45) + np.array((90, 122, 128), np.float32) * th_[..., None] * .45
    # chart water-lines parallel to the shore, tighter near the coast
    dsea = cv2.GaussianBlur(dsea, (0, 0), U * .5 * SS); sp = U * .55 * SS; ph = np.sqrt(dsea / sp) * 2.2
    wl = (np.abs(ph - np.round(ph)) < .07).astype(np.float32) * (dsea > 1) * (ph < 9)
    wl = cv2.GaussianBlur(wl, (0, 0), .6) * (ms * fade)
    a_ = a_ * (1 - wl[..., None] * .55) + np.array((48, 88, 110), np.float32) * wl[..., None] * .55
    # crisp shoreline
    edge = cv2.morphologyEx((ms > .5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32) * fade
    a_ = a_ * (1 - edge[..., None] * .8) + np.array((52, 84, 98), np.float32) * edge[..., None] * .8
    im = Image.fromarray(np.clip(a_, 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
    print('sea px', int(ms.sum()), 'tidal', int(mt.sum()))
# SURROUNDINGS (25 Nov, Shane liked the feathered landscape on Detailed): real OSM fields, woods, roads, buildings, water from
# SURRCTX=ctx.json (ctx_fetch2.py), engraved in the same inks, feathering out from the course edge. Course size/position unchanged.
SCX = os.environ.get('SURRCTX')
if SCX and os.path.exists(SCX):
    sx_ = json.load(open(SCX)); PG = lambda k: unary_union([Polygon(list(zip(q[0::2], q[1::2]))).buffer(0) for q in sx_.get(k, []) if len(q) >= 6])
    reach = land.buffer(float(os.environ.get('SURR_M', '700')))
    lay = Image.new('RGB', im.size, PAPER); ld = ImageDraw.Draw(lay); rng_ = np.random.default_rng(12)
    def lf(g, col, out=None, w=1):
        for p_ in polys(g.intersection(reach).difference(land)):
            ld.polygon([T(*q) for q in p_.exterior.coords], fill=col, outline=out, width=w)
            for i_ in p_.interiors: ld.polygon([T(*q) for q in i_.coords], fill=PAPER)
    ld.rectangle([0, 0, im.size[0], im.size[1]], fill=(230, 216, 196) if HER else LANDF)
    fld = PG('field'); lf(fld, (234, 220, 200) if HER else (226, 224, 200), INK2, LW(.05))
    wd = unary_union([PG('wood'), PG('scrub')]); lf(wd, (214, 208, 178) if HER else (214, 216, 190))
    # engraved trees: little inked crowns scattered through the woods
    for p_ in polys(wd.intersection(reach).difference(land)):
        bx0, by0, bx1, by1 = p_.bounds; n = int(p_.area / 70); pp_ = prep(p_)
        for x, y in zip(rng_.uniform(bx0, bx1, n * 2), rng_.uniform(by0, by1, n * 2)):
            if n <= 0: break
            if not pp_.contains(Point(x, y)): continue
            n -= 1; X, Y = T(x, y); r = rng_.uniform(3, 5.5) * s * SS
            ld.ellipse([X - r + r * .25, Y - r + r * .3, X + r + r * .25, Y + r + r * .3], fill=(196, 198, 170))
            ld.ellipse([X - r, Y - r, X + r, Y + r], fill=(220, 222, 196), outline=INK2, width=LW(.04))
    wt = PG('water'); lf(wt, WATERF, WATERL, LW(.05))
    for r_ in sx_.get('road', []):
        w_ = r_[0]; ln_ = LineString(list(zip(r_[1::2], r_[2::2])))
        if ln_.distance(land) > float(os.environ.get('SURR_M', '700')): continue
        pts = [T(*q) for q in ln_.coords]
        ld.line(pts, fill=INK2, width=max(LW(.06), int((w_ + 1.2) * s * SS)), joint='curve'); ld.line(pts, fill=PAPER, width=max(1, int(w_ * s * SS)), joint='curve')
    bl = PG('bld').intersection(reach).difference(land.buffer(2))
    for p_ in polys(bl):
        q = [T(*v) for v in p_.exterior.coords]; ld.polygon(q, fill=BLD, outline=INK, width=LW(.05))
    # feather: solid at the course edge, out to nothing ~R; never under the title/strip
    lm = Image.new('L', im.size, 0); ImageDraw.Draw(lm).polygon([T(*q) for q in max(polys(land), key=lambda g: g.area).exterior.coords], fill=255)
    for p_ in polys(land): ImageDraw.Draw(lm).polygon([T(*q) for q in p_.exterior.coords], fill=255)
    dist_ = cv2.distanceTransform((np.asarray(lm) < 128).astype(np.uint8), cv2.DIST_L2, 5)
    R_ = min(W, H) * SS * float(os.environ.get('SURR_R', '.13'))
    nz_ = cv2.GaussianBlur(rng_.normal(0, 1, dist_.shape).astype(np.float32), (0, 0), R_ * .06); nz_ /= nz_.std() + 1e-6
    fa_ = np.clip(1 - (dist_ + nz_ * R_ * .08) / R_, 0, 1) ** 1.3
    yy_, xx_ = np.mgrid[0:im.size[1], 0:im.size[0]].astype(np.float32)
    edge_ = np.minimum.reduce([xx_, im.size[0] - xx_, yy_, (bot_ + U * 1.2) * SS - yy_]) / (MG * SS * .8)
    fa_ = fa_ * np.clip(edge_, 0, 1) ** 1.5
    a_ = np.asarray(im, np.float32); b_ = np.asarray(lay, np.float32)
    im = Image.fromarray(np.clip(a_ * (1 - fa_[..., None]) + b_ * fa_[..., None], 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
# land + contours (real DEM, 2 m; every 10 m bolder)
fill(land, LANDF, T)
if 'elev' not in d:   # no height data for this course: flat ground (no invented relief)
    _b = land.buffer(60).bounds; _w = int((_b[2] - _b[0]) / 8) + 2; _h = int((_b[3] - _b[1]) / 8) + 2
    d['elev'] = {'x0': _b[0], 'y1': _b[3], 'step': 8, 'w': _w, 'h': _h, 'zmin': 0, 'q': [0] * (_w * _h)}
e = d['elev']; Z = np.array(e['q'], np.float32).reshape(e['h'], e['w']) / 4 + e['zmin']
Z = cv2.GaussianBlur(Z, (0, 0), 1.2)
# hillshade wash (real DEM, light from NW) inside the land: dunes and slopes read like a hand-shaded plan
gy_, gx_ = np.gradient(Z, e['step']); hs = np.clip(.5 + (-gx_ * .7 + gy_ * .7) * (3.0 if HOLE else 1.6) / (1 + np.hypot(gx_, gy_)), 0, 1)
hs = cv2.resize(hs.astype(np.float32), (e['w'] * 4, e['h'] * 4), interpolation=cv2.INTER_CUBIC)
_Mi = np.float32([[s / (e['step'] / 4) * 0 + 1, 0, 0], [0, 1, 0]])
Mw = grid_affine(e['step'] / 4)
hsw = cv2.warpAffine(hs, Mw, im.size, flags=cv2.INTER_LINEAR, borderValue=.5)
lm = Image.new('L', im.size, 0); fill_l = ImageDraw.Draw(lm)
for p_ in polys(land): fill_l.polygon([T(*q) for q in p_.exterior.coords], fill=255)
lm = cv2.GaussianBlur(np.asarray(lm, np.float32) / 255, (0, 0), 2)
arr = np.asarray(im, np.float32); k_ = ((hsw - .5) * lm)[..., None]
_hk = .3 if HER else (.9 if LINKS else .55)
arr = np.where(k_ < 0, arr * (1 + k_ * _hk), arr + (255 - arr) * k_ * .25)
im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
lay = Image.new('L', im.size, 0); ld = ImageDraw.Draw(lay)
zr = Z.max() - Z.min(); step = 1 if zr < 15 else (2 if zr < 60 else 5)
UPS = 6; Zu = cv2.GaussianBlur(cv2.resize(Z, (e['w'] * UPS, e['h'] * UPS), interpolation=cv2.INTER_CUBIC), (0, 0), UPS * .9)
levels = np.arange(math.floor(Z.min() / step) * step + step, Z.max(), step)
allsegs = []
for lev in levels:
    m_ = (Zu >= lev).astype(np.uint8); m_[0, :] = m_[-1, :] = m_[:, 0] = m_[:, -1] = 0
    cn, _ = cv2.findContours(m_, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    allsegs.append([np.stack([e['x0'] + (q[:, 0, 0] + .5) / UPS * e['step'] - e['step'] / 2, e['y1'] - (q[:, 0, 1] + .5) / UPS * e['step'] + e['step'] / 2], 1) for q in cn if len(q) > 8])
def chaikin(P, n):
    P = np.asarray(P, np.float64)
    for _ in range(n):
        if len(P) < 4: break
        Q = .75 * P[:-1] + .25 * P[1:]; R = .25 * P[:-1] + .75 * P[1:]; P = np.empty((len(Q) * 2, 2)); P[0::2] = Q; P[1::2] = R
    return P
class _CS: pass
cs = _CS(); cs.levels = levels; cs.allsegs = allsegs
lp = prep(land)
for lev, segs in zip(cs.levels, cs.allsegs):
    for sg in segs:
        if len(sg) < 4: continue
        ls = LineString(chaikin(sg[::2], 3)).intersection(land)
        for g_ in getattr(ls, 'geoms', [ls]):
            if g_.geom_type != 'LineString' or g_.length < 8: continue
            ld.line([T(*q) for q in g_.coords], fill=150 if round(lev) % (step * 5) == 0 else 70, width=LW(.08) if round(lev) % (step * 5) == 0 else LW(.05), joint='curve')
lay = np.asarray(lay, np.float32)[..., None] / 255
_ck = .35 if HER else .8
arr = np.asarray(im, np.float32); arr = arr * (1 - lay * _ck) + np.array((132, 118, 92), np.float32) * lay * _ck
im = Image.fromarray(arr.astype(np.uint8)); dr = ImageDraw.Draw(im)
def draw_all(T_, sx, clipg=None, small=False):
    C = (lambda g: g.intersection(clipg)) if clipg is not None else (lambda g: g)
    wf, ff, gg, tt, bb, ww, bl = [C(g) for g in (woods, fair, green, tee, bunk, water, bld)]
    fill(wf, TREEF, T_); line(wf, (70, 88, 66), LW(.04 if small else .055), T_)
    fill(ff, FAIR, T_); line(ff, INK, LW(.05 if small else .08), T_)
    fill(tt, FAIR, T_); line(tt, INK, LW(.05 if small else .07), T_)
    fill(gg, GRN, T_); line(gg, INK, LW(.06 if small else .09), T_)
    fill(ww, WATERF, T_); line(ww, WATERL, LW(.06 if small else .09), T_)
    if not small: waterlines(ww, T_, sx, LW(.04))
    fill(bb, SANDF, T_); stipple(bb, T_, sx, dens=(0 if small else .35) / max(sx, .3) ** 1.0 * .5); line(bb, SANDD, LW(.05 if small else .07), T_)
    if not small: fill(bl, BLD, T_); line(bl, INK2, LW(.05), T_)
def mask_of(g, T_):
    m = Image.new('L', im.size, 0); md = ImageDraw.Draw(m)
    for p_ in polys(g):
        md.polygon([T_(*q) for q in p_.exterior.coords], fill=255)
        for i_ in p_.interiors: md.polygon([T_(*q) for q in i_.coords], fill=0)
    return np.asarray(m, np.float32) / 255
# hachures: engraved slope strokes from the real DEM (dunes, banks, hollows), drawn downhill, heavier where steeper, darker on slopes facing away from the NW light
_Zs = cv2.GaussianBlur(cv2.resize(Z, (e['w'] * 4, e['h'] * 4), interpolation=cv2.INTER_CUBIC), (0, 0), 2.5); _st = e['step'] / 4
_gy, _gx = np.gradient(_Zs, _st); _gy = -_gy   # array rows run north->south
sp = U * (.42 if LINKS else .55) / s; rng_h = np.random.default_rng(5)
playb = prep(playg.buffer(1.5)); landp = prep(land.buffer(-1.5))
if LINKS:   # dune tone: high ground (ridges of marram) darker, hollows lighter, from the real DEM relief
    _rel = .6 * (_Zs - cv2.GaussianBlur(_Zs, (0, 0), 40 / _st)) + .8 * (_Zs - cv2.GaussianBlur(_Zs, (0, 0), 14 / _st))
    _sl = np.hypot(_gx, _gy)
    _w = np.clip(_rel / 2.2, 0, 1) * .75 + np.clip(_sl / .2, 0, 1) * .55
    _w = cv2.GaussianBlur(np.clip(_w, 0, 1).astype(np.float32), (0, 0), .8)
    ww_ = cv2.warpAffine(_w, grid_affine(_st), im.size, flags=cv2.INTER_LINEAR, borderValue=0)
    pm = mask_of(playg, T); lmk = mask_of(land, T)
    ww_ = (ww_ * lmk * (1 - cv2.GaussianBlur(pm, (0, 0), 3)))[..., None]
    a_ = np.asarray(im, np.float32); _dk = .72 if HOLE else .5; a_ = a_ * (1 - ww_ * _dk) + np.array((100, 118, 72), np.float32) * ww_ * _dk
    im = Image.fromarray(np.clip(a_, 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
hl_ = Image.new('L', im.size, 0); hd = ImageDraw.Draw(hl_)
bx0, by0, bx1, by1 = land.bounds
for gyv in np.arange(by0, by1, sp):
    for gxv in np.arange(bx0, bx1, sp):
        x = gxv + rng_h.uniform(-.35, .35) * sp; y = gyv + rng_h.uniform(-.35, .35) * sp
        i_ = int((e['y1'] + e['step'] / 2 - y) / _st); j_ = int((x - e['x0'] + e['step'] / 2) / _st)
        if not (0 <= i_ < _Zs.shape[0] and 0 <= j_ < _Zs.shape[1]): continue
        gx_, gy_ = _gx[i_, j_], _gy[i_, j_]; sl = math.hypot(gx_, gy_)
        if sl < (.035 if LINKS else .05): continue
        pt = Point(x, y)
        if not landp.contains(pt) or playb.contains(pt): continue
        ux, uy = -gx_ / sl, -gy_ / sl; Lh = sp * min(1.0, .45 + sl * 2.2)
        shade = .5 + .5 * max(-1, min(1, (ux * .7 - uy * .7)))   # downhill toward SE -> slope faces SE (lit side is NW-facing)
        val = int(min(255, 60 + 900 * sl * (.55 + .6 * shade)))
        hd.line([T(x, y), T(x + ux * Lh, y + uy * Lh)], fill=val, width=max(1, LW(.03 + min(.05, sl * .12))))
hl_ = np.asarray(hl_, np.float32)[..., None] / 255
a_ = np.asarray(im, np.float32); _hs = (.3 if HER else (.9 if LINKS else .55)); a_ = a_ * (1 - hl_ * _hs) + np.array((54, 70, 44) if LINKS else (84, 80, 56), np.float32) * hl_ * _hs
im = Image.fromarray(np.clip(a_, 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
# ---------- main map, v2 (24 Nov, Shane: "go as far as I can ... same vibe")
paths = unary_union([p_ for p_ in FE.get('path', []) if p_.intersects(near)]).intersection(land.buffer(-2)).difference(unary_union([g for g in (green, bunk, water) if not g.is_empty]))
fill(paths, (226, 219, 198), T); line(paths, (176, 166, 140), LW(.035), T)
if LINKS:   # marram tufts: denser, taller and darker on the real dune ridges/slopes (DEM), sparse in the flat rough
    rng_ = np.random.default_rng(11); rough = prep(land.buffer(-3).difference(playg.buffer(3)))
    bx0, by0, bx1, by1 = land.bounds; n_ = int(land.area / 14)
    for x, y in zip(rng_.uniform(bx0, bx1, n_), rng_.uniform(by0, by1, n_)):
        i_ = int((e['y1'] + e['step'] / 2 - y) / _st); j_ = int((x - e['x0'] + e['step'] / 2) / _st)
        wv = float(_w[i_, j_]) if (0 <= i_ < _w.shape[0] and 0 <= j_ < _w.shape[1]) else 0
        if rng_.random() > .12 + .88 * wv ** 1.2: continue
        if not rough.contains(Point(x, y)): continue
        X, Y = T(x, y); h_ = U * (.26 + .34 * wv) * SS * rng_.uniform(.8, 1.2)
        col = tuple(int(a * (1 - wv) + b * wv) for a, b in zip((122, 132, 86), (58, 76, 46)))
        for dx in (-.6, -.2, .2, .6): dr.line([(X + dx * h_ * .35, Y), (X + dx * h_, Y - h_ * (1 - abs(dx) * .4))], fill=col, width=LW(.035 + .02 * wv))
# scrub / woodland areas from the trace that have no individual crowns (e.g. sea buckthorn and gorse on Bull Island): scalloped clumps
_sc = unary_union([p_ for k_ in ('wood', 'heath', 'scrub') for p_ in FE.get(k_, [])]).intersection(land.buffer(-1.5)).difference(playg.buffer(1))
if crowns and not _sc.is_empty: _sc = _sc.difference(unary_union([Point(x, y).buffer(r, 6) for x, y, r in crowns]))
_sc = unary_union([q for q in polys(_sc) if q.area > 20])
if not _sc.is_empty:
    rng_s = np.random.default_rng(9); bx0, by0, bx1, by1 = _sc.bounds; scp = prep(_sc)
    n_ = int(_sc.area / (16 if HER else 6))
    pts_ = [(x, y) for x, y in zip(rng_s.uniform(bx0, bx1, n_ * 3), rng_s.uniform(by0, by1, n_ * 3)) if scp.contains(Point(x, y))][:n_]
    for x, y in sorted(pts_, key=lambda t: -t[1]):
        X, Y = T(x, y); R_ = max(1.2 * SS, rng_s.uniform(2.4, 3.8) * s * SS) if HER else max(1.2 * SS, rng_s.uniform(1.3, 2.2) * s * SS)
        k_ = rng_s.integers(3); col = ([(66, 84, 40), (80, 98, 44), (58, 74, 36)] if HER else [(98, 112, 74), (110, 122, 80), (88, 102, 68)])[k_]
        dr.ellipse([X - R_, Y - R_, X + R_, Y + R_], fill=col, outline=(38, 48, 26) if HER else (62, 74, 50), width=max(1, LW(.025)))
        if HER: hr = R_ * .5; dr.ellipse([X - R_ * .35 - hr, Y - R_ * .35 - hr, X - R_ * .35 + hr, Y - R_ * .35 + hr], fill=tuple(int(a * .6 + b * .4) for a, b in zip((126, 142, 64), col)))
# gorse / scrub: dark spiky clumps (real 'gorse' crowns from the trace)
_gp = prep(land.buffer(-2)); _pl = prep(playg)
for x, y, r in gorse:
    if not _gp.contains(Point(x, y)) or _pl.contains(Point(x, y)): continue
    X, Y = T(x, y); R_ = max(1.2 * SS, r * .5 * s * SS)
    dr.ellipse([X - R_, Y - R_, X + R_, Y + R_], fill=(96, 104, 70), outline=(62, 70, 46), width=max(1, LW(.03)))
    for k in range(7):
        a = k * 2 * math.pi / 7; dr.line([(X, Y), (X + math.cos(a) * R_ * 1.25, Y + math.sin(a) * R_ * 1.25)], fill=(62, 70, 46), width=max(1, LW(.025)))
# trees: soft cast shadow (light from NW), then each real crown shaded with a lit side, three greens for variety
TG = [(112, 132, 96), (124, 144, 104), (100, 122, 90)]; TL = (168, 182, 136); TO = (64, 82, 60)
if HER: TG = [(56, 70, 30), (66, 79, 36), (48, 62, 26)]; TL = (100, 112, 56); TO = (26, 38, 16)   # 27 Nov: dark greens sampled from Shane's reference trees (shadow 32,46,19 / mid 66,78,37 / lit 92,104,56)
sh = Image.new('L', im.size, 0); sd = ImageDraw.Draw(sh)
CK = .74 if HER else .62
for x, y, r in crowns:
    X, Y = T(x, y); R_ = max(1.5 * SS, r * CK * s * SS); o_ = R_ * .45
    sd.ellipse([X - R_ + o_, Y - R_ + o_, X + R_ + o_, Y + R_ + o_], fill=255)
sh = cv2.GaussianBlur(np.asarray(sh, np.float32) / 255, (0, 0), max(1, U * .12 * SS))[..., None]
a_ = np.asarray(im, np.float32); a_ = a_ * (1 - sh * (.42 if HER else .22)); im = Image.fromarray(a_.astype(np.uint8)); dr = ImageDraw.Draw(im)
for x, y, r in sorted(crowns, key=lambda t: -t[1]):
    X, Y = T(x, y); R_ = max(1.5 * SS, r * CK * s * SS); k_ = (int(x * 7.3) ^ int(y * 3.1)) % 3
    dr.ellipse([X - R_, Y - R_, X + R_, Y + R_], fill=TG[k_], outline=TO, width=max(1, LW(.035)))
    hr = R_ * .55; dr.ellipse([X - R_ * .38 - hr, Y - R_ * .38 - hr, X - R_ * .38 + hr, Y - R_ * .38 + hr], fill=tuple(int(a * .55 + b * .45) for a, b in zip(TL, TG[k_])) if not HER else tuple(int(a * .45 + b * .55) for a, b in zip(TL, TG[k_])))
    if HER: hr2 = R_ * .22; dr.ellipse([X - R_ * .45 - hr2, Y - R_ * .45 - hr2, X - R_ * .45 + hr2, Y - R_ * .45 + hr2], fill=(112, 124, 62) if HER else TL)
# first cut: a slightly lighter band of semi-rough hugging every fairway (real on every course)
_fc = unary_union([fair, green]).buffer(3.5 if HER else 2.5, 8).difference(unary_union([fair, green, bunk, water]))
if HER:   # first cut: a darker green that melts into the rough (no hard band)
    _fm2 = cv2.GaussianBlur(mask_of(unary_union([fair, green]).buffer(4.5, 8).intersection(land), T), (0, 0), max(1, U * .45 * SS))[..., None]
    a_ = np.asarray(im, np.float32); a_ = a_ * (1 - _fm2 * .8) + np.array((90, 104, 44), np.float32) * _fm2 * .8; im = Image.fromarray(np.clip(a_, 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
else: fill(_fc.intersection(land), tuple(min(255, int(v * 1.0 + 18)) for v in LANDF), T)
# fairways with mowing stripes along each hole
fill(fair, FAIR, T); fill(tee, FAIR, T)
fm = mask_of(unary_union([fair, tee]), T); stripe = np.zeros(fm.shape, np.float32)
yy_, xx_ = np.mgrid[0:fm.shape[0], 0:fm.shape[1]].astype(np.float32)
for h in HL:
    P0 = np.array(T(*h['pts'][0])); P1 = np.array(T(*h['pts'][-1])); v = P1 - P0; v /= (np.linalg.norm(v) + 1e-6); nrm = np.array([-v[1], v[0]])
    zm = mask_of(LineString(h['pts']).buffer(45, 8), T) * fm
    band = np.floor(((xx_ - P0[0]) * nrm[0] + (yy_ - P0[1]) * nrm[1]) / (7 * s * SS)) % 2
    stripe = np.maximum(stripe, zm * band)
if HER:   # turf with no hole of its own (practice ground): stripes running north-south
    _cov = np.zeros(fm.shape, np.float32)
    for h in HL: _cov = np.maximum(_cov, mask_of(LineString(h['pts']).buffer(45, 8), T))
    stripe = np.maximum(stripe, fm * (1 - _cov) * (np.floor(xx_ / (7 * s * SS)) % 2))
a_ = np.asarray(im, np.float32); a_ = a_ + (255 - a_) * (stripe * (.13 if HER else .16))[..., None]; im = Image.fromarray(np.clip(a_, 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
if HER:   # soft bevel: turf darkens toward its edge and lifts in the middle, so fairways read as raised, mown ground
    _bl = cv2.GaussianBlur(fm, (0, 0), max(1, U * .55 * SS)); _ed = np.clip(fm - _bl, 0, 1) * 2.2
    _sh2 = np.clip(np.roll(np.roll(fm, int(U * .25 * SS), 0), int(U * .25 * SS), 1) - fm, 0, 1)   # thin drop shadow to the SE
    _sh2 = cv2.GaussianBlur(_sh2, (0, 0), max(1, U * .2 * SS))
    _ce = np.clip(1 - _ed, 0, 1) * cv2.GaussianBlur(fm, (0, 0), max(1, U * 1.2 * SS))   # lighter, yellower centre line of the mown turf
    a_ = np.asarray(im, np.float32); a_ = a_ * (1 - np.clip(_ed, 0, 1)[..., None] * .2) * (1 - _sh2[..., None] * .35)
    a_ = a_ + (np.array((156, 168, 84), np.float32) - a_) * (_ce[..., None] * .26)
    im = Image.fromarray(np.clip(a_, 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
if HER: line(fair, (78, 94, 38), LW(.05), T); line(tee, (78, 94, 38), LW(.05), T)
else: line(fair, INK, LW(.08), T); line(tee, INK, LW(.07), T)
# greens: fill, a lighter collar ring inside, crisp edge
fill(green, GRN, T)
gi = green.buffer(-1.2, 8)
fill(gi, (138, 156, 66) if HER else (182, 206, 140), T); line(green, (92, 110, 44) if HER else INK, LW(.05 if HER else .09), T)
# water: fill, waterlines, darker shore edge
fill(water, WATERF, T)
if not HER: waterlines(water, T, s, LW(.04))
line(water, WATERL, LW(.06 if HER else .1), T)
if os.environ.get('TREEFIX') and _tf.get('islands'):   # the lake fill paints its island holes with ground colour: put the island trees back on top
    _ip = prep(unary_union([Polygon(q) for q in _tf['islands']]).buffer(2))
    for x, y, r in sorted([q for q in crowns if _ip.contains(Point(q[0], q[1]))], key=lambda t: -t[1]):
        X, Y = T(x, y); R_ = max(1.5 * SS, r * CK * s * SS); k_ = (int(x * 7.3) ^ int(y * 3.1)) % 3; o_ = R_ * .3
        dr.ellipse([X - R_ + o_, Y - R_ + o_, X + R_ + o_, Y + R_ + o_], fill=tuple(int(v * .62) for v in LANDF))
        dr.ellipse([X - R_, Y - R_, X + R_, Y + R_], fill=TG[k_], outline=TO, width=max(1, LW(.035)))
        hr = R_ * .55; dr.ellipse([X - R_ * .38 - hr, Y - R_ * .38 - hr, X - R_ * .38 + hr, Y - R_ * .38 + hr], fill=tuple(int(a * .45 + b * .55) for a, b in zip(TL, TG[k_])) if HER else tuple(int(a * .55 + b * .45) for a, b in zip(TL, TG[k_])))
# bunkers: sand with fine stipple and a shaded lip on the far (NW-lit) side for depth
bunk = bunk.buffer(.35, 6)   # hairline grow so small pot bunkers still read at print size (shape unchanged)
fill(bunk, (234, 222, 196) if HER else (251, 248, 238), T); stipple(bunk, T, s, dens=.25 / max(s, .3) * .5, col=(196, 178, 140))
bm = mask_of(bunk, T); sh_ = max(1, int(U * .14 * SS))
lip = np.clip(bm - np.roll(np.roll(bm, sh_, 0), sh_, 1), 0, 1); lip = cv2.GaussianBlur(lip, (0, 0), sh_ * .6)[..., None] * bm[..., None]
a_ = np.asarray(im, np.float32); a_ = a_ * (1 - lip * .42); im = Image.fromarray(a_.astype(np.uint8)); dr = ImageDraw.Draw(im)
line(bunk, (176, 156, 116) if HER else (92, 80, 58), LW(.05 if HER else .08), T)
fill(bld, BLD, T); line(bld, INK2, LW(.05), T)
# LIDAR (24 Nov): real 1-2 m laser relief (GSI/OPW open data, CC BY 4.0) from LIDAR=hillshade.png + .json bbox in local metres.
# High-pass relief only: every dune ridge, hollow, bunker bowl and mound on the course, engraved in ink; light on the greens, fuller in the rough.
LIDAR = os.environ.get('LIDAR'); LHP = None
if LIDAR and os.path.exists(LIDAR):
    lm_ = json.load(open(LIDAR.rsplit('.', 1)[0] + '.json')); hs = np.asarray(Image.open(LIDAR).convert('L'), np.float32)
    hp = hs - cv2.GaussianBlur(hs, (0, 0), 10)
    px = lambda j, i: T(lm_['x0'] + j * (lm_['x1'] - lm_['x0']) / lm_['w'], lm_['y1'] - i * (lm_['y1'] - lm_['y0']) / lm_['h'])
    M_ = cv2.getAffineTransform(np.float32([[0, 0], [1000, 0], [0, 1000]]), np.float32([px(0, 0), px(1000, 0), px(0, 1000)]))
    hpw = cv2.warpAffine(hp, M_, im.size, flags=cv2.INTER_LINEAR, borderValue=0); LHP = (hp, lm_)
    lmk = mask_of(land.buffer(-1), T); pm_ = cv2.GaussianBlur(mask_of(playg, T), (0, 0), 2); gm_ = mask_of(green.buffer(1), T)
    k_ = lmk * (1 - pm_ * .6) * (1 - gm_ * .6) * (1 - mask_of(bld.buffer(2), T))
    if crowns:   # the LiDAR is a surface model: tree canopies would print as doubled blobs, so relief only on open ground
        k_ = k_ * (1 - cv2.GaussianBlur(mask_of(unary_union([Point(x, y).buffer(r * .62 + 6, 8) for x, y, r in crowns]), T), (0, 0), 3))
    dk = np.clip(-hpw / 40, 0, 1) ** 1.3 * k_; lt = np.clip(hpw / 40, 0, 1) * k_
    a_ = np.asarray(im, np.float32)
    _d1, _l1 = (.16, .08) if HER else (.55, .35)
    a_ = a_ * (1 - dk[..., None] * _d1) + np.array((60, 74, 50), np.float32) * dk[..., None] * _d1
    a_ = a_ * (1 - lt[..., None] * _l1) + np.array((252, 248, 228), np.float32) * lt[..., None] * _l1
    im = Image.fromarray(np.clip(a_, 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
if HER:   # painted finish (27 Nov, Shane: 'more of the painted look'): wobble every edge a little like a brush, smooth into flat washes, add paper/brush texture
    _a = np.asarray(im); _a0 = _a.astype(np.float32)
    _h, _w = _a.shape[:2]; _rg = np.random.default_rng(21)
    def _nz(sig): n = cv2.GaussianBlur(_rg.normal(0, 1, (_h // 4, _w // 4)).astype(np.float32), (0, 0), sig / 4); n = cv2.resize(n / (n.std() + 1e-6), (_w, _h)); return n
    _amp = U * .22 * SS; _dx = (_nz(U * 1.2 * SS) * .7 + _nz(U * .35 * SS) * .3) * _amp; _dy = (_nz(U * 1.2 * SS) * .7 + _nz(U * .35 * SS) * .3) * _amp
    _yy, _xx = np.mgrid[0:_h, 0:_w].astype(np.float32)
    _a = cv2.remap(_a, _xx + _dx, _yy + _dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    _small = cv2.resize(_a, (_w // 2, _h // 2), interpolation=cv2.INTER_AREA)
    _small = cv2.edgePreservingFilter(_small, flags=1, sigma_s=5, sigma_r=.12)
    _p = cv2.resize(_small, (_w, _h), interpolation=cv2.INTER_CUBIC).astype(np.float32) * .55 + _a.astype(np.float32) * .45
    _wash = (_nz(U * 3 * SS) * .035 + _nz(U * .6 * SS) * .025 + cv2.GaussianBlur(_rg.normal(0, 1, (_h, _w)).astype(np.float32), (0, 0), 1.2) * .02)
    _lmk = cv2.GaussianBlur(mask_of(land, T), (0, 0), max(1, U * .3 * SS))
    _p = _p * (1 + (_wash * _lmk)[..., None])
    _p = np.where(_lmk[..., None] > .01, _p, _a0)   # paper/surroundings untouched by the wobble wash
    # water with depth: deeper teal in the middle, lighter toward the banks, soft ripple light (27 Nov, Shane: 'water is all a flat blue')
    _wm = mask_of(water, T); _dw = cv2.distanceTransform((_wm > .5).astype(np.uint8), cv2.DIST_L2, 5)
    _dp = np.clip(_dw / (U * 2.2 * SS), 0, 1) ** .8
    _wc = np.array((112, 152, 152), np.float32) * (1 - _dp[..., None]) + np.array((44, 86, 100), np.float32) * _dp[..., None]
    _rp = (_nz(U * .5 * SS) * .5 + _nz(U * .18 * SS) * .5); _wc = _wc * (1 + _rp[..., None] * .05)
    _wmb = cv2.GaussianBlur(_wm, (0, 0), max(1, U * .08 * SS))[..., None]
    _p = _p * (1 - _wmb * .9) + _wc * _wmb * .9
    _rim = np.clip(_wm - cv2.GaussianBlur(_wm, (0, 0), max(1, U * .25 * SS)), 0, 1)[..., None] * 1.6
    _p = _p * (1 - np.clip(_rim, 0, 1) * .35)
    # overall contrast on the course only: gentle S-curve + a touch more saturation
    _x = _p / 255; _c = _x + (_x - _x * _x) * (_x - .5) * 1.1 * 2 * .55; _g = _c.mean(-1, keepdims=True); _c = _g + (_c - _g) * 1.12
    _p = _p * (1 - _lmk[..., None]) + np.clip(_c, 0, 1) * 255 * _lmk[..., None]
    im = Image.fromarray(np.clip(_p, 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
# flags on the real pin positions (inside a green)
gp = prep(green.buffer(2))
NOPLAY = (HOLE and os.environ.get('NOPLAY') == '1') or PV or os.environ.get('NOPLAY') == '2'   # shop single-hole print: the designer/robot draw the customer's own shots + flag on top
for px_, py_ in ([] if NOPLAY else d.get('pins', [])):
    if not gp.contains(Point(px_, py_)): continue
    X, Y = T(px_, py_); hpole = U * 1.0 * SS
    dr.ellipse([X - U * .12 * SS, Y - U * .06 * SS, X + U * .12 * SS, Y + U * .06 * SS], fill=(60, 70, 60))
    dr.line([(X, Y), (X, Y - hpole)], fill=INK, width=LW(.06))
    dr.polygon([(X, Y - hpole), (X + U * .55 * SS, Y - hpole + U * .2 * SS), (X, Y - hpole + U * .4 * SS)], fill=RED)
if os.environ.get('GEODUMP'):   # 27 Nov: course geometry in poster px, for styles drawn from the vectors (estate.py)
    _P = lambda g: [[[(T(*q)[0] / SS, T(*q)[1] / SS) for q in p_.exterior.coords]] + [[(T(*q)[0] / SS, T(*q)[1] / SS) for q in i_.coords] for i_ in p_.interiors] for p_ in polys(g)]
    json.dump({'W': W, 'H': H, 'px_m': s, **{k: _P(g) for k, g in (('land', land), ('fair', fair), ('green', green), ('tee', tee), ('bunk', bunk), ('water', water), ('woods', woods), ('scrub', _sc))},
               'crowns': [(T(x, y)[0] / SS, T(x, y)[1] / SS, r * CK * s) for x, y, r in crowns]}, open(os.environ['GEODUMP'], 'w'))
line(land, (44, 58, 30) if HER else INK, LW(.1 if HER else .14), T)
# line of play + numbered tee circles (red ink, like an architect's routing)
fn = F('corm500', U * (1.75 if HER else 1.35) * SS); rr = U * (1.45 if HER else 1.05) * SS
_mk = {}   # tee marker positions; push apart any that overlap (K Club 11 + 15 tees are 13 m apart, one hid the other)
for h in ([] if NOPLAY else HL):
    (ax, ay), (bx, by) = h['pts'][0], h['pts'][1]; Lm = math.hypot(bx - ax, by - ay) or 1
    _mk[id(h)] = list(T(ax - (bx - ax) / Lm * 16, ay - (by - ay) / Lm * 16))
for _ in range(60):
    _mv = False; _ks = list(_mk)
    for i in range(len(_ks)):
        for j in range(i + 1, len(_ks)):
            p1, p2 = _mk[_ks[i]], _mk[_ks[j]]; dx, dy = p2[0] - p1[0], p2[1] - p1[1]; dd = math.hypot(dx, dy)
            if dd < rr * 2.25:
                if dd < 1e-3: dx, dy, dd = 1.0, 0.0, 1.0
                push = (rr * 2.25 - dd) / 2 + .5; ux, uy = dx / dd, dy / dd
                p1[0] -= ux * push; p1[1] -= uy * push; p2[0] += ux * push; p2[1] += uy * push; _mv = True
    if not _mv: break
for h in ([] if NOPLAY else HL):
    pts = [T(*p) for p in h['pts']]
    L = LineString(pts)
    for i in ([] if HER else range(0, int(L.length), int(U * 1.4 * SS))):   # dashed (none on heritage)
        a = L.interpolate(i); b = L.interpolate(min(L.length, i + U * .75 * SS)); dr.line([a.coords[0], b.coords[0]], fill=RED, width=LW(.07))
    (ax, ay), (bx, by) = h['pts'][0], h['pts'][1]; Lm = math.hypot(bx - ax, by - ay) or 1
    px, py = _mk[id(h)]
    if HER:
        dr.ellipse([px - rr + U * .12 * SS, py - rr + U * .16 * SS, px + rr + U * .12 * SS, py + rr + U * .16 * SS], fill=(40, 54, 40))
        dr.ellipse([px - rr, py - rr, px + rr, py + rr], fill=(242, 232, 214), outline=INK, width=LW(.07)); dr.ellipse([px - rr * .82, py - rr * .82, px + rr * .82, py + rr * .82], outline=INK2, width=LW(.03))
        dr.text((px, py + rr * .08), str(h['ref']), font=fn, fill=INK, anchor='mm')
    else:
        dr.ellipse([px - rr, py - rr, px + rr, py + rr], fill=PAPER, outline=RED, width=LW(.08))
        dr.text((px, py + rr * .08), str(h['ref']), font=fn, fill=RED, anchor='mm')
# ---- 18-hole strip: every hole on its own, tee at the bottom, same scale for all
order = [] if (HOLE or PV) else sorted(HL, key=lambda h: int(re.sub(r'\D', '', str(h['ref'])) or 0))
n = len(order); rows = 1 if land_o else 2; per = max(1, math.ceil(n / rows))
cellw = (W - 2 * MG) / per; cellh = (STRIP[1] - STRIP[0]) / rows
segs = []
for h in order:
    p = h['pts']; ang = math.degrees(math.atan2(p[-1][1] - p[0][1], p[-1][0] - p[0][0]))
    zone = LineString(p).buffer(float(os.environ.get('STRIP_Z', '30')), 16)
    rot = lambda g, a_=90 - ang, o_=Point(p[0]): affinity.rotate(g, a_, origin=o_)
    segs.append((h, zone, rot, ang))
hmax = max(rot(zone).bounds[3] - rot(zone).bounds[1] for h, zone, rot, ang in segs) if segs else 1; wmax = max(rot(zone).bounds[2] - rot(zone).bounds[0] for h, zone, rot, ang in segs) if segs else 1
ssx = min((cellh - U * 3.4) / hmax, (cellw * .92) / wmax)
fs = F('jost400', U * 1.0 * SS)
_strip = []
for i, (h, zone, rot, ang) in enumerate(segs):
    r_, k_ = divmod(i, per); ccx = MG + cellw * (k_ + .5); top = STRIP[0] + r_ * cellh
    bz = rot(zone).bounds; zc = ((bz[0] + bz[2]) / 2, (bz[1] + bz[3]) / 2)
    T2 = mk(ssx, ccx, top + (cellh - U * 2.6) / 2, zc[0], zc[1])
    R = lambda g, rot=rot, zone=zone: rot(g.intersection(zone))
    global_saved = (woods, fair, green, tee, bunk, water, bld)
    rz = rot(zone)
    fill(rz.intersection(rot(land)), LANDF, T2)
    for g, col, oc in ((R(woods), TREEF if HER else (118, 138, 100), TO if HER else (64, 82, 60)), (R(fair), FAIR, INK), (R(tee), FAIR, INK), (R(green), GRN, INK), (R(water), WATERF, WATERL), (R(bunk.buffer(.35, 6)), (251, 248, 238), (92, 80, 58))):
        fill(g, col, T2); line(g, oc, LW(.04), T2)
    if LHP is not None:   # same real LiDAR relief in each hole of the strip
        hp_, lm2 = LHP
        def P2(j, i, rot=rot, T2=T2):
            q = rot(Point(lm2['x0'] + j * (lm2['x1'] - lm2['x0']) / lm2['w'], lm2['y1'] - i * (lm2['y1'] - lm2['y0']) / lm2['h'])); return T2(q.x, q.y)
        M2 = cv2.getAffineTransform(np.float32([[0, 0], [1000, 0], [0, 1000]]), np.float32([P2(0, 0), P2(1000, 0), P2(0, 1000)]))
        hw2 = cv2.warpAffine(hp_, M2, im.size, flags=cv2.INTER_LINEAR, borderValue=0)
        k2 = mask_of(rz.intersection(rot(land)), T2) * (1 - mask_of(R(green), T2) * .6)
        d2 = np.clip(-hw2 / 40, 0, 1) ** 1.3 * k2; l2 = np.clip(hw2 / 40, 0, 1) * k2
        a_ = np.asarray(im, np.float32)
        a_ = a_ * (1 - d2[..., None] * .5) + np.array((60, 74, 50), np.float32) * d2[..., None] * .5
        a_ = a_ * (1 - l2[..., None] * .3) + np.array((252, 248, 228), np.float32) * l2[..., None] * .3
        im = Image.fromarray(np.clip(a_, 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
        for g, oc in ((R(fair), INK), (R(green), INK), (R(bunk), (92, 80, 58))): line(g, oc, LW(.04), T2)
    if os.environ.get('GEODUMP'):
        _P2 = lambda g, T2=T2: [[[(T2(*q)[0] / SS, T2(*q)[1] / SS) for q in p_.exterior.coords]] + [[(T2(*q)[0] / SS, T2(*q)[1] / SS) for q in i_.coords] for i_ in p_.interiors] for p_ in polys(g)]
        _strip.append({'land': _P2(rz.intersection(rot(land))), 'fair': _P2(unary_union([R(fair), R(tee)])), 'green': _P2(R(green)), 'bunk': _P2(R(bunk.buffer(.35, 6))), 'water': _P2(R(water)), 'woods': _P2(R(woods))})
    rp = rot(LineString(h['pts']))
    if not HER: dr.line([T2(*q) for q in rp.coords], fill=RED, width=LW(.05))
    dr.text((ccx * SS, (top + cellh - U * .9) * SS), str(h['ref']), font=fs, fill=INK, anchor='ms')
# scale bar (real metres) + small north arrow, bottom right of the map area
_bx1 = W - MG; _by = bot_ - U * .5
for L_m in (500, 250, 200, 100):
    if L_m * s < W * .14: break
fsb = F('jost400', U * .95 * SS); x1_ = _bx1 * SS; x0_ = x1_ - L_m * s * SS; yb = _by * SS
for k_ in range(4):
    xa = x0_ + (x1_ - x0_) * k_ / 4; xb = x0_ + (x1_ - x0_) * (k_ + 1) / 4
    dr.rectangle([xa, yb - U * .28 * SS, xb, yb], fill=INK if k_ % 2 == 0 else PAPER, outline=INK, width=LW(.05))
dr.text((x0_, yb - U * .7 * SS), '0', font=fsb, fill=INK2, anchor='ms'); dr.text((x1_, yb - U * .7 * SS), f'{L_m} m', font=fsb, fill=INK2, anchor='ms')
nx, ny = x1_ - U * 1.2 * SS, yb - U * 4.6 * SS; nh = U * 1.6 * SS
_rt = lambda px, py: (nx + px * math.cos(math.radians(-ROT)) - py * math.sin(math.radians(-ROT)), ny + px * math.sin(math.radians(-ROT)) + py * math.cos(math.radians(-ROT)))
dr.polygon([_rt(0, -nh), _rt(nh * .32, nh * .35), _rt(0, nh * .12)], fill=INK); dr.polygon([_rt(0, -nh), _rt(-nh * .32, nh * .35), _rt(0, nh * .12)], outline=INK, fill=PAPER, width=LW(.05))
_tx, _ty = _rt(0, -nh - U * 1.1 * SS); dr.text((_tx, _ty), 'N', font=F('corm500', U * 1.3 * SS), fill=INK, anchor='mm')
im = im.resize((W, H), Image.LANCZOS)
# paper grain
a = np.asarray(im, np.float32); rng = np.random.default_rng(3)
g = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.0) * (1.6 if HER else 2.4) + cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 7) * (0 if HER else 1.8)
im = Image.fromarray(np.clip(a + g[..., None], 0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
# hairline under the strip label area + type block
if not (HOLE or PV): dr.line([(MG, STRIP[0] - U * 1.4), (W - MG, STRIP[0] - U * 1.4)], fill=INK2, width=max(1, int(U * .06)))
name = c['name']; m = re.search(r'\s+(Golf (Club|Course|Links|Resort|Centre)|Golf (&|and) \w+ Club)', name)
title = (name[:m.start()] if m else name.split(' - ')[0].split(',')[0]).replace(' (Old Course)', '').strip()
if c.get('title'): title = c['title']
_crs = '' if c.get('title') or ',' not in name else name.split(',', 1)[1].strip()   # 27 Nov: multi-course clubs name the course under the title (Killeen Course, Old Links...)
def spw(t, f, tr): return sum(f.getlength(ch) for ch in t) + tr * (len(t) - 1)
def spd(cx_, y, t, f, fill_, tr):
    x = cx_ - spw(t, f, tr) / 2
    for ch in t: dr.text((x, y), ch, font=f, fill=fill_, anchor='ls'); x += f.getlength(ch) + tr
ty = H * ((.888 if HER else .905) if land_o else (.9 if HER else .91)); ft = F('corm500', U * (4.6 if HER else 4.0)); tr = U * 1.2
while spw(title.upper(), ft, tr) > W * .62: ft = F('corm500', ft.size * .95); tr *= .95
if HOLE:
    sub0 = title.upper() + (f'  ·  {_crs.upper()}' if _crs else ''); title = f'HOLE {HOLE}'
if HER:
    for k_, w_ in ((.42, .09), (.52, .04)): dr.rectangle([MG * k_, MG * k_, W - MG * k_, H - MG * k_], outline=INK2 if k_ > .5 else INK, width=max(1, int(U * w_)))
spd(W / 2, ty, title.upper(), ft, INK, tr)
lat, lon = c['lat'], c['lon']
sub = '  ·  '.join([p for p in [(sub0 if HOLE else _crs.upper()), (c.get('place', '').upper().split(', ')[-1] if HOLE else c.get('place', '').upper().replace(', ', '  ·  ')), f"{abs(lat):.4f}° {'N' if lat >= 0 else 'S'}  {abs(lon):.4f}° {'W' if lon < 0 else 'E'}"] if p])
fs2 = F('jost400', U * 1.2); trs = U * .42
while spw(sub, fs2, trs) > W * .74: fs2 = F('jost400', fs2.size * .95); trs *= .95
if HER and not HOLE:
    _o1, _o2, _o3 = (2.2, 4.5, 6.2) if land_o else (2.6, 5.4, 7.4)   # 27 Nov: landscape title block tighter so the coordinates clear the border
    _cy = ty + U * _o1
    for sgn in (-1, 1): dr.line([(W / 2 + sgn * U * 2.2, _cy), (W / 2 + sgn * U * 16, _cy)], fill=INK2, width=max(1, int(U * .08)))
    dr.polygon([(W / 2, _cy - U * .7), (W / 2 + U * .7, _cy), (W / 2, _cy + U * .7), (W / 2 - U * .7, _cy)], outline=INK, fill=PAPER, width=max(1, int(U * .1)))
    _pl = c.get('place', '').upper(); _pl = _pl if 'IRELAND' in _pl else _pl + ', IRELAND'; _pl = (_crs.upper() + '  ·  ' + _pl) if _crs else _pl; _co = f"{abs(lat):.4f}° {'N' if lat >= 0 else 'S'}    {abs(lon):.4f}° {'W' if lon < 0 else 'E'}"
    _f3 = fs2; _t3 = trs
    while spw(_pl, _f3, _t3) > W * .74: _f3 = F('jost400', _f3.size * .95); _t3 *= .95
    spd(W / 2, ty + U * _o2, _pl, _f3, INK, _t3); spd(W / 2, ty + U * _o3, _co, F('jost400', U * 1.05), INK2, trs * .8)
else:
    dr.line([(W / 2 - U * 3, ty + U * 2.2), (W / 2 + U * 3, ty + U * 2.2)], fill=INK2, width=max(1, int(U * .1)))
    spd(W / 2, ty + U * 4.4, sub, fs2, INK2, trs)
msg = os.environ.get('MSG', '').strip()
if msg:
    z = U * 2.0; fm = F('corm500i', z)
    while fm.getlength(msg) > W * .7 and z > 10: z -= 2; fm = F('corm500i', z)
    dr.text((W / 2, MG * .6), msg, font=fm, fill=INK2, anchor='ms')
if os.environ.get('GEODUMP'):
    _gd = json.load(open(os.environ['GEODUMP'])); _gd['strip'] = _strip; json.dump(_gd, open(os.environ['GEODUMP'], 'w'))
im.save(out, quality=94); print(out, im.size, len(crowns), 'trees', step, 'm contours')
