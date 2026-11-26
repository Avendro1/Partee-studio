# surr2.py ISL.png FRAME.json COURSE.json CTX.json LIDAR.png OUT.png E PPM R : real surroundings drawn in the course's OWN colours/texture (sampled from the island), alpha 1 at the course edge feathering to 0 at R px
import sys, os, json, numpy as np, cv2
from PIL import Image
ip, fp, cp, xp, lp, out, E, PPM, R = sys.argv[1:10]; E = int(E); PPM = float(PPM); R = float(R)
isl = np.asarray(Image.open(ip).convert('RGBA'), np.float32)
f = json.load(open(fp)); c = json.load(open(cp))
Wp = np.array([h['pts'][0] for h in c['holes']]); Pp = np.array(f['tee_px'])
if len(Wp) >= 3: M, *_ = np.linalg.lstsq(np.c_[Wp, np.ones(len(Wp))], Pp, rcond=None)
else: M = np.array([[PPM, 0], [0, -PPM], [Pp[0][0] - PPM * Wp[0][0], Pp[0][1] + PPM * Wp[0][1]]])   # single hole: known scale, y up
M[2] -= [f['x0'] - E, f['y0'] - E]
ROTH = c.get('rot'); _cr, _sr = (np.cos(ROTH), np.sin(ROTH)) if ROTH is not None else (1., 0.)   # hole files are rotated (tee bottom, green top)
h0, w0 = isl.shape[:2]; H, W = h0 + 2 * E, w0 + 2 * E
A = np.zeros((H, W), np.float32); A[E:E + h0, E:E + w0] = isl[..., 3] / 255
# palette sampled from the island itself
ia = isl[..., 3] > 240; irgb = isl[..., :3]
dd = cv2.distanceTransform(ia.astype(np.uint8), cv2.DIST_L2, 5); band = (dd > 3) & (dd < 30)
ROUGH = np.median(irgb[band], 0); RSTD = irgb[band].std(0).mean()
L = irgb.mean(-1); hsv = cv2.cvtColor(irgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
SAND = np.median(irgb[ia & (L > 200) & (hsv[..., 1] < 60)], 0)
BEACH = SAND * .55 + np.array((214, 200, 160)) * .45
import os
PAPER_ = np.array([float(v) for v in os.environ.get('SURR_PAPER', '244,241,233').split(',')], np.float32); DARK_ = PAPER_.mean() < 128
# coastal (25 Nov, Shane: Old Head sea "why is it blue", glowing ring): sea is a muted slate tied to the style's own paper, lighter near the shore,
# sinking into the paper further out, so land and sea feather out together like Druids Glen. No saturated blue on any style.
if DARK_: SEA0 = PAPER_ * .22 + np.array((100, 116, 114), np.float32) * .78; SEA1 = PAPER_ * .80 + np.array((60, 76, 80), np.float32) * .20
else: SEA0 = np.array((150, 168, 170), np.float32); SEA1 = PAPER_ * .55 + np.array((150, 168, 170), np.float32) * .45   # stamper water / deep water after the ink grade
BEACH = BEACH * .55 + ROUGH * .45 if DARK_ else BEACH * .75 + ROUGH * .25   # strand muted toward the land tone, never a glowing band
MARSH = ROUGH * np.array((.92, 1.0, .95)); TIDAL = ROUGH * .45 + SEA0 * .35 + SAND * .20   # salt marsh / mud flats: grey-green like the photo
TREE = np.median(irgb[ia & (L < 75)], 0) if (ia & (L < 75)).sum() > 500 else ROUGH * .5
FIELD = ROUGH * .8 + np.array((170, 176, 120)) * .2
print('tree', TREE, 'rough', ROUGH, 'std', RSTD, 'sand', SAND)
def P(pts):
    pts = np.array(pts, float).reshape(-1, 2); pts = np.c_[pts[:, 0] * _cr - pts[:, 1] * _sr, pts[:, 0] * _sr + pts[:, 1] * _cr]
    return (np.c_[pts, np.ones(len(pts))] @ M).astype(np.int32)
ctx = json.load(open(xp))
def mask(keys, b=1.2):
    m = np.zeros((H, W), np.uint8)
    for k in keys:
        for poly in ctx.get(k, []):
            if len(poly) >= 6: cv2.fillPoly(m, [P(poly)], 1)
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), b)
rgb = np.ones((H, W, 3), np.float32) * ROUGH
def put(m, col, k=1.):
    global rgb; m = m[..., None] * k; rgb = rgb * (1 - m) + np.asarray(col, np.float32) * m
rng = np.random.default_rng(4)
def noise(sig, amp=1):
    n = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sig); return n / (n.std() + 1e-6) * amp
put(mask(['marsh', 'grass', 'scrub'], 2) * .7, MARSH)
tid = mask(['tidal'], 2); put(tid, TIDAL)
put(mask(['beach', 'sand'], 1.5), BEACH)
sea = mask(['sea'], 3.5 * PPM); wat = mask(['water'], 1.2)
din = cv2.GaussianBlur(cv2.distanceTransform((sea > .5).astype(np.uint8), cv2.DIST_L2, 5), (0, 0), 3)
deep = np.clip(din / (90 * PPM), 0, 1) ** .7
put(sea, SEA0[None, None] * (1 - deep[..., None]) + SEA1[None, None] * deep[..., None])
put(wat, SEA0)
if DARK_:   # shore glow (25 Nov, Old Head read as a cut-out): sea near the land picks up a soft land-tinted wash that dies away slowly
    _g = np.exp(-din / (R * .45))[..., None] * sea[..., None]
    _shore = ROUGH.astype(np.float32) * .55 + SEA0 * .45
    rgb[:] = rgb * (1 - _g * .75) + _shore * _g * .75
put(mask(['field'], 1.5), FIELD)
woodm = mask(['wood'], 1.5); put(woodm, TREE * 1.12)
land = np.clip(1 - sea - wat, 0, 1)
# grass texture like the course rough (fine grain + mottling), only on vegetated land
veg = land * (1 - mask(['beach', 'sand'], 1.5)) * (1 - tid * .6)
rgb *= (1 + (noise(.8, .05) + noise(4, .05) + noise(14, .04)) * veg)[..., None]
# real terrain from the laser survey: dunes, hollows, marsh creeks
if lp != '-':
    j = json.load(open(lp.rsplit('.', 1)[0] + '.json'))
    hs = cv2.imread(lp, cv2.IMREAD_GRAYSCALE).astype(np.float32); hp = hs - cv2.GaussianBlur(hs, (0, 0), 12)
    Mi = np.linalg.inv(np.r_[M.T, [[0, 0, 1]]]); yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wx = Mi[0, 0] * xx + Mi[0, 1] * yy + Mi[0, 2]; wy = Mi[1, 0] * xx + Mi[1, 1] * yy + Mi[1, 2]
    wx, wy = wx * _cr + wy * _sr, -wx * _sr + wy * _cr   # back to the course frame
    lx = ((wx - j['x0']) / (j['x1'] - j['x0']) * j['w']).astype(np.float32); ly = ((j['y1'] - wy) / (j['y1'] - j['y0']) * j['h']).astype(np.float32)
    hpw = cv2.remap(hp, lx, ly, cv2.INTER_LINEAR, borderValue=0)
    inl = ((lx >= 0) & (lx < j['w']) & (ly >= 0) & (ly < j['h'])).astype(np.float32)
    inl = cv2.GaussianBlur(cv2.erode(inl, np.ones((41, 41), np.uint8)), (0, 0), 14)
    rel = np.clip(hpw / 35, -1, 1) * inl * land
    rgb *= (1 - np.clip(-rel, 0, 1) * .30)[..., None]; rgb *= (1 + np.clip(rel, 0, 1) * .16)[..., None]
# woods: crown texture (blobs) on top of the laser canopy relief
cn = noise(2.2 * PPM, 1); put(woodm * np.clip(cn - .3, 0, 1) * .6, TREE * .75); put(woodm * np.clip(-cn - .6, 0, 1) * .5, TREE * 1.35)
rd = np.zeros((H, W), np.uint8)
for r_ in ctx.get('road', []):
    w_ = r_[0]; pts = P(r_[1:]); cv2.polylines(rd, [pts], False, 1, max(1, int(w_ * PPM)), cv2.LINE_AA)
rdm = cv2.GaussianBlur(rd.astype(np.float32), (0, 0), 1.6); put(rdm, (196, 190, 166), .55)   # painted: roads as a soft pale wash
bm = mask(['bld'], 2.2); put(bm, (188, 180, 160), .5)   # painted: buildings as soft warm dabs, no outlines
# sea: faint chart lines parallel to the shore + soft surf line
for d_ in (10, 25, 45, 70):
    put(np.clip(1 - np.abs(din - d_ * PPM) / 1.0, 0, 1) * sea * .10, SEA0 * 1.12)
put(np.clip(1 - din / (3 * PPM), 0, 1) * sea * .12, SEA0 * 1.15)   # faint shore line, no white surf ring
# painted finish: edge-preserving smoothing + soft colour washes (like the St Anne's look)
_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
rgb = cv2.edgePreservingFilter(_u8, flags=1, sigma_s=float(os.environ.get('PAINT_S', '25')), sigma_r=.3).astype(np.float32)
_wl = noise(40, .04); wash = np.stack([_wl + noise(40, .012), _wl + noise(40, .008), _wl + noise(40, .012)], -1)   # mostly tonal wash: no pink/green tints; rgb *= (1 + wash + noise(8, .03)[..., None])
# feather: solid at the course edge, fading out to nothing at R (gently irregular)
# gaps enclosed by the course (between holes) count as course, so they never fade to paper (25 Nov, Mount Juliet middle went pale)
_f = 8; _a = cv2.resize((A > .5).astype(np.uint8), (W // _f, H // _f), interpolation=cv2.INTER_AREA)
_k = max(3, int(R * .9 / _f) | 1); _c = cv2.morphologyEx(_a, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_k, _k)))
_c = cv2.resize(_c, (W, H), interpolation=cv2.INTER_LINEAR); Ac = np.maximum((A > .5).astype(np.uint8), _c)
dist = cv2.distanceTransform((Ac < 1).astype(np.uint8), cv2.DIST_L2, 5)
fa = np.clip(1 - (dist + noise(R * .12, R * .025)) / R, 0, 1); fa = fa ** 1.4
import os
if DARK_: rgb *= .80   # dark styles: surroundings sit a step darker than the course, like Druids Glen charcoal
if os.environ.get('HALO') == '2':   # best overall: detail readable near the course, lighter + greyer outward, watercolour-bleed edge
    t_ = np.clip(dist / R, 0, 1)[..., None]
    lum_ = rgb.mean(-1, keepdims=True); rgb = rgb * (1 - t_ * .5) + lum_ * t_ * .5            # desaturate outward
    k_ = t_ ** .7 * .7; rgb = rgb * (1 - k_) + np.array([float(v) for v in os.environ.get('SURR_PAPER', '244,241,233').split(',')], np.float32) * k_   # lighten toward the paper
    bleed = noise(R * .05, 1) * .6 + noise(R * .015, 1) * .4
    fa = np.clip(1 - (dist + bleed * R * .09) / R, 0, 1); fa = np.clip(fa * 1.25, 0, 1) ** 1.2; fa = np.maximum(fa, A)
if os.environ.get('HALO') == '3':   # 26 Nov: painted wash, not fog. Keep the land's own hue (no grey), tint toward a pale green-ivory,
    # hold the paint nearly solid then end on an irregular watercolour edge with a faint pigment rim; dark styles sink into their paper
    PAP = np.array([float(v) for v in os.environ.get('SURR_PAPER', '244,241,233').split(',')], np.float32)
    t_ = np.clip(dist / R, 0, 1)[..., None]
    lum_ = rgb.mean(-1, keepdims=True); rgb = rgb * (1 - t_ * .22) + lum_ * t_ * .22          # only a touch of desaturation
    TINT = ROUGH * .35 + PAP * .65 if not DARK_ else ROUGH * .30 + PAP * .70
    k_ = np.clip((t_ - .05) / .75, 0, 1) ** .9 * .72; rgb = rgb * (1 - k_) + TINT * k_       # wash toward a pale tint of the land
    bleed = noise(R * .08, 1) * .6 + noise(R * .025, 1) * .3 + noise(R * .01, 1) * .1
    e_ = (dist + bleed * R * .09) / (R * .82)                                                          # 0 at course .. 1 at edge
    fa = np.clip((1 - e_) / .10, 0, 1); fa = fa * fa * (3 - 2 * fa)                            # solid most of the way, short soft end
    fa = fa * (.62 + .38 * np.clip(1 - e_ / .6, 0, 1) ** 1.5)                                        # a little transparent toward the edge
    rim = np.exp(-((e_ - .93) / .03) ** 2)[..., None] * (fa[..., None] > .05)
    rgb = rgb * (1 - rim * (.14 if not DARK_ else -.08))                                       # pigment pools at the edge (lighter on dark)
    rgb *= (1 + noise(1.2, .025))[..., None]                                                   # paper tooth
    fa = np.maximum(fa, A)
if os.environ.get('HALO') == '1':   # tight halo: lighter as it goes out, paper shows round it
    k_ = np.clip(dist / R, 0, 1)[..., None] ** .5 * .8; rgb = rgb * (1 - k_) + np.array((244, 241, 233), np.float32) * k_; fa = fa ** 1.3
fa = np.maximum(fa, A)
Image.fromarray(np.dstack([np.clip(rgb, 0, 255), fa * 255]).astype(np.uint8)).save(out); print(out, W, H)
