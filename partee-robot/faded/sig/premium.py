# premium.py ISLAND.png COURSE.json THEME OUT.jpg [orient]
# Partee "Signature" poster: real traced course island, raised like a tabletop model, serif title block.
# Only verifiable facts (name, holes counted from data, county, coordinates). No invented est./par/yardage.
import sys, json, math, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
HERE = __file__.rsplit('/', 1)[0]
isl_p, cj, theme, out = sys.argv[1:5]; orient = sys.argv[5] if len(sys.argv) > 5 else 'auto'
c = json.load(open(cj))
isl = np.asarray(Image.open(isl_p).convert('RGBA'), np.float32) / 255
# trim to island
a = isl[..., 3]; ys, xs = np.where(a > .02); TY, TX = ys.min(), xs.min(); isl = isl[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
ih, iw = isl.shape[:2]
if orient == 'auto': orient = 'land' if iw >= ih * .95 else 'port'
import os
GAL = os.environ.get('LAYOUT', 'gallery') == 'gallery'   # 24 Nov 2026 Shane chose Gallery for every whole-course print; LAYOUT=classic = old title/compass layout
_sz = [int(v) for v in os.environ.get('SIZE', '3543x4724').split('x')]   # short x long side; default 30 x 40 cm at 300 dpi
W, H = (max(_sz), min(_sz)) if orient == 'land' else (min(_sz), max(_sz))
T = dict(
    charcoal=dict(bg=(20, 22, 19), bg2=(30, 33, 28), ink=(201, 169, 106), ink2=(160, 138, 95), wall=(52, 44, 30), grid=None, emblem='shamrock', panel=False),
    ivory=dict(bg=(244, 241, 233), bg2=(236, 232, 221), ink=(28, 40, 32), ink2=(92, 96, 82), wall=(96, 82, 58), grid=None, emblem='compass', panel=False),
    midnight=dict(bg=(19, 36, 27), bg2=(25, 45, 34), ink=(222, 204, 160), ink2=(160, 158, 132), wall=(10, 20, 15), grid=None, emblem='shamrock', panel=False),
    navy=dict(bg=(17, 33, 61), bg2=(23, 42, 76), ink=(214, 190, 140), ink2=(150, 165, 190), wall=(20, 28, 40), grid=(40, 62, 100), emblem='shamrock', panel=True),
)[theme]
# ---------- background: soft vignette + fine grain (+ blueprint grid)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.hypot((xx - W * .55) / W, (yy - H * .45) / H)
bg = np.array(T['bg2'], np.float32)[None, None] * (1 - np.clip(r * 1.3, 0, 1))[..., None] + np.array(T['bg'], np.float32)[None, None] * np.clip(r * 1.3, 0, 1)[..., None]
g = cv2.GaussianBlur(np.random.default_rng(3).normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.2) * 3.5
bg += g[..., None]
canvas = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
d = ImageDraw.Draw(canvas)
M = int(min(W, H) * .055)                                   # outer margin
if T['grid'] and not GAL:
    step = int(min(W, H) / 28)
    for x in range(M, W - M, step): d.line([(x, M), (x, H - M)], fill=T['grid'], width=2)
    for y in range(M, H - M, step): d.line([(M, y), (W - M, y)], fill=T['grid'], width=2)
# thin inner keyline
if not GAL: d.rectangle([M * .55, M * .55, W - M * .55, H - M * .55], outline=T['ink2'] if theme != 'charcoal' else (70, 62, 45), width=3)
# ---------- map box: largest size that keeps clear of the type, as near the page centre as it can sit (Shane 24 Nov: centred on every one, never touching the compass)
U0 = min(W, H) / 100; MG = M * 1.15
if orient == 'land':
    KEEP = [(0, H * .17 - U0 * 11, W * .40, H * .17 + U0 * 10), (0, H * .86 - U0 * 5, max(W * .36, W * .085 + len(c.get('place', '')) * U0 * 1.9), H), (0, H * .33 - U0 * 4, W * .30, H * .33 + U0 * 2)]
    if T['panel']: KEEP.append((W * .81, 0, W, H))
else:
    KEEP = [(0, 0, W, H * .195), (0, H * .895, W, H)]
if GAL:   # Gallery: even margins, map centred above the one type block
    GTOP = min(W, H) * .085; MG = GTOP
    KEEP = [(0, H * (.835 if orient == 'land' else .862), W, H)]
_q = 8; _a = cv2.resize(isl[..., 3], (max(1, iw // _q), max(1, ih // _q)), interpolation=cv2.INTER_AREA) > .05
def _fits(s_, ox_, oy_):
    nw_, nh_ = iw * s_, ih * s_
    if ox_ < MG or oy_ < MG or ox_ + nw_ > W - MG or oy_ + nh_ > H - MG * 1.25: return False
    yy_, xx_ = np.nonzero(_a); X = ox_ + (xx_ + .5) * _q * s_; Y = oy_ + (yy_ + .5) * _q * s_; pad_ = U0 * 3
    for (k0, k1, k2, k3) in KEEP:
        if ((X > k0 - pad_) & (X < k2 + pad_) & (Y > k1 - pad_) & (Y < k3 + pad_)).any(): return False
    return True
s = min((W - 2 * MG) / iw, (H - 2.25 * MG) / ih); best = None
while best is None and s > .05:
    nw_, nh_ = iw * s, ih * s; c0x, c0y = (W - nw_) / 2, (H - nh_) / 2
    cand = sorted(((dx, dy) for dx in np.arange(-.3, .301, .01) for dy in np.arange(-.2, .201, .01)), key=lambda t: t[0] ** 2 + (t[1] * 1.3) ** 2)
    for dx, dy in cand:
        if _fits(s, c0x + dx * W, c0y + dy * H): best = (s, int(c0x + dx * W), int(c0y + dy * H)); break
    if best is None: s *= .96
s, ox, oy = best
if os.environ.get('SURR_SHRINK'):   # leave paper round the landscape halo
    _k = float(os.environ['SURR_SHRINK']); _cx, _cy = ox + iw * s / 2, oy + ih * s / 2; s *= _k; ox, oy = int(_cx - iw * s / 2), int(_cy - ih * s / 2)
nw, nh = int(iw * s), int(ih * s)
top = cv2.resize(isl, (nw, nh), interpolation=cv2.INTER_AREA)
SURR = os.environ.get('SURR'); ISLM = None   # 25 Nov Shane: the real landscape round the course, same colours and texture as the course, feathering out (surr2.py layer = island px + SURR_E margin)
if SURR:
    _e = int(os.environ.get('SURR_E', '0')); sr = np.asarray(Image.open(SURR).convert('RGBA'), np.float32) / 255
    y0_, x0_ = _e + TY, _e + TX; ia_ = isl[..., 3:]
    sr[y0_:y0_ + ih, x0_:x0_ + iw, :3] = sr[y0_:y0_ + ih, x0_:x0_ + iw, :3] * (1 - ia_) + isl[..., :3] * ia_
    ISLM = np.zeros(sr.shape[:2], np.float32); ISLM[y0_:y0_ + ih, x0_:x0_ + iw] = isl[..., 3]
    sx0 = max(0, int(math.ceil(x0_ - ox / s))); sy0 = max(0, int(math.ceil(y0_ - oy / s)))
    sx1 = min(sr.shape[1], int(x0_ + (W - ox) / s)); sy1 = min(sr.shape[0], int(y0_ + (H - oy) / s))
    ox = max(0, int(round(ox + (sx0 - x0_) * s))); oy = max(0, int(round(oy + (sy0 - y0_) * s)))
    nw, nh = min(W - ox, int((sx1 - sx0) * s)), min(H - oy, int((sy1 - sy0) * s))
    top = cv2.resize(sr[sy0:sy1, sx0:sx1], (nw, nh), interpolation=cv2.INTER_AREA); ISLM = cv2.resize(ISLM[sy0:sy1, sx0:sx1], (nw, nh), interpolation=cv2.INTER_AREA)
    yy_, xx_ = np.mgrid[oy:oy + nh, ox:ox + nw].astype(np.float32)   # feather finishes before the page edge and the name block
    ed = np.minimum.reduce([xx_, W - xx_, yy_, (H * (.835 if orient == 'land' else .862) - U0 * 2) - yy_]) / (min(W, H) * .11)
    top[..., 3] = np.maximum(top[..., 3] * np.clip(ed, 0, 1) ** 1.3, ISLM)
# ---------- raised model: side wall + contact shadow + long soft shadow + rim light
INK = os.environ.get('GRADE', 'ink') == 'ink'   # 'printed ink' look (Shane 24 Nov: map looked cartoony next to premium type)
A = top[..., 3]; th = max(6, int(min(W, H) * (.0045 if INK else .009)))           # slab thickness
pad = th * 12
Ab = np.zeros((nh + 2 * pad, nw + 2 * pad), np.float32); Ab[pad:pad + nh, pad:pad + nw] = A
base = np.pad(np.asarray(canvas, np.float32), ((pad, pad), (pad, pad), (0, 0)), mode='edge')
def comp(rgb, alpha, x, y):
    h_, w_ = alpha.shape; sl = (slice(y + pad, y + pad + h_), slice(x + pad, x + pad + w_))
    base[sl] = base[sl] * (1 - alpha[..., None]) + rgb * alpha[..., None]
MOUNT = os.environ.get('MOUNT', 'print')   # print (default 24 Nov) = inked into the paper; raised = old slab + shadow; plate = inked into the paper (Shane 24 Nov: "feels like it's just slapped on there")
PRINT = MOUNT in ('print', 'plate', 'soft')
if PRINT:
    paper = base.copy()
    if MOUNT == 'plate':   # etching plate mark: a faint pressed rectangle round the map
        pm = int(min(W, H) * .028); x0, y0, x1, y1 = ox - pm + pad, oy - pm + pad, ox + nw + pm + pad, oy + nh + pm + pad
        x0, y0 = max(x0, pad + int(M * .9)), max(y0, pad + int(M * .9)); x1, y1 = min(x1, pad + W - int(M * .9)), min(y1, pad + H - int(M * .9))
        R = np.zeros(base.shape[:2], np.float32); cv2.rectangle(R, (x0, y0), (x1, y1), 1, -1)
        Rs = cv2.GaussianBlur(R, (0, 0), 2.2)
        inner = cv2.GaussianBlur(np.roll(np.roll(1 - R, 3, 0), 3, 1) * R, (0, 0), 3) * .10   # inner shadow along top/left edge
        lip = cv2.GaussianBlur(np.roll(np.roll(R, 2, 0), 2, 1) * (1 - R), (0, 0), 1.5) * .25  # lit lip bottom/right
        base *= (1 - .012 * Rs - inner)[..., None]; base += (lip * 18)[..., None]
        paper = base.copy()
if not PRINT:
  sh = cv2.GaussianBlur(np.roll(Ab, int(th * 3.5), 0), (0, 0), th * 3.2) * (.28 if INK else .55)      # long soft shadow
  comp(np.zeros(3), sh, ox - pad, oy - pad)
  sh2 = cv2.GaussianBlur(np.roll(Ab, th + 2, 0), (0, 0), th * .5) * (.35 if INK else .6)               # contact shadow
  comp(np.zeros(3), sh2, ox - pad, oy - pad)
  wall = np.array(T['wall'], np.float32)
  if INK: wall = wall * .55 + np.array(T['bg'], np.float32) * .45   # paper-coloured edge, not brown soil
  for k in range(th, 0, -1):                                                         # the slab's cut side
      shade = .55 + .45 * (1 - k / th)
      comp((wall * shade)[None, None], np.roll(Ab, k, 0), ox - pad, oy - pad)
# top surface: gentle edge bevel (lit from top-left)
rgb = top[..., :3] * 255
dist = cv2.distanceTransform((A > .5).astype(np.uint8), cv2.DIST_L2, 5)
bev = np.clip(1 - dist / (th * 1.2), 0, 1)
gx = cv2.Sobel(cv2.GaussianBlur(A, (0, 0), th * .5), cv2.CV_32F, 1, 0); gy = cv2.Sobel(cv2.GaussianBlur(A, (0, 0), th * .5), cv2.CV_32F, 0, 1)
nl = np.hypot(gx, gy) + 1e-6; lit = (gx * .7 + gy * .7) / nl                            # edge facing up-left -> lit
if not PRINT: rgb *= (1 + bev * .22 * lit)[..., None]
# extra richness: slight contrast + saturation lift so the turf reads like a model under gallery light
if INK and os.environ.get('INKC') == '1':   # muted inks (colour now done in stamper.py; INKC=1 only for islands rendered before 24 Nov) (recipe agreed with the content thread 24 Nov): greens pulled toward olive/sage, highlights compressed; water/sand/paths untouched
    hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.float32) / 255, cv2.COLOR_RGB2HSV)   # H 0-360, S/V 0-1
    g_ = cv2.GaussianBlur(((hsv[..., 0] > 50) & (hsv[..., 0] < 160) & (hsv[..., 1] > .2)).astype(np.float32), (0, 0), 1.5)
    h2 = np.stack([.6 * hsv[..., 0] + .4 * 84, .68 * hsv[..., 1], (.12 + .8 * hsv[..., 2]) ** 1.12], -1)
    g_ = g_ * float(os.environ.get('INKS', '1'))   # strength: 1 = the muted look Shane picked 24 Nov (GRADE=bright for the old look)
    hsv = hsv * (1 - g_[..., None]) + h2 * g_[..., None]
    rgb = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB) * 255
elif not INK:
    lum = rgb.mean(-1, keepdims=True); rgb = lum + (rgb - lum) * 1.12
if INK:   # depth against the ivory: deepen mid/dark greens (trees, rough) so the course reads crisply on the paper, keep fairway lights
    _dp = float(os.environ.get('DEPTH', '1.15')); rgb = 255 * np.clip(rgb / 255, 0, 1) ** _dp
if INK and os.environ.get('TONE', '1') == '1':   # 24 Nov: links courses (no woods) read washed out on ivory: bring the island's mid-tone down to a set level + a little richness
    _m = (A if ISLM is None else ISLM) > .5; _L = (rgb[..., 0] * .3 + rgb[..., 1] * .59 + rgb[..., 2] * .11)[_m]
    if _L.size:
        _med = float(np.median(_L)); _tg = float(os.environ.get('TONE_MID', '112'))
        if _med > _tg:
            _g = math.log(_tg / 255) / math.log(_med / 255); rgb = 255 * np.clip(rgb / 255, 0, 1) ** _g
            _lum = rgb.mean(-1, keepdims=True); rgb = _lum + (rgb - _lum) * 1.12
            _bl = cv2.GaussianBlur(rgb, (0, 0), max(4, nw / 250)); rgb = rgb + (rgb - _bl) * .35   # local contrast: dunes and hollows show
if theme == 'midnight':   # 25 Nov: night recolour of the drawing (same tone curve as nightify.py), bunkers stay pale sand
    _lab = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.float32) / 255, cv2.COLOR_RGB2LAB); _L, _a, _b = _lab[..., 0], _lab[..., 1], _lab[..., 2]
    _Lo = 100 * (.10 + .62 * (_L / 100) ** 1.55); _sd = np.clip((_L - 78) / 10, 0, 1) * np.clip((_b - 8) / 10, 0, 1) * np.clip(1 - np.abs(_a) / 14, 0, 1)
    _Lo = _Lo * (1 - _sd) + _L * .9 * _sd
    rgb = cv2.cvtColor(np.dstack([_Lo, _a * .85, _b * .85 - 4 * (1 - _sd)]).astype(np.float32), cv2.COLOR_LAB2RGB) * 255
light = 1 + (.04 if INK else .10) * (1 - np.hypot((np.arange(nw)[None] - nw * .35) / nw, (np.arange(nh)[:, None] - nh * .3) / nh))
rgb *= light[..., None]
if PRINT:   # ink on paper: the paper's grain and tone show through the map, edge softly bleeds like real ink
    sl = (slice(oy + pad, oy + pad + nh), slice(ox + pad, ox + pad + nw)); pp = paper[sl]
    tone = pp / np.array(T['bg'], np.float32)                                   # paper grain + vignette as a multiplier
    if theme == 'ivory': rgb = rgb * tone ** 1.6   # paper grain through the ink (dark papers: the vignette would blow the map out)
    else: rgb = rgb * (1 + (tone.mean(-1, keepdims=True) - tone.mean()) * .3)
    rgb = rgb * .94 + pp * .06                                                   # a breath of paper through the ink
    A = cv2.GaussianBlur(A, (0, 0), .7)
    if MOUNT == 'soft':   # outer edge of the woods fades into the paper like ink on a watercolour sheet (only the outermost ~1% of the page)
        fw = min(W, H) * .009
        dt = cv2.distanceTransform((A > .5).astype(np.uint8), cv2.DIST_L2, 5)
        nz = cv2.GaussianBlur(np.random.default_rng(7).normal(0, 1, A.shape).astype(np.float32), (0, 0), fw * .6); nz /= nz.std() + 1e-6
        A = A * np.clip((dt + nz * fw * .25) / fw, 0, 1) ** .8
    edge = np.clip(cv2.GaussianBlur(A, (0, 0), 1.6) * (1 - cv2.erode((A > .5).astype(np.uint8), np.ones((3, 3), np.uint8))), 0, 1)
    if ISLM is None: rgb *= (1 - .10 * edge)[..., None]                                           # ink pools very slightly at the edge
comp(np.clip(rgb, 0, 255), A, ox, oy)
canvas = Image.fromarray(np.clip(base[pad:-pad, pad:-pad], 0, 255).astype(np.uint8)); d = ImageDraw.Draw(canvas)
# ---------- type
if os.environ.get('TDUMP'):   # where island px land on the poster (designer + robot draw shots/flag with it): poster = (island_px + off) * s + o
    json.dump({'s': s, 'ox': ox, 'oy': oy, 'offx': (_e - sx0) if SURR else -int(TX), 'offy': (_e - sy0) if SURR else -int(TY), 'W': W, 'H': H}, open(os.environ['TDUMP'], 'w'))
import os; FD = os.environ.get('FONTS', HERE + '/../../fonts')
F = lambda n, sz: ImageFont.truetype(f'{FD}/{n}.ttf', int(sz))
U = min(W, H) / 100
name = c['name']
import re
m_ = re.search(r'\s+(Golf (Club|Course|Links|Resort|Centre)|Golf (&|and) \w+ Club)', name)
if m_:
    line1 = name[:m_.start()]; rest = name[m_.start():].strip(' ,')
    line2 = ' · '.join(p.strip() for p in rest.split(',') if p.strip())
elif ' - ' in name: line1, line2 = [t.strip() for t in name.split(' - ', 1)]
elif ',' in name:  # e.g. 'Royal Portrush, Dunluce Links' -> title + second line
    line1, line2 = [t.strip() for t in name.split(',', 1)]
else: line1, line2 = name, 'Golf Club'
line1 = line1.replace(' (Old Course)', '')
line1 = re.sub(r'\s*(,|&|\band)\s*$', '', line1.strip())   # 'Tulfarris Hotel and' -> 'Tulfarris Hotel'
def spaced(t, f, track): return sum(f.getlength(ch) for ch in t) + track * (len(t) - 1)
def draw_sp(x, y, t, f, fill, track, anchor='l'):
    w = spaced(t, f, track); x = x - w / 2 if anchor == 'm' else (x - w if anchor == 'r' else x)
    for ch in t: d.text((x, y), ch, font=f, fill=fill, anchor='ls'); x += f.getlength(ch) + track
    return w
ink, ink2 = T['ink'], T['ink2']
if GAL:   # Gallery (Shane picked 24 Nov): one centred, letter-spaced type block under the map; no compass, keyline, facts or brand mark
    ty = H * (.893 if orient == 'land' else .905)
    ft = F('corm500', U * 4.2); tr = U * 1.25
    while spaced(line1.upper(), ft, tr) > W * .62: ft = F('corm500', ft.size * .95); tr *= .95
    if os.environ.get('TITLE'):   # single-hole print: 'HOLE 13' over 'DRUIDS GLEN · CO. WICKLOW · coords'
        _pl = c.get('place', ''); c['place'] = line1 + (', ' + _pl.split(', ')[-1] if _pl else ''); line1 = os.environ['TITLE']
    draw_sp(W / 2, ty, line1.upper(), ft, ink, tr, 'm')
    lat, lon = c['lat'], c['lon']
    sub = '  ·  '.join([p_ for p_ in [c.get('place', '').upper().replace(', ', '  ·  '), f"{abs(lat):.4f}° {'N' if lat >= 0 else 'S'}  {abs(lon):.4f}° {'W' if lon < 0 else 'E'}"] if p_])
    fs_ = F('jost400', U * 1.25); trs = U * .42
    while spaced(sub, fs_, trs) > W * .74: fs_ = F('jost400', fs_.size * .95); trs *= .95
    d.line([(W / 2 - U * 3, ty + U * 2.3), (W / 2 + U * 3, ty + U * 2.3)], fill=ink2, width=max(1, int(U * .1)))
    draw_sp(W / 2, ty + U * 4.6, sub, fs_, ink2, trs, 'm')
    msg = os.environ.get('MSG', '').strip()
    if msg:   # personal line: small italic, centred in the top margin (designer.js renderSig uses the same numbers)
        sz = U * 2.2; fm = F('corm500i', sz)
        while fm.getlength(msg) > W * .7 and sz > 10: sz -= 2; fm = F('corm500i', sz)
        d.text((W / 2, GTOP * .62), msg, font=fm, fill=ink2, anchor='ms')
else:
    if orient == 'land':
        tx, ty, al = W * .085, H * .17, 'l'
    else:
        tx, ty, al = W / 2, H * .13, 'm'
    ft = F('corm500', U * 9.5)
    size = U * 9.5
    maxw = W * .8
    if orient == 'land':  # as wide as possible without touching the course: stop short of the island in the title rows
        r0, r1 = int(ty - U * 10 - oy), int(ty + U * 12 - oy)
        band = A[max(0, r0):max(0, min(nh, r1))]
        cols = np.where((band > .1).any(0))[0] if band.size else []
        edge = ox + cols.min() if len(cols) else W
        maxw = max(W * .2, min(W * .4, edge - tx - U * 4))
    while spaced(line1.upper(), ft, U * 1.0) > maxw:
        size *= .93; ft = F('corm500', size)
    w1 = draw_sp(tx, ty, line1.upper(), ft, ink, U * 1.0, al)
    fs = F('jost400', U * 2.4)
    y2 = ty + U * 5.2
    w2 = spaced(line2.upper(), fs, U * .9)
    cx = tx + w1 / 2 if al == 'l' else tx
    draw_sp(cx, y2, line2.upper(), fs, ink, U * .9, 'm')
    rule = U * 4.5
    d.line([(cx - w2 / 2 - rule - U * 1.2, y2 - U * .8), (cx - w2 / 2 - U * 1.2, y2 - U * .8)], fill=ink, width=max(2, int(U * .15)))
    d.line([(cx + w2 / 2 + U * 1.2, y2 - U * .8), (cx + w2 / 2 + rule + U * 1.2, y2 - U * .8)], fill=ink, width=max(2, int(U * .15)))
    lat, lon = c['lat'], c['lon']
    coord = f"{abs(lat):.4f}° {'N' if lat >= 0 else 'S'}  ·  {abs(lon):.4f}° {'W' if lon < 0 else 'E'}"
    fi = F('corm500i', U * 2.6)
    d.text((cx, y2 + U * 4.6), coord, font=fi, fill=ink2, anchor='ms')
    # facts: holes counted from our traced data, county from the record
    holes = len([h for h in c['holes'] if len(h.get('pts', [])) > 1])
    fsm = F('jost500', U * 1.9)
    place = c.get('place', '').upper()
    # 24 Nov 2026: long places ("NEWTOWNMOUNTKENNEDY, CO. WICKLOW") ran into the course. Every text block now fits its box:
    # split at the comma onto two lines, then shrink the font until the widest line fits.
    def fit_lines(t, name, size, track, maxw, min_size):
        lines = [x.strip() for x in t.split(',')] if ',' in t else [t]
        lines = [lines[0] + ','] + lines[1:] if len(lines) > 1 else lines
        sz = size
        while sz > min_size and max(spaced(x, F(name, sz), track * sz / size) for x in lines) > maxw: sz -= 1
        if len(lines) > 1 and spaced(t, F(name, size), track) <= maxw: lines, sz = [t], size   # fits on one line as it is
        return lines, F(name, sz), track * sz / size
    county = place.split(', ')[-1] if ', ' in place else place
    if orient == 'land':
        lim = W * .33 - U * 2 - W * .085                                  # left column, clear of the course box (starts at .33 W)
        if not T['panel']:
            if not c.get('preview'): draw_sp(W * .085, H * .86, f'{holes} HOLES', fsm, ink, U * .45)   # satellite preview: holes not routed yet, so no count
            ls_, fl, tr_ = fit_lines(place, 'jost500', U * 1.9, U * .45, lim, U * 1.3)
            for i, t in enumerate(ls_): draw_sp(W * .085, H * .86 + U * 3.6 * (i + (0 if c.get('preview') else 1)), t, fl, ink2, tr_)
        elif ', ' in place:                                               # navy: town under the title column, county in the panel
            ls_, fl, tr_ = fit_lines(place.rsplit(', ', 1)[0], 'jost500', U * 1.9, U * .45, lim, U * 1.3)
            for i, t in enumerate(ls_): draw_sp(W * .085, H * .86 + U * 3.6 * i, t, fl, ink2, tr_)
    if orient == 'land' and not T['panel']:
        pass
    elif orient == 'land':
        px0, px1 = W * .835, W * .925; py0, py1 = H * .30, H * .78
        d.rectangle([px0, py0, px1, py1], outline=ink2, width=3)
        pcx = (px0 + px1) / 2
        fbig = F('corm500', U * 6)
        d.text((pcx, py0 + (py1 - py0) * .38), str(holes), font=fbig, fill=ink, anchor='ms')
        draw_sp(pcx, py0 + (py1 - py0) * .47, 'HOLES', F('jost400', U * 1.6), ink2, U * .45, 'm')
        d.line([(pcx - U * 2, py0 + (py1 - py0) * .58), (pcx + U * 2, py0 + (py1 - py0) * .58)], fill=ink2, width=3)
        pl = county.split(' ', 1) if county.startswith('CO.') else county.split(' ')
        pw = (px1 - px0) - U * 2; sz = U * 1.7
        while sz > U * 1.1 and max(spaced(t, F('jost500', sz), U * .45 * sz / (U * 1.7)) for t in pl) > pw: sz -= 1
        for i, t in enumerate(pl):
            draw_sp(pcx, py0 + (py1 - py0) * (.72 + i * .09), t, F('jost500', sz), ink, U * .45 * sz / (U * 1.7), 'm')
    else:
        t = place if c.get('preview') else f'{holes} HOLES   |   {place}'; sz = U * 1.9
        while sz > U * 1.3 and spaced(t, F('jost500', sz), U * .45 * sz / (U * 1.9)) > W * .84: sz -= 1
        draw_sp(W / 2, H * .93, t, F('jost500', sz), ink, U * .45 * sz / (U * 1.9), 'm')
    # emblem
    ex, ey, er = (cx, H * .64, U * 3.6) if orient == 'land' else (W * .5, H * .035 + U * 3.2, U * 2.6)
    if orient == 'land':   # compass goes in the first free corner, well clear of the course (Shane 24 Nov: St Anne's nearly touched it)
        _ys, _xs = np.nonzero(A > .05); _P = np.stack([_xs[::7] + ox, _ys[::7] + oy], 1).astype(np.float32)
        for cx_, cy_ in ((cx, H * .64), (W * .9, H * .15), (W * .9, H * .84), (cx, H * .5)):
            if np.hypot(_P[:, 0] - cx_, _P[:, 1] - cy_).min() > er * 1.35 + U * 3: ex, ey = cx_, cy_; break
        else: ex = None
    if T['panel'] and orient == 'land': ex, ey = (W * .835 + W * .925) / 2, H * .30 + U * 7
    if orient == 'port': ey = H * .045 + U * 2
    lw = max(3, int(U * .22))
    if T['emblem'] == 'shamrock' and not (orient == 'port') and ex is not None:
        tt = np.linspace(0, 2 * math.pi, 120)
        hx, hy = 16 * np.sin(tt) ** 3, -(13 * np.cos(tt) - 5 * np.cos(2 * tt) - 2 * np.cos(3 * tt) - np.cos(4 * tt))
        hx, hy = hx / 17 * er * .5, (hy + 5) / 17 * er * .5                         # heart, point down at its tip
        for ang in (-90, 30, 150):
            a_ = math.radians(ang + 90)                                             # point each heart's tip to the centre
            ca, sa = math.cos(a_), math.sin(a_)
            px_ = hx * ca - (hy - er * .62) * sa; py_ = hx * sa + (hy - er * .62) * ca
            d.line(list(zip(ex + px_, ey + py_)) + [(ex + px_[0], ey + py_[0])], fill=ink, width=lw, joint='curve')
        st = [(ex + math.sin(t_) * er * .18 + t_ * er * .25, ey + t_ * er * 1.1) for t_ in np.linspace(.15, 1.0, 20)]
        d.line(st, fill=ink, width=lw, joint='curve')
    elif T['emblem'] == 'compass' and orient == 'land' and ex is not None:
        R = er; d.line([(ex, ey - R), (ex, ey + R)], fill=ink2, width=2); d.line([(ex - R, ey), (ex + R, ey)], fill=ink2, width=2)
        for k in range(4):
            a_ = k * math.pi / 2; b_ = a_ + math.pi / 4
            p1 = (ex + math.sin(a_) * R, ey - math.cos(a_) * R); pl = (ex + math.sin(b_) * R * .18, ey - math.cos(b_) * R * .18); pr = (ex + math.sin(b_ - math.pi / 2) * R * .18, ey - math.cos(b_ - math.pi / 2) * R * .18)
            d.polygon([p1, pl, (ex, ey)], fill=ink); d.polygon([p1, pr, (ex, ey)], outline=ink, width=2)
        fn = F('jost400', U * 1.5)
        for t, (dx, dy) in zip('NESW', ((0, -1), (1, 0), (0, 1), (-1, 0))):
            d.text((ex + dx * R * 1.35, ey + dy * R * 1.35), t, font=fn, fill=ink, anchor='mm')
    # optional personal line (same place and size as the designer preview)
    msg = os.environ.get('MSG', '').strip()
    if msg:
        sz = U * (2.4 if orient == 'land' else 2.6); mw = W * (.25 if orient == 'land' else .8)
        fm = F('corm500i', sz)
        while fm.getlength(msg) > mw and sz > 10: sz -= 2; fm = F('corm500i', sz)
        if orient == 'land': d.text((W * .085, H * .33), msg, font=fm, fill=ink2, anchor='ls')
        else: d.text((W / 2, H * .905), msg, font=fm, fill=ink2, anchor='ms')
    # brand mark, tiny
    d.text((W / 2, H - M * .55 - U * .9), 'PARTEE PRINTS', font=F('jost400', U * 1.0), fill=ink2, anchor='ms')
# optional scorecard (24 Nov 2026): the whole poster shrinks to 88% (top-centred) and an 18-hole card sits in the freed band.
# Same fractions as renderSig()/drawSigCard() in partee-site/designer.js, so print = preview.
card = [v for v in os.environ.get('CARD', '').split(',')] if os.environ.get('CARD', '').strip(',') else None
if card and any(v.strip() for v in card):
    card = (card + [''] * 18)[:18]
    S_ = .88; small = canvas.resize((round(W * S_), round(H * S_)), Image.LANCZOS)
    full = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)); full.paste(small, (round((W - W * S_) / 2), round(H * .03))); canvas = full; d = ImageDraw.Draw(canvas)
    hs = c.get('holes') or []
    def _par(i):
        for h_ in hs:
            if str(h_.get('ref')) == str(i + 1) and h_.get('par'): return str(h_['par'])
        return ''
    pars = [_par(i) for i in range(18)]
    rows = [('HOLE', [str(i + 1) for i in range(18)], 'TOT')]
    if any(pars): rows.append(('PAR', pars, str(sum(int(v) for v in pars if v.isdigit()) or '')))
    rows.append(('SCORE', card, str(sum(int(v) for v in card if v.strip().isdigit()) or '')))
    rh = H * (.022 if orient == 'land' else .018); x0 = W * .12; cwid = W * .76; lw = cwid * .08; tw = cwid * .06; cell = (cwid - lw - tw) / 18
    y0 = H * (.03 + S_) + (H * (1 - .03 - S_) - len(rows) * rh) / 2 - H * .004
    fnt = F('jost400', rh * .52); fntb = F('jost500', rh * .52) if os.path.exists(FD + '/jost500.ttf') else fnt
    lc = tuple(int(v) for v in T['ink2'])
    for j, (lab, vals, tot) in enumerate(rows):
        y = y0 + j * rh; last = j == len(rows) - 1
        if j == 0: d.rectangle([x0, y, x0 + cwid, y + rh], fill=T['bg2'])
        d.rectangle([x0, y, x0 + cwid, y + rh], outline=lc, width=max(1, round(rh * .05)))
        ft = fntb if last else fnt
        d.text((x0 + lw * .14, y + rh / 2), lab, font=ft, fill=T['ink'], anchor='lm')
        for i, v in enumerate(vals):
            x = x0 + lw + i * cell; d.line([(x, y), (x, y + rh)], fill=lc, width=max(1, round(rh * .05)))
            d.text((x + cell / 2, y + rh / 2), v, font=ft, fill=T['ink'], anchor='mm')
        xt = x0 + cwid - tw; d.line([(xt, y), (xt, y + rh)], fill=lc, width=max(1, round(rh * .05)))
        d.text((xt + tw / 2, y + rh / 2), tot, font=ft, fill=T['ink'], anchor='mm')
canvas.save(out, quality=93)
print(out, orient, canvas.size)
