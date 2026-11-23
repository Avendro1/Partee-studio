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
a = isl[..., 3]; ys, xs = np.where(a > .02); isl = isl[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
ih, iw = isl.shape[:2]
if orient == 'auto': orient = 'land' if iw >= ih * .95 else 'port'
import os
_sz = [int(v) for v in os.environ.get('SIZE', '3543x4724').split('x')]   # short x long side; default 30 x 40 cm at 300 dpi
W, H = (max(_sz), min(_sz)) if orient == 'land' else (min(_sz), max(_sz))
T = dict(
    charcoal=dict(bg=(20, 22, 19), bg2=(30, 33, 28), ink=(201, 169, 106), ink2=(160, 138, 95), wall=(52, 44, 30), grid=None, emblem='shamrock', panel=False),
    ivory=dict(bg=(241, 234, 219), bg2=(232, 223, 203), ink=(28, 40, 32), ink2=(92, 96, 82), wall=(96, 82, 58), grid=None, emblem='compass', panel=False),
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
if T['grid']:
    step = int(min(W, H) / 28)
    for x in range(M, W - M, step): d.line([(x, M), (x, H - M)], fill=T['grid'], width=2)
    for y in range(M, H - M, step): d.line([(M, y), (W - M, y)], fill=T['grid'], width=2)
# thin inner keyline
d.rectangle([M * .55, M * .55, W - M * .55, H - M * .55], outline=T['ink2'] if theme != 'charcoal' else (70, 62, 45), width=3)
# ---------- map box
if orient == 'land':
    bx0, by0, bx1, by1 = int(W * .33), int(H * .08), int(W * (.79 if T['panel'] else .94)), int(H * .92)
else:
    bx0, by0, bx1, by1 = int(W * .07), int(H * .215), int(W * .93), int(H * .872)
s = min((bx1 - bx0) / iw, (by1 - by0) / ih)
nw, nh = int(iw * s), int(ih * s)
top = cv2.resize(isl, (nw, nh), interpolation=cv2.INTER_AREA)
ox, oy = bx0 + (bx1 - bx0 - nw) // 2, by0 + (by1 - by0 - nh) // 2
# ---------- raised model: side wall + contact shadow + long soft shadow + rim light
A = top[..., 3]; th = max(6, int(min(W, H) * .009))           # slab thickness
pad = th * 12
Ab = np.zeros((nh + 2 * pad, nw + 2 * pad), np.float32); Ab[pad:pad + nh, pad:pad + nw] = A
base = np.pad(np.asarray(canvas, np.float32), ((pad, pad), (pad, pad), (0, 0)), mode='edge')
def comp(rgb, alpha, x, y):
    h_, w_ = alpha.shape; sl = (slice(y + pad, y + pad + h_), slice(x + pad, x + pad + w_))
    base[sl] = base[sl] * (1 - alpha[..., None]) + rgb * alpha[..., None]
sh = cv2.GaussianBlur(np.roll(Ab, int(th * 3.5), 0), (0, 0), th * 3.2) * .55      # long soft shadow
comp(np.zeros(3), sh, ox - pad, oy - pad)
sh2 = cv2.GaussianBlur(np.roll(Ab, th + 2, 0), (0, 0), th * .5) * .6               # contact shadow
comp(np.zeros(3), sh2, ox - pad, oy - pad)
wall = np.array(T['wall'], np.float32)
for k in range(th, 0, -1):                                                         # the slab's cut side
    shade = .55 + .45 * (1 - k / th)
    comp((wall * shade)[None, None], np.roll(Ab, k, 0), ox - pad, oy - pad)
# top surface: gentle edge bevel (lit from top-left)
rgb = top[..., :3] * 255
dist = cv2.distanceTransform((A > .5).astype(np.uint8), cv2.DIST_L2, 5)
bev = np.clip(1 - dist / (th * 1.2), 0, 1)
gx = cv2.Sobel(cv2.GaussianBlur(A, (0, 0), th * .5), cv2.CV_32F, 1, 0); gy = cv2.Sobel(cv2.GaussianBlur(A, (0, 0), th * .5), cv2.CV_32F, 0, 1)
nl = np.hypot(gx, gy) + 1e-6; lit = (gx * .7 + gy * .7) / nl                            # edge facing up-left -> lit
rgb *= (1 + bev * .22 * lit)[..., None]
# extra richness: slight contrast + saturation lift so the turf reads like a model under gallery light
lum = rgb.mean(-1, keepdims=True); rgb = lum + (rgb - lum) * 1.12
light = 1 + .10 * (1 - np.hypot((np.arange(nw)[None] - nw * .35) / nw, (np.arange(nh)[:, None] - nh * .3) / nh))
rgb *= light[..., None]
comp(np.clip(rgb, 0, 255), A, ox, oy)
canvas = Image.fromarray(np.clip(base[pad:-pad, pad:-pad], 0, 255).astype(np.uint8)); d = ImageDraw.Draw(canvas)
# ---------- type
import os; FD = os.environ.get('FONTS', HERE + '/../../fonts')
F = lambda n, sz: ImageFont.truetype(f'{FD}/{n}.ttf', int(sz))
U = min(W, H) / 100
name = c['name']
import re
m_ = re.search(r'\s+(Golf (Club|Course|Links|Resort)|Golf & Country Club)', name)
if m_:
    line1 = name[:m_.start()]; rest = name[m_.start():].strip(' ,')
    line2 = ' · '.join(p.strip() for p in rest.split(',') if p.strip())
else: line1, line2 = name, 'Golf Club'
line1 = line1.replace(' (Old Course)', '')
def spaced(t, f, track): return sum(f.getlength(ch) for ch in t) + track * (len(t) - 1)
def draw_sp(x, y, t, f, fill, track, anchor='l'):
    w = spaced(t, f, track); x = x - w / 2 if anchor == 'm' else (x - w if anchor == 'r' else x)
    for ch in t: d.text((x, y), ch, font=f, fill=fill, anchor='ls'); x += f.getlength(ch) + track
    return w
ink, ink2 = T['ink'], T['ink2']
if orient == 'land':
    tx, ty, al = W * .085, H * .17, 'l'
else:
    tx, ty, al = W / 2, H * .13, 'm'
ft = F('corm500', U * 9.5)
size = U * 9.5
while spaced(line1.upper(), ft, U * 1.0) > (W * .26 if orient == 'land' else W * .8):
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
if orient == 'land' and not T['panel']:
    draw_sp(W * .085, H * .86, f'{holes} HOLES', fsm, ink, U * .45)
    draw_sp(W * .085, H * .86 + U * 3.6, place, fsm, ink2, U * .45)
elif orient == 'land':
    px0, px1 = W * .835, W * .925; py0, py1 = H * .30, H * .78
    d.rectangle([px0, py0, px1, py1], outline=ink2, width=3)
    pcx = (px0 + px1) / 2
    fbig = F('corm500', U * 6)
    d.text((pcx, py0 + (py1 - py0) * .38), str(holes), font=fbig, fill=ink, anchor='ms')
    draw_sp(pcx, py0 + (py1 - py0) * .47, 'HOLES', F('jost400', U * 1.6), ink2, U * .45, 'm')
    d.line([(pcx - U * 2, py0 + (py1 - py0) * .58), (pcx + U * 2, py0 + (py1 - py0) * .58)], fill=ink2, width=3)
    for i, t in enumerate(place.replace('CO. ', 'CO. ').split(' ', 1) if place.startswith('CO.') else [place]):
        draw_sp(pcx, py0 + (py1 - py0) * (.72 + i * .09), t, F('jost500', U * 1.7), ink, U * .45, 'm')
else:
    draw_sp(W / 2, H * .93, f'{holes} HOLES   |   {place}', fsm, ink, U * .45, 'm')
# emblem
ex, ey, er = (W * .915, H * .14, U * 4.2) if orient == 'land' else (W * .5, H * .035 + U * 3.2, U * 2.6)
if T['panel'] and orient == 'land': ex, ey = (W * .835 + W * .925) / 2, H * .30 + U * 7
if orient == 'port': ey = H * .045 + U * 2
lw = max(3, int(U * .22))
if T['emblem'] == 'shamrock' and not (orient == 'port'):
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
elif T['emblem'] == 'compass' and orient == 'land':
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
canvas.save(out, quality=93)
print(out, orient, canvas.size)
