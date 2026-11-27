# real tree canopy from satellite -> extra crown points (local metres) so the heritage style shows the true tree density
import sys, json, math, io, urllib.request, numpy as np, cv2
from PIL import Image
course, out, satdir = sys.argv[1:4]
d = json.load(open(course)); lat, lon = d['lat'], d['lon']
xs = [x for _, xy in d['feats'] for x in xy[::2]]; ys = [y for _, xy in d['feats'] for y in xy[1::2]]
x0, x1, y0, y1 = min(xs) - 40, max(xs) + 40, min(ys) - 40, max(ys) + 40
import os; os.makedirs(satdir, exist_ok=True); sp = f'{satdir}/sat.jpg'
mpp = .5;  # request size W = min(4000, int((x1 - x0) / mpp)); H = int(W * (y1 - y0) / (x1 - x0))
if not os.path.exists(sp):
    c = math.cos(math.radians(lat)); ll = lambda x, y: (lon + x / (111320 * c), lat + y / 110574)
    (lo0, la0), (lo1, la1) = ll(x0, y0), ll(x1, y1)
    u = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={lo0},{la0},{lo1},{la1}&bboxSR=4326&imageSR=4326&size={W},{H}&format=jpg&f=image'
    Image.open(io.BytesIO(urllib.request.urlopen(u, timeout=120).read())).convert('RGB').save(sp, quality=92)
if os.path.exists(f'{satdir}/sat.json'): b = json.load(open(f'{satdir}/sat.json')); x0, x1, y0, y1 = b['x0'], b['x1'], b['y0'], b['y1']
else: json.dump(dict(x0=x0, x1=x1, y0=y0, y1=y1), open(f'{satdir}/sat.json', 'w'))
a = np.asarray(Image.open(sp).convert('RGB')); H, W = a.shape[:2]
hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(np.float32); v, h = hsv[..., 2], hsv[..., 0]
tex = cv2.GaussianBlur(np.abs(v - cv2.GaussianBlur(v, (0, 0), 1.5)), (0, 0), 3)
m = ((v < 95) & (h > 25) & (h < 95) & (tex > 4)).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
# never on mapped water/greens/fairways/bunkers (dark water edges look like trees)
px = lambda x, y: ((x - x0) / (x1 - x0) * W, (y1 - y) / (y1 - y0) * H)
no = np.zeros((H, W), np.uint8)
for k, xy in d['feats']:
    if k in ('water', 'green', 'fairway', 'bunker', 'tee'):
        cv2.fillPoly(no, [np.array([px(*p) for p in zip(xy[::2], xy[1::2])], np.int32)], 1)
no = cv2.dilate(no, np.ones((7, 7), np.uint8)); m[no > 0] = 0
# open water on the photo itself (very dark, smooth, big) + a 5 m bank: the mapped lakes are drawn smaller than the photo's, so their dark rims read as trees
wd = ((v < 48) & (tex < 6)).astype(np.uint8); wd = cv2.morphologyEx(wd, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
nw, lw, sw, _ = cv2.connectedComponentsWithStats(wd); bigw = np.zeros(nw, bool); bigw[1:] = sw[1:, cv2.CC_STAT_AREA] > 200 / (mpp * mpp)
wd = cv2.dilate(bigw[lw].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))); m[wd > 0] = 0
n, lab, st, _ = cv2.connectedComponentsWithStats(m)
keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 12 / (mpp * mpp) * .25   # >= ~3 m2 canopy
m = keep[lab].astype(np.uint8)
rng = np.random.default_rng(3); pts = []; step = 3.2
for yy in np.arange(y0, y1, step):
    for xx in np.arange(x0, x1, step):
        x = xx + rng.uniform(-.4, .4) * step; y = yy + rng.uniform(-.4, .4) * step
        j, i = px(x, y); i, j = int(i), int(j)
        if 0 <= i < H and 0 <= j < W and m[i, j]: pts.append([round(x, 1), round(y, 1), round(float(rng.uniform(4.2, 6.0)), 1), 'round'])
json.dump(pts, open(out, 'w')); print(len(pts), 'canopy crowns', f'{m.mean()*100:.1f}% canopy')
