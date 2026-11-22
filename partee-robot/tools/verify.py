# verify.py KEY [--apply] : test every bunker and lake against the satellite photo; ones that aren't there go in overlay 'drop'
# bunker = clearly brighter + paler than the grass around it; water = clearly darker than its surroundings (or blue)
import sys, json, math, urllib.request, os, numpy as np, cv2
from shapely.geometry import Polygon
key = sys.argv[1]; apply = '--apply' in sys.argv
R = '/home/user/avendro-web-v2'; CJ = f'{R}/partee-site/courses/{key}.json'; OV = f'{R}/partee-robot/overlay/{key}.json'
d = json.load(open(CJ)); c = math.cos(math.radians(d['lat']))
ll = lambda x, y: (d['lon'] + x / (111320 * c), d['lat'] + y / 110574)
merc = lambda lo, la: (lo * 20037508.34 / 180, math.log(math.tan(math.radians(90 + la) / 2)) * 6378137)
F = [(i, k, np.array(cs, float).reshape(-1, 2)) for i, (k, cs) in enumerate(d['feats']) if k in ('bunker', 'water')]
allp = np.vstack([q for _, _, q in F]); x0, y0 = allp.min(0) - 20; x1, y1 = allp.max(0) + 20; MPP = .4
W = int((x1 - x0) / MPP); H = int((y1 - y0) / MPP); im = np.zeros((H, W, 3), np.uint8); T = 1800
for ty in range(0, H, T):
    for tx in range(0, W, T):
        w, hh = min(T, W - tx), min(T, H - ty)
        (a, b), (cc, dd) = merc(*ll(x0 + tx * MPP, y1 - (ty + hh) * MPP)), merc(*ll(x0 + (tx + w) * MPP, y1 - ty * MPP))
        u = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={a},{b},{cc},{dd}&bboxSR=3857&imageSR=3857&size={w},{hh}&format=jpg&f=image'
        im[ty:ty + hh, tx:tx + w] = cv2.resize(cv2.imdecode(np.frombuffer(urllib.request.urlopen(u, timeout=90).read(), np.uint8), cv2.IMREAD_COLOR), (w, hh))
lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
px = lambda q: np.stack([(q[:, 0] - x0) / MPP, (y1 - q[:, 1]) / MPP], 1).astype(np.int32)
drop = []; chk = im.copy()
for i, k, q in F:
    P = Polygon(q)
    if P.area < (6 if k == 'bunker' else 60): continue
    mi = np.zeros((H, W), np.uint8); cv2.fillPoly(mi, [px(q)], 1)
    er = cv2.erode(mi, np.ones((3, 3), np.uint8)); mi = er if er.sum() > 8 else mi
    ring = cv2.dilate(mi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(10 / MPP) | 1,) * 2)) - cv2.dilate(mi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(3 / MPP) | 1,) * 2))
    if mi.sum() < 4 or ring.sum() < 4: continue
    Li = np.median(lab[..., 0][mi > 0]); Lr = np.median(lab[..., 0][ring > 0])
    frac_bright = (lab[..., 0][mi > 0] > Lr + 30).mean()
    if k == 'bunker': ok = frac_bright > .35
    else: ok = (Li < Lr - 12) or (np.median(hsv[..., 0][mi > 0]) > 85 and np.median(hsv[..., 0][mi > 0]) < 130)
    col = (0, 255, 0) if ok else (0, 0, 255); cv2.polylines(chk, [px(q)], True, col, 2)
    if not ok and k == 'bunker':   # water is reported on the picture only (streams under trees fail the test)
        rp = P.representative_point(); drop.append({'kind': k, 'll': [round(v, 7) for v in ll(rp.x, rp.y)], 'why': f'L in {Li:.0f} ring {Lr:.0f} bright {frac_bright:.2f}'})
s = 2400 / max(W, H); cv2.imwrite(f'/tmp/claude-0/osm/fw/verify-{key}.jpg', cv2.resize(chk, (int(W * s), int(H * s))), [cv2.IMWRITE_JPEG_QUALITY, 80])
nb = sum(1 for _, k, _ in F if k == 'bunker'); print(key, 'bunkers', nb, 'water', len(F) - nb, 'DROP', len(drop), [x['kind'] for x in drop].count('bunker'), 'bunkers')
if apply and drop:
    O = json.load(open(OV)) if os.path.exists(OV) else {'feats': [], 'holes': []}
    O['drop'] = [x for x in O.get('drop', []) if x['kind'] not in ('bunker', 'water')] + drop; json.dump(O, open(OV, 'w'), indent=0)
    print('written to overlay')

if apply:   # also take them out of the course file now, so renders are right without a full rebuild
    from shapely.geometry import Point
    pts = [Point(*[(a - d['lon']) * 111320 * c, (b - d['lat']) * 110574][0:2]) for a, b in (x['ll'] for x in drop)]
    keep = [f for f in d['feats'] if not (f[0] == 'bunker' and any(Polygon(np.array(f[1]).reshape(-1, 2)).distance(p) < 2 for p in pts))]
    print('course file', len(d['feats']), '->', len(keep)); d['feats'] = keep; json.dump(d, open(CJ, 'w'), separators=(',', ':'))
