# qa_sat.py KEY : checks every hole against the satellite photo with simple, independent photo tests.
#  bunkers : each of ours must be sand in the photo (no made-up bunkers); every sand patch in the photo must be ours (none missing)
#  water   : ours must be water in the photo; big water in the photo must be ours
#  trees   : our tree crowns vs the photo's tree canopy (overlap score)
#  greens/fairways : our shapes (as rendered) vs the photo classifier's map (overlap score)
import sys, json, math, numpy as np, cv2
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
key = sys.argv[1]; MPP = .3
d = json.load(open(f'/home/user/avendro-web-v2/partee-site/courses/{key}.json'))
zone_all = unary_union([LineString(h['pts']).buffer(110) for h in d['holes']]); x0, y0, x1, y1 = zone_all.bounds
im = cv2.imread(f'/tmp/claude-0/osm/sat/{key}-{MPP}.png'); H, W = im.shape[:2]
cls = np.load(f'/tmp/claude-0/osm/sat/{key}-cls.npy')
CL = ['rough', 'fairway', 'green', 'bunker', 'water', 'trees', 'heather', 'hard']
lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); L = lab[..., 0] * 100 / 255; A = lab[..., 1] - 128; B = lab[..., 2] - 128
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32); S = hsv[..., 1]; V = hsv[..., 2]
def rast(polys, buf=0):
    m = np.zeros((H, W), np.uint8)
    for p in polys:
        p = p.buffer(buf) if buf else p
        for g in getattr(p, 'geoms', [p]):
            if g.geom_type != 'Polygon' or g.is_empty: continue
            cv2.fillPoly(m, [np.round(np.c_[(np.array(g.exterior.xy[0]) - x0) / MPP, (y1 - np.array(g.exterior.xy[1])) / MPP]).astype(np.int32)], 1)
    return m.astype(bool)
F = {}
for k, c in d['feats']:
    try: F.setdefault(k, []).append(Polygon(list(zip(c[0::2], c[1::2]))).buffer(0))
    except Exception: pass
def close(ps, r):
    g = unary_union(ps).buffer(r, 12).buffer(-r, 12); return [p for p in getattr(g, 'geoms', [g]) if p.geom_type == 'Polygon' and p.area > 3]
ours = {'bunker': rast(close(F.get('bunker', []), .4)), 'water': rast(F.get('water', [])), 'fairway': rast(close(F.get('fairway', []), 4)),
        'green': rast(close(F.get('green', []), 1.5)), 'hard': rast(F.get('path', []) + F.get('building', []) + F.get('parking', []) + F.get('road', []), 1.5), 'island': rast(F.get('island', []))}
tr = np.zeros((H, W), np.uint8)
for c in d.get('crowns', []): cv2.circle(tr, (int((c[0] - x0) / MPP), int((y1 - c[1]) / MPP)), int(c[2] / MPP), 1, -1)
ours['trees'] = tr.astype(bool)
grass = (cls == 0) | (cls == 1); gL = np.median(L[grass]) if grass.any() else 45
hardp = cv2.dilate(((cls == 7) | ours['hard']).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))).astype(bool) | cv2.dilate(rast(F.get('building', [])).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))).astype(bool)
import os
TJ = f'/tmp/claude-0/osm/teacher/{key}.json'
tg = [Polygon(list(zip(c[0::2], c[1::2]))).buffer(0) for k, c in json.load(open(TJ))['feats'] if k == 'green'] if os.path.exists(TJ) else []
photo = {'sand': (L > gL + 22) & (S < 90) & (B > 11) & ~hardp,
         'water': (cls == 4) & (V < 110),
         'trees': cls == 5, 'fairway': cls == 1, 'green': rast(tg) if tg else cls == 2}   # greens: photo-snapped (SAM) outline
loose = (L > gL + 12) & (B > 8)
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)); photo['sand'] = cv2.morphologyEx(photo['sand'].astype(np.uint8), cv2.MORPH_OPEN, k3).astype(bool)
from scipy.ndimage import binary_fill_holes as ndi_fill
def iou(a, b): u = (a | b).sum(); return 1.0 if u == 0 else float((a & b).sum() / u)
def blobs(m, amin_m2):
    n, lb, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8); out = []
    for i in range(1, n):
        if st[i, 4] * MPP * MPP >= amin_m2: out.append((lb == i, st[i, 4] * MPP * MPP, cen[i]))
    return out
report = {}
for h in d['holes']:
    if len(h['pts']) < 2: continue
    z = rast([LineString(h['pts']).buffer(45)]); r = {'par': h.get('par')}
    # bunkers
    made, miss = [], []
    for m, a, c in blobs(ours['bunker'] & z, 2):
        if (loose & m).sum() / m.sum() < .35:   # made-up = not even shaded sand
            if a < 15 and (ours['bunker'] & ~m & cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(bool)).any(): continue   # sliver of a real bunker
            if a < 15 and (photo['sand'] & cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(bool)).sum() > 3 * m.sum(): continue   # sliver beside a real bunker
            made.append(round(a))
    for m, a, c in blobs(photo['sand'] & z, 5):
        if cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3).max() * MPP < 1.4: continue   # thin strips are paths/tracks, not bunkers
        if not (ours['bunker'] & cv2.dilate(m.astype(np.uint8), k3).astype(bool)).any(): miss.append(round(a))
    r['bunkers_made_up'] = made; r['bunkers_missing'] = miss
    r['bunker_overlap'] = round(iou(ours['bunker'] & z, photo['sand'] & z), 2)
    wo = ours['water'] & z
    _core = cv2.erode(wo.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(bool)   # ignore streams/ditches thinner than ~4 m
    wo = wo & cv2.dilate(_core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(bool)
    if wo.sum() * MPP * MPP < 300: wo = wo & False
    r['water_ours_is_water'] = round(float((photo['water'] & wo).sum() / max(1, wo.sum())), 2) if wo.sum() else None
    r['water_missing'] = [round(a) for m, a, c in blobs(photo['water'] & z, 150) if (wo & m).sum() / m.sum() < .3]
    r['trees_overlap'] = round(iou(ours['trees'] & z, photo['trees'] & z), 2)
    for k in (('green',) if str(h.get('par')) == '3' else ('fairway', 'green')): r[k + '_overlap'] = round(iou(ours[k] & z, photo[k] & z), 2) if (photo[k] & z).sum() > 50 else None
    # lake islands: land the photo shows inside a lake must not be painted as water
    pw = photo['water'] & z; filled = ndi_fill(pw) & ~pw
    r['lake_islands_missing'] = [round(a) for m, a, c in blobs(filled, 15) if (ours['water'] & ~ours['island'] & m).sum() / m.sum() > .5]
    # buildings/car parks: big hard surfaces in the photo must be drawn
    r['buildings_missing'] = [round(a) for m, a, c in blobs((cls == 7) & z, 250) if (ours['hard'] & m).sum() / m.sum() < .3]
    fails = []
    if made: fails.append(f'{len(made)} bunker(s) not in photo')
    if len([a for a in miss if a >= 8]): fails.append(f'{len([a for a in miss if a >= 8])} bunker(s) missing')
    if r['water_ours_is_water'] is not None and r['water_ours_is_water'] < .6: fails.append('water shape off')
    if r['water_missing']: fails.append('water missing')
    if r['lake_islands_missing']: fails.append(f"{len(r['lake_islands_missing'])} lake island(s) missing")
    if r['buildings_missing']: fails.append(f"{len(r['buildings_missing'])} building/car park area(s) missing")
    if r['trees_overlap'] < .55: fails.append('trees off')
    if r['green_overlap'] is not None and r['green_overlap'] < .7: fails.append('green shape off')
    if r.get('fairway_overlap') is not None and r['fairway_overlap'] < .6: fails.append('fairway shape off')
    r['pass'] = not fails; r['problems'] = fails; report[str(h['ref'])] = r
json.dump(report, open(f'/tmp/claude-0/osm/qa/{key}.json', 'w'), indent=1)
for k, r in report.items(): print(k.rjust(2), 'PASS' if r['pass'] else 'FAIL', r['problems'], '| bunk', r['bunker_overlap'], 'trees', r['trees_overlap'], 'fw', r.get('fairway_overlap'), 'gr', r['green_overlap'], 'water', r['water_ours_is_water'])
print(sum(r['pass'] for r in report.values()), '/', len(report), 'pass')
