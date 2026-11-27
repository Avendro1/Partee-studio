# fac_fetch.py COURSE.json OUT.json : the club's own buildings (clubhouse, pro shop, sheds) from OpenStreetMap, in local metres.
# 27 Nov (Shane: "club houses should be in every map with the proper size and detail and proportions if they are in the facility").
# Facility = buildings inside the OSM golf course boundary (leisure=golf_course), or tagged as a clubhouse; fallback = the largest
# building near the first tee / last green. Output {"fac": [[x,y,...]], "how": "..."}; plan.py FAC= draws them whole, never clipped.
import json, sys, math, os, subprocess, xml.etree.ElementTree as ET
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union, linemerge, polygonize
d = json.load(open(sys.argv[1])); lat0, lon0 = d['lat'], d['lon']; kx = 111320 * math.cos(math.radians(lat0)); ky = 110574
H = [h for h in d['holes'] if len(h.get('pts', [])) > 1]
hl = unary_union([LineString(h['pts']) for h in H]); x0, y0, x1, y1 = hl.buffer(400).envelope.bounds
s, n, w, e = lat0 + y0 / ky, lat0 + y1 / ky, lon0 + x0 / kx, lon0 + x1 / kx
cache = os.path.join(os.path.dirname(os.path.abspath(sys.argv[2])), 'fac-osm'); os.makedirs(cache, exist_ok=True)
N, WY, REL = {}, {}, []
nt = max(1, math.ceil((n - s) / .012)); mt = max(1, math.ceil((e - w) / .02))
for i in range(nt):
    for j in range(mt):
        a, b = s + (n - s) * i / nt, s + (n - s) * (i + 1) / nt; c, dd = w + (e - w) * j / mt, w + (e - w) * (j + 1) / mt
        f = os.path.join(cache, f'{i}-{j}.osm')
        for t in range(4):
            if os.path.exists(f) and os.path.getsize(f) > 200: break
            subprocess.run(['curl', '-s', '-m', '120', '-A', 'partee-map-render', '-o', f, f'https://api.openstreetmap.org/api/0.6/map?bbox={c:.5f},{a:.5f},{dd:.5f},{b:.5f}'])
        try: root = ET.parse(f).getroot()
        except Exception: print('tile failed', f); continue
        for el in root:
            if el.tag == 'node': N[el.get('id')] = (float(el.get('lon')), float(el.get('lat')))
            elif el.tag == 'way': WY[el.get('id')] = ([nd.get('ref') for nd in el.findall('nd')], {t.get('k'): t.get('v') for t in el.findall('tag')})
            elif el.tag == 'relation':
                tg = {t.get('k'): t.get('v') for t in el.findall('tag')}
                if tg.get('type') == 'multipolygon': REL.append(([(m.get('ref'), m.get('role')) for m in el.findall('member') if m.get('type') == 'way'], tg))
L = lambda ids: [((N[i][0] - lon0) * kx, (N[i][1] - lat0) * ky) for i in ids if i in N]
def poly(ids):
    c = L(ids); return Polygon(c).buffer(0) if len(c) > 3 else None
gc, B = [], []
for wid, (ids, t) in WY.items():
    if t.get('leisure') == 'golf_course' and ids[0] == ids[-1]:
        p = poly(ids); p and gc.append(p)
    if 'building' in t and ids[0] == ids[-1]:
        p = poly(ids)
        if p and p.area > 8: B.append((p, t))
for mem, t in REL:
    if t.get('leisure') == 'golf_course' or 'building' in t:
        ls = [LineString(L(WY[r][0])) for r, role in mem if r in WY and role != 'inner' and len(L(WY[r][0])) > 1]
        inn = [Polygon(L(WY[r][0])) for r, role in mem if r in WY and role == 'inner' and len(L(WY[r][0])) > 3]
        if not ls: continue
        g = unary_union(list(polygonize(linemerge(ls))))
        for q in inn: g = g.difference(q)
        if g.is_empty: continue
        if t.get('leisure') == 'golf_course': gc.append(g)
        else: B.append((g, t))
course = unary_union([g for g in gc if g.distance(hl) < 50]) if gc else None
def clubby(t): return t.get('building') == 'clubhouse' or t.get('golf') == 'clubhouse' or t.get('club') == 'golf' or 'club' in (t.get('name', '') + t.get('amenity', '')).lower()
fac, how = [], ''
if course is not None and not course.is_empty:
    fac = [p for p, t in B if course.buffer(5).contains(p.representative_point()) and p.distance(hl) < 400]; how = 'inside golf course boundary'
tagged = [p for p, t in B if clubby(t) and p.distance(hl) < 600]
fac += [p for p in tagged if not any(p.equals(q) for q in fac)]
if tagged and not how: how = 'tagged clubhouse'
if not fac:
    ends = [Point(H[0]['pts'][0]), Point(H[-1]['pts'][-1])]
    c_ = [p for p, t in B if p.area >= 120 and min(p.distance(q) for q in ends) < 150]
    if c_: fac = [max(c_, key=lambda p: p.area / (1 + min(p.distance(q) for q in ends)))]; how = 'largest building by tee 1 / last green'
out = []
for g in fac:
    for p in ([g] if g.geom_type == 'Polygon' else list(g.geoms)):
        out.append([round(v, 1) for xy in p.exterior.coords for v in xy])
json.dump({'fac': out, 'how': how, 'boundary': bool(course is not None and not course.is_empty)}, open(sys.argv[2], 'w'))
print(d['key'], len(out), 'facility buildings,', how or 'none found', ', largest', round(max([Polygon(list(zip(o[::2], o[1::2]))).area for o in out], default=0)), 'm2')
