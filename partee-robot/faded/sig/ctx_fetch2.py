# fetch.py COURSE.json OUT.json [R m] : real surroundings from OpenStreetMap (sea, beach, dunes, salt marsh, scrub, water)
# in the course's local metres. Uses the OSM API map call in small tiles (Overpass mirrors were timing out).
import json, sys, math, os, subprocess, xml.etree.ElementTree as ET
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union, split, linemerge, polygonize
d = json.load(open(sys.argv[1])); lat0, lon0 = d['lat'], d['lon']; kx = 111320 * math.cos(math.radians(lat0)); ky = 110574
hl = unary_union([LineString(h['pts']) for h in d['holes'] if len(h['pts']) > 1])
R = float(sys.argv[3]) if len(sys.argv) > 3 else 500
bx = hl.buffer(R).envelope; x0, y0, x1, y1 = bx.bounds
s, n, w, e = lat0 + y0 / ky, lat0 + y1 / ky, lon0 + x0 / kx, lon0 + x1 / kx
N, WY, REL = {}, {}, []
cache = os.path.join(os.path.dirname(os.path.abspath(sys.argv[2])), d['key'] + '-osm')
os.makedirs(cache, exist_ok=True)
nt = max(1, math.ceil((n - s) / .02)); mt = max(1, math.ceil((e - w) / .03))
for i in range(nt):
    for j in range(mt):
        a, b = s + (n - s) * i / nt, s + (n - s) * (i + 1) / nt; c, dd = w + (e - w) * j / mt, w + (e - w) * (j + 1) / mt
        f = os.path.join(cache, f'{i}-{j}.osm')
        if not os.path.exists(f) or os.path.getsize(f) < 200:
            subprocess.run(['curl', '-s', '-m', '120', '-A', 'partee-map-render', '-o', f, f'https://api.openstreetmap.org/api/0.6/map?bbox={c:.5f},{a:.5f},{dd:.5f},{b:.5f}'])
        try: root = ET.parse(f).getroot()
        except ET.ParseError: print('tile failed', f); continue
        for el in root:
            if el.tag == 'node': N[el.get('id')] = (float(el.get('lon')), float(el.get('lat')))
            elif el.tag == 'way': WY[el.get('id')] = ([nd.get('ref') for nd in el.findall('nd')], {t.get('k'): t.get('v') for t in el.findall('tag')})
            elif el.tag == 'relation':
                tg = {t.get('k'): t.get('v') for t in el.findall('tag')}
                if tg.get('type') == 'multipolygon': REL.append(([(m.get('ref'), m.get('role')) for m in el.findall('member') if m.get('type') == 'way'], tg))
L = lambda ids: [((N[i][0] - lon0) * kx, (N[i][1] - lat0) * ky) for i in ids if i in N]
def kind(t):
    nat = t.get('natural'); k = {'beach': 'beach', 'sand': 'sand', 'dune': 'dune', 'wetland': 'marsh', 'water': 'water', 'bay': 'sea', 'scrub': 'scrub', 'heath': 'scrub', 'grassland': 'grass'}.get(nat)
    if k == 'marsh' and t.get('wetland') in ('tidalflat',): k = 'tidal'
    if k == 'marsh' and t.get('wetland') not in (None, 'saltmarsh', 'marsh', 'reedbed', 'wet_meadow', 'tidalflat'): k = None
    if not k and (nat == 'wood' or t.get('landuse') == 'forest'): k = 'wood'
    if not k and t.get('landuse') in ('farmland', 'meadow', 'grass', 'orchard'): k = 'field'
    if not k and 'building' in t: k = 'bld'
    return k
out = {k: [] for k in ('sea', 'beach', 'sand', 'tidal', 'dune', 'marsh', 'water', 'scrub', 'grass', 'wood', 'field', 'bld')}; coast = []; roads = []
RW = dict(motorway=14, trunk=12, primary=10, secondary=8, tertiary=7, unclassified=5, residential=5, service=3, track=2.5)
for wid, (ids, t) in WY.items():
    if t.get('natural') == 'coastline': coast.append(LineString(L(ids))); continue
    if t.get('highway') in RW:
        ln = LineString(L(ids)) if len(L(ids)) > 1 else None
        if ln is not None:
            ln = ln.intersection(bx)
            for g in getattr(ln, 'geoms', [ln]):
                if g.geom_type == 'LineString' and g.length > 5: roads.append([RW[t['highway']]] + [round(c, 1) for xy in g.coords for c in xy])
    k = kind(t); pts = L(ids)
    if k and len(pts) > 3 and ids[0] == ids[-1]:
        p = Polygon(pts).buffer(0).intersection(bx)
        if not p.is_empty: out[k].append(p)
for mem, t in REL:
    k = kind(t)
    if not k: continue
    outer = [LineString(L(WY[m][0])) for m, r in mem if r == 'outer' and m in WY and len(L(WY[m][0])) > 1]
    inner = [LineString(L(WY[m][0])) for m, r in mem if r == 'inner' and m in WY and len(L(WY[m][0])) > 1]
    if not outer: continue
    o = unary_union(list(polygonize(linemerge(outer))))
    if inner: o = o.difference(unary_union(list(polygonize(linemerge(inner)))))
    o = o.intersection(bx)
    if not o.is_empty: out[k].append(o)
if coast:   # sea = pieces of the frame on the RIGHT of the coastline's direction (OSM: land on the left)
    _u = unary_union(coast); cl = (linemerge(_u) if _u.geom_type == "MultiLineString" else _u).intersection(bx); pieces = list(polygonize(unary_union([cl, bx.exterior])))
    for pc in pieces:
        v = 0; pb = pc.exterior.buffer(3)
        for ln in getattr(cl, 'geoms', [cl]):
            cs = list(ln.coords)
            for (ax, ay), (bx_, by_) in zip(cs[:-1], cs[1:]):
                mx, my = (ax + bx_) / 2, (ay + by_) / 2; dx, dy = bx_ - ax, by_ - ay; l = math.hypot(dx, dy) or 1
                if l < 2 or not pb.contains(Point(mx, my)): continue
                if pc.contains(Point(mx + dy / l * 1.5, my - dx / l * 1.5)): v += 1
                elif pc.contains(Point(mx - dy / l * 1.5, my + dx / l * 1.5)): v -= 1
        if v > 0: out['sea'].append(pc)
# OSM coastline rings round small islands are often broken: never let 'sea' cover mapped land or the course itself
_land = unary_union([p for k in ('beach', 'sand', 'tidal', 'dune', 'marsh', 'scrub', 'grass') for p in out[k]]
                    + [Polygon(list(zip(c[0::2], c[1::2]))).buffer(30) for k, c in d['feats'] if len(c) >= 6] + [hl.buffer(40)])
out['sea'] = [p.difference(_land) for p in out['sea']]
res = {k: [[round(c, 1) for xy in g.exterior.coords for c in xy] for p in v for g in getattr(p, 'geoms', [p]) if g.geom_type == 'Polygon' and g.area > 50] for k, v in out.items()}
res['road'] = roads; json.dump(res, open(sys.argv[2], 'w')); print({k: len(v) for k, v in res.items()}, 'coast ways', len(coast), 'tiles', nt * mt)
