# detail.py COURSE.json IN.json OSMDIR OUT.json : real small OSM details (single trees, footpaths/tracks, hedges, walls, tree rows) near the course, local metres
import json, sys, math, glob, xml.etree.ElementTree as ET
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
c = json.load(open(sys.argv[1])); d = json.load(open(sys.argv[2])); lat0, lon0 = c['lat'], c['lon']
kx = 111320 * math.cos(math.radians(lat0)); ky = 110574
N, WY = {}, {}
for f in glob.glob(sys.argv[3] + '/*.osm'):
    for el in ET.parse(f).getroot():
        t = {x.get('k'): x.get('v') for x in el.findall('tag')}
        if el.tag == 'node': N[el.get('id')] = ((float(el.get('lon')) - lon0) * kx, (float(el.get('lat')) - lat0) * ky, t)
        elif el.tag == 'way': WY[el.get('id')] = ([n.get('ref') for n in el.findall('nd')], t)
def P(f): q = f[1]; return Polygon(list(zip(q[0::2], q[1::2]))).buffer(0)
play = unary_union([P(f) for f in d['feats'] if f[0] in ('fairway', 'green', 'tee', 'bunker')] + [LineString(h['pts']).buffer(30) for h in d['holes'] if len(h['pts']) > 1])
zone = play.buffer(40)   # only what sits on the course itself
out = {'trees': [], 'paths': [], 'hedges': [], 'walls': [], 'rows': []}
for i, (x, y, t) in N.items():
    if t.get('natural') == 'tree' and zone.contains(Point(x, y)): out['trees'].append([round(x, 1), round(y, 1)])
for i, (ids, t) in WY.items():
    pts = [(N[n][0], N[n][1]) for n in ids if n in N]
    if len(pts) < 2: continue
    k = None; hw = t.get('highway')
    if hw in ('footway', 'path', 'track', 'bridleway', 'cycleway') and t.get('footway') not in ('sidewalk', 'crossing'): k = 'paths'
    elif t.get('barrier') == 'hedge': k = 'hedges'
    elif t.get('barrier') in ('wall', 'fence', 'retaining_wall'): k = 'walls'
    elif t.get('natural') == 'tree_row': k = 'rows'
    if not k: continue
    g = LineString(pts).intersection(zone)
    for s in getattr(g, 'geoms', [g]):
        if s.geom_type == 'LineString' and s.length > 6: out[k].append([[round(a, 1), round(b, 1)] for a, b in s.coords] + ([hw] if k == 'paths' else []))
json.dump(out, open(sys.argv[4], 'w')); print({k: len(v) for k, v in out.items()})
