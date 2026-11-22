# hole_input.py COURSE.json REF OUT.json : cut one hole (+ surroundings) out of a course JSON for the stamper
import json, sys
from shapely.geometry import LineString, Polygon, Point
d = json.load(open(sys.argv[1])); ref = sys.argv[2]
h = [h for h in d['holes'] if str(h['ref']) == ref][0]; L = LineString(h['pts']); zone = L.buffer(140)
feats = []
for k, c in d['feats']:
    if k == 'rough': continue
    try: p = Polygon(list(zip(c[0::2], c[1::2]))).buffer(0)
    except Exception: continue
    if p.is_empty or not p.intersects(zone) or (k == 'bunker' and p.area > 3000) or (k == 'road' and 2 * p.area / max(p.length, 1) > 12): continue
    q = p.intersection(zone)
    for g in getattr(q, 'geoms', [q]):
        if g.geom_type == 'Polygon' and g.area > .5: feats.append([k, [round(v, 2) for xy in list(g.exterior.coords)[:-1] for v in xy]])
crowns = [c[:3] + (['conifer'] if len(c) > 3 and c[3] == 'conifer' else ['gorse'] if d.get('links') else ['round']) for c in d.get('crowns', []) if zone.contains(Point(c[0], c[1]))]
end = Point(h['pts'][-1]); greens = [Polygon(list(zip(c[0::2], c[1::2]))) for k, c in feats if k == 'green']
pin = min(greens, key=lambda g: g.distance(end)).centroid if greens else end
import math
others = [LineString(o['pts']) for o in d['holes'] if o is not h and len(o['pts']) > 1]
own = []
for k, c in feats:
    if k in ('fairway', 'green', 'tee'):
        p = Polygon(list(zip(c[0::2], c[1::2])))
        if p.distance(L) <= min([p.distance(o) for o in others] + [1e9]) + 2: own.append(c)
(ax, ay), (bx, by) = h['pts'][0], h['pts'][-1]; rot = math.pi / 2 - math.atan2(by - ay, bx - ax)   # tee bottom, green top
cr_, sr_ = math.cos(rot), math.sin(rot)
R2 = lambda x, y: (round(x * cr_ - y * sr_, 2), round(x * sr_ + y * cr_, 2))
def rc(c): return [v for i in range(0, len(c), 2) for v in R2(c[i], c[i + 1])]
feats = [[k, rc(c)] for k, c in feats]; own = [rc(c) for c in own]
crowns = [list(R2(c[0], c[1])) + c[2:] for c in crowns]; h = dict(h, pts=[list(R2(*p)) for p in h['pts']]); pin = Point(*R2(pin.x, pin.y))
out = dict(rot=rot, own=own, key=d['key'], name=d['name'], place=d.get('place', ''), holes=[h], feats=feats, crowns=crowns, pins=[[pin.x, pin.y]], elev=d.get('elev'))
if out['elev'] is None: out.pop('elev')
from shapely.ops import unary_union as _uu
_z = _uu([LineString(hh['pts']).buffer(110) for hh in d['holes'] if len(hh['pts']) > 1]).bounds
out['sat'] = {'path': f"/tmp/claude-0/osm/sat/{d['key']}-0.3.png", 'x0': _z[0], 'y1': _z[3], 'mpp': .3}
json.dump(out, open(sys.argv[3], 'w')); print(len(feats), 'feats', len(crowns), 'crowns')
