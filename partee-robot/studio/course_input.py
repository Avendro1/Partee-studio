# course_input.py COURSE.json OUT.json : whole course for the stamper (north up, every hole's own features)
import json, sys
from shapely.geometry import LineString, Polygon, Point
d = json.load(open(sys.argv[1])); _fat = lambda c: (lambda q: 2 * q.area / max(q.length, 1) > 12)(Polygon(list(zip(c[0::2], c[1::2]))).buffer(0))   # loop road filled in as a slab
_A = lambda c: Polygon(list(zip(c[0::2], c[1::2]))).buffer(0).area
feats = [[k, c] for k, c in d['feats'] if k != 'rough' and not (k == 'bunker' and _A(c) > 3000) and not (k == 'road' and _fat(c))]   # no real bunker is a field-sized patch
own = [c for k, c in feats if k in ('fairway', 'green', 'tee')]
greens = [Polygon(list(zip(c[0::2], c[1::2]))) for k, c in feats if k == 'green']
pins = []
for h in d['holes']:
    if len(h['pts']) < 2 or not greens: continue
    e = Point(h['pts'][-1]); g = min(greens, key=lambda g: g.distance(e))
    if g.distance(e) < 40: pins.append([g.centroid.x, g.centroid.y])
crowns = [c[:3] + ['gorse' if d.get('links') else 'round'] for c in d.get('crowns', [])]
out = dict(key=d['key'], name=d['name'], place=d.get('place', ''), holes=d['holes'], feats=feats, crowns=crowns, pins=pins, own=own)
if d.get('elev'): out['elev'] = d['elev']
out['links'] = bool(d.get('links'))
from shapely.ops import unary_union as _uu
_z = _uu([LineString(hh['pts']).buffer(110) for hh in d['holes'] if len(hh['pts']) > 1]).bounds
out['sat'] = {'path': f"/tmp/claude-0/osm/sat/{d['key']}-0.3.png", 'x0': _z[0], 'y1': _z[3], 'mpp': .3}
json.dump(out, open(sys.argv[2], 'w')); print(len(feats), 'feats', len(crowns), 'crowns', len(pins), 'pins')
