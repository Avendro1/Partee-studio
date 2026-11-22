# osm_buildings.py KEY : adds every OSM building (clubhouse, hotel, sheds...) near the holes to the course JSON and teacher,
# including ones outside the golf-course boundary that build_course.py leaves out.
import sys, json, math, os, urllib.request, xml.etree.ElementTree as ET
from shapely.geometry import Polygon
key = sys.argv[1]; R = '/home/user/avendro-web-v2'; CJ = f'{R}/partee-site/courses/{key}.json'; d = json.load(open(CJ))
c = math.cos(math.radians(d['lat'])); xs = [p[0] for h in d['holes'] for p in h['pts']]; ys = [p[1] for h in d['holes'] for p in h['pts']]
lo = lambda x: d['lon'] + x / (111320 * c); la = lambda y: d['lat'] + y / 110574
bb = (lo(min(xs) - 150), la(min(ys) - 150), lo(max(xs) + 150), la(max(ys) + 150))
u = f'https://api.openstreetmap.org/api/0.6/map?bbox={bb[0]:.5f},{bb[1]:.5f},{bb[2]:.5f},{bb[3]:.5f}'
x = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'partee-course-builder/1.0'}), timeout=120).read()
root = ET.fromstring(x); N = {n.get('id'): (float(n.get('lon')), float(n.get('lat'))) for n in root.iter('node')}
loc = lambda p: ((p[0] - d['lon']) * 111320 * c, (p[1] - d['lat']) * 110574)
new = []; coast = set()
for e in list(root.iter('way')) + list(root.iter('relation')):   # links test: dunes/beach/coastline next to the course
    t = {q.get('k'): q.get('v') for q in e.iter('tag')}
    if t.get('natural') in ('coastline', 'beach', 'dune'): coast.add(t['natural'])
    if t.get('golf:course') == 'links' or t.get('golf_course') == 'links': coast.add('links')
sandy = any(f[0] == 'sand' for f in d['feats'])
PARKLAND = {'courtown', 'deer-park', 'tara-glen', 'galway', 'woodbrook', 'st-helen-s-bay'}   # coastal but tree-lined parkland: checked by eye 22 Nov
links = key not in PARKLAND and bool(coast & {'dune', 'links'}) or (key not in PARKLAND and bool(coast & {'coastline', 'beach'}) and sandy)
for w in root.iter('way'):
    tags = {t.get('k'): t.get('v') for t in w.iter('tag')}
    hw = tags.get('highway'); RW = {'motorway': 9, 'trunk': 8, 'primary': 7.5, 'secondary': 7, 'tertiary': 6.5, 'unclassified': 5.5, 'residential': 5.5, 'service': 4, 'living_street': 5}
    if hw in RW and tags.get('area') != 'yes':
        pts = [loc(N[n.get('ref')]) for n in w.iter('nd') if n.get('ref') in N]
        if len(pts) >= 2:
            from shapely.geometry import LineString
            P = LineString(pts).buffer(RW[hw] / 2, cap_style=2, join_style=2)
            new.append(('road', [round(v, 1) for q in list(P.exterior.coords)[:-1] for v in q]))
        continue
    kind = 'building' if 'building' in tags else 'parking' if tags.get('amenity') == 'parking' and w.find('nd') is not None else None
    if not kind: continue
    pts = [loc(N[n.get('ref')]) for n in w.iter('nd') if n.get('ref') in N]
    if len(pts) < 4: continue
    P = Polygon(pts).buffer(0)
    if P.area < 8: continue
    new.append((kind, [round(v, 1) for q in list(P.exterior.coords)[:-1] for v in q]))
def add(path):
    if not os.path.exists(path): return 0
    j = json.load(open(path))
    j['coastal_links'] = links
    j['feats'] = [f for f in j['feats'] if f[0] not in ('building', 'parking', 'road')] + [[k, b] for k, b in new]
    s = json.dumps(j); open(path, 'w').write(s); return len(new)
print('coast', sorted(coast), 'links', links)
print('buildings', add(CJ), add(f'/tmp/claude-0/osm/teacher/{key}.json'))
