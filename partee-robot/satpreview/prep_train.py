# prep_train.py : download the satellite photo for each training course (checked courses) into the cache satonly.py reads.
import json, math, os, sys, urllib.request, numpy as np, cv2
from shapely.geometry import LineString
from shapely.ops import unary_union
MPP = .3; root = sys.argv[1]
for k in os.environ.get('TRAIN', 'county-longford birr tullamore mount-juliet castle elmgreen').split():
    cache = f'/tmp/claude-0/osm/sat/{k}-0.3.png'
    if os.path.exists(cache): continue
    d = json.load(open(f'{root}/partee-site/courses/{k}.json'))
    lat, lon = d['lat'], d['lon']
    x0, y0, x1, y1 = unary_union([LineString(h['pts']).buffer(110) for h in d['holes'] if len(h['pts']) > 1]).bounds
    c = math.cos(math.radians(lat)); ll = lambda x, y: (lon + x / (111320 * c), lat + y / 110574)
    merc = lambda lo, la: (lo * 20037508.34 / 180, math.log(math.tan(math.radians(90 + la) / 2)) * 6378137)
    W, H = int((x1 - x0) / MPP), int((y1 - y0) / MPP); im = np.zeros((H, W, 3), np.uint8); T = 1800
    for ty in range(0, H, T):
        for tx in range(0, W, T):
            w, hh = min(T, W - tx), min(T, H - ty)
            (a, b), (cc, dd) = merc(*ll(x0 + tx * MPP, y1 - (ty + hh) * MPP)), merc(*ll(x0 + (tx + w) * MPP, y1 - ty * MPP))
            u = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={a},{b},{cc},{dd}&bboxSR=3857&imageSR=3857&size={w},{hh}&format=jpg&f=image'
            im[ty:ty + hh, tx:tx + w] = cv2.resize(cv2.imdecode(np.frombuffer(urllib.request.urlopen(u, timeout=120).read(), np.uint8), cv2.IMREAD_COLOR), (w, hh))
    cv2.imwrite(cache, im); print('train photo', k, W, H, flush=True)
