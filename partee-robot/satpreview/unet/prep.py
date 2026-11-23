# prep.py OUT : satellite photo + label mask for every checked course -> OUT/<key>.jpg, OUT/<key>.png
import sys, os, json, glob, numpy as np, cv2
from shapely.geometry import LineString
from shapely.ops import unary_union
sys.path.insert(0, os.path.dirname(__file__)); from common import *
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
SKIP = set(os.environ.get('SKIP', '').split())
root = os.path.join(os.path.dirname(__file__), '../../../partee-site/courses')
for f in sorted(glob.glob(root + '/*.json')):
    d = json.load(open(f)); k = d['key']
    if k in SKIP or 'sat_release' in d or os.path.exists(f'{out}/{k}.png'): continue
    try:
        L = [LineString(h['pts']) for h in d['holes'] if len(h['pts']) > 1]
        x0, y0, x1, y1 = unary_union([l.buffer(110) for l in L]).bounds
        im = photo(d['lat'], d['lon'], x0, y0, x1, y1); H, W = im.shape[:2]
        F = {}
        for t, c in d['feats']: F.setdefault(t, []).append(np.array(c, float).reshape(-1, 2))
        lab = np.full((H, W), 255, np.uint8)
        near = rast([np.array(l.buffer(150).exterior.coords) for l in L], x0, y1, H, W); lab[near] = 0
        def put(ts, c, ero=0): lab[rast(sum([F.get(t, []) for t in ts], []), x0, y1, H, W, ero=ero)] = c
        put(['building', 'road', 'parking', 'path'], 7)
        put(['wood'], 5)
        tr = np.zeros((H, W), np.uint8)
        for cr in d.get('crowns', []): cv2.circle(tr, (int((cr[0] - x0) / MPP), int((y1 - cr[1]) / MPP)), max(1, int(cr[2] * .7 / MPP)), 1, -1)
        lab[(tr > 0) & (lab != 7)] = 5
        put(['fairway'], 1); put(['tee'], 6); put(['green'], 2)
        lab[rast([w for w in F.get('water', []) if len(w) > 3], x0, y1, H, W, ero=1)] = 4
        put(['bunker'], 3)
        # fuzzy edges: ignore a 1 px band around every class border
        e = cv2.morphologyEx(lab, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0; lab[e] = 255
        cv2.imwrite(f'{out}/{k}.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 92]); cv2.imwrite(f'{out}/{k}.png', lab); print('ok', k, W, H, flush=True)
    except Exception as ex: print('fail', k, ex, flush=True)
