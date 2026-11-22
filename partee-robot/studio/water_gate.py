# water_gate.py: keep a water polygon ONLY if the satellite clearly shows open water there (Shane 22 Nov: "make sure you are 100% sure").
# OSM 'water hazard' areas often sit under tree belts or are dry reedy hollows; dark tree shade is not water.
# Open water = smooth surface (low local texture) AND neutral/blue-grey dark tone. Per-course 'water_no' [[x,y],...] (polygon representative_point, matched within 1.5 m) drops ones rejected by eye; 'water_yes' keeps ones confirmed by eye.
import os, numpy as np, cv2
from shapely.geometry import Polygon, Point
def gate(feats, sat, water_no=(), water_yes=()):
    """feats: [[kind, flat coords]], sat: {'path','x0','y1','mpp'}. Returns (feats kept, report)."""
    if not sat or not os.path.exists(sat['path']): return feats, {'skipped': 'no photo'}   # never guess: without the photo there is no proof
    im = cv2.imread(sat['path']); mpp = sat['mpp']; x0, y1 = sat['x0'], sat['y1']
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); L = lab[..., 0] * 100 / 255; A = lab[..., 1] - 128; B = lab[..., 2] - 128
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32); m = cv2.blur(g, (7, 7)); sd = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - m * m, 0))
    ow = (sd < 6) & ((B < 4) | (L < 28)) & (A > -14)
    no = [Point(p) for p in water_no]; yes = [Point(p) for p in water_yes]; out = []; rep = []
    for k, c in feats:
        if k != 'water': out.append([k, c]); continue
        p = np.array(c).reshape(-1, 2); px = np.stack([(p[:, 0] - x0) / mpp, (y1 - p[:, 1]) / mpp], 1).astype(np.int32)
        mk = np.zeros(g.shape, np.uint8); cv2.fillPoly(mk, [px], 1); mk = cv2.erode(mk, np.ones((3, 3), np.uint8)) > 0
        P = Polygon(p).buffer(0); n = int(mk.sum()); rp = P.representative_point()   # stable id of this polygon for by-eye decisions
        s = dict(x=round(rp.x, 1), y=round(rp.y, 1), area=round(P.area), ow=round(float(ow[mk].mean()), 2) if n else 0,
                 sd=round(float(np.median(sd[mk])), 1) if n else 99, L=round(float(np.median(L[mk])), 1) if n else 99)
        s['keep'] = n >= 20 and s['ow'] >= .45 and s['sd'] < 6 and s['L'] < 22 and not any(rp.distance(q) < 1.5 for q in no)
        if any(rp.distance(q) < 1.5 for q in yes): s['keep'] = True   # open water confirmed by eye on the satellite
        rep.append(s)
        if s['keep']: out.append([k, c])
    return out, {'water': rep, 'kept': sum(r['keep'] for r in rep), 'dropped': sum(not r['keep'] for r in rep)}
def gate_course(d):
    """filter d['feats'] water using the course's cached tracing photo; writes the report next to the photo"""
    from shapely.geometry import LineString; from shapely.ops import unary_union; import json
    z = unary_union([LineString(h['pts']).buffer(110) for h in d['holes'] if len(h['pts']) > 1]).bounds
    sat = {'path': f"/tmp/claude-0/osm/sat/{d['key']}-0.3.png", 'x0': z[0], 'y1': z[3], 'mpp': .3}
    f, rep = gate(d['feats'], sat, d.get('water_no', []), d.get('water_yes', []))
    try: json.dump(rep, open(f"/tmp/claude-0/osm/sat/{d['key']}-water.json", 'w'))
    except OSError: pass
    return f
