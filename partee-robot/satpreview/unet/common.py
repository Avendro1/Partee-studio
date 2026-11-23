# shared: satellite download + label rasterising for the fairway/green model
import json, math, os, urllib.request, numpy as np, cv2
from concurrent.futures import ThreadPoolExecutor
MPP = float(os.environ.get('UMPP', '0.5'))
CL = ['rough', 'fairway', 'green', 'bunker', 'water', 'trees', 'tee', 'built']
def merc(lo, la): return lo * 20037508.34 / 180, math.log(math.tan(math.radians(90 + la) / 2)) * 6378137
def photo(lat, lon, x0, y0, x1, y1, mpp=MPP):
    c = math.cos(math.radians(lat)); ll = lambda x, y: (lon + x / (111320 * c), lat + y / 110574)
    W, H = int((x1 - x0) / mpp), int((y1 - y0) / mpp); im = np.zeros((H, W, 3), np.uint8); T = 1500; jobs = []
    for ty in range(0, H, T):
        for tx in range(0, W, T): jobs.append((tx, ty, min(T, W - tx), min(T, H - ty)))
    def get(j):
        tx, ty, w, hh = j
        (a, b), (cc, dd) = merc(*ll(x0 + tx * mpp, y1 - (ty + hh) * mpp)), merc(*ll(x0 + (tx + w) * mpp, y1 - ty * mpp))
        u = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={a},{b},{cc},{dd}&bboxSR=3857&imageSR=3857&size={w},{hh}&format=jpg&f=image'
        for i in range(4):
            try: return j, cv2.resize(cv2.imdecode(np.frombuffer(urllib.request.urlopen(u, timeout=120).read(), np.uint8), cv2.IMREAD_COLOR), (w, hh))
            except Exception: pass
        raise RuntimeError('photo')
    with ThreadPoolExecutor(6) as ex:
        for (tx, ty, w, hh), t in ex.map(get, jobs): im[ty:ty + hh, tx:tx + w] = t
    return im
def rast(polys, x0, y1, H, W, mpp=MPP, ero=0):
    m = np.zeros((H, W), np.uint8)
    for q in polys:
        q = np.asarray(q, float)
        if len(q) < 3: continue
        cv2.fillPoly(m, [np.stack([(q[:, 0] - x0) / mpp, (y1 - q[:, 1]) / mpp], 1).round().astype(np.int32)], 1)
    if ero > 0: m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ero / mpp) * 2 + 1,) * 2))
    return m > 0
