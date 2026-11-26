# lidar_fetch.py COURSE.json OUT.png [service] : real LiDAR hillshade (GSI/OPW open data, CC BY 4.0, credit "GSI & OPW")
# around the course at 1 m/px + OUT.json bbox in the course's local metres (for plan.py LIDAR=OUT.png).
import json, math, subprocess, sys, os, numpy as np
from PIL import Image
d = json.load(open(sys.argv[1])); out = sys.argv[2]
svc = sys.argv[3] if len(sys.argv) > 3 else 'IE_GSI_LiDAR_DSM_HS_OPW_NASC_IE26_2m_ITM_MH_TIFF'
lat0, lon0 = d['lat'], d['lon']; kx = 111320 * math.cos(math.radians(lat0)); ky = 110574
xs = [p[0] for h in d['holes'] for p in h['pts']]; ys = [p[1] for h in d['holes'] for p in h['pts']]
x0, x1, y0, y1 = min(xs) - 400, max(xs) + 400, min(ys) - 400, max(ys) + 400
merc = lambda lon, lat: (lon * 20037508.34 / 180, math.log(math.tan((90 + lat) * math.pi / 360)) * 20037508.34 / math.pi)
mx0, my0 = merc(lon0 + x0 / kx, lat0 + y0 / ky); mx1, my1 = merc(lon0 + x1 / kx, lat0 + y1 / ky)
W, H = int(x1 - x0), int(y1 - y0); A = np.zeros((H, W), np.uint8); TS = 2000; tmp = out + '.part.png'
for i in range(0, H, TS):
    for j in range(0, W, TS):
        w, h = min(TS, W - j), min(TS, H - i)
        ax0 = mx0 + (mx1 - mx0) * j / W; ax1 = mx0 + (mx1 - mx0) * (j + w) / W; ay1 = my1 - (my1 - my0) * i / H; ay0 = my1 - (my1 - my0) * (i + h) / H
        u = (f'https://gsi.geodata.gov.ie/imagehost/rest/services/Lidar/{svc}/ImageServer/exportImage?bbox={ax0},{ay0},{ax1},{ay1}'
             f'&bboxSR=3857&imageSR=3857&size={w},{h}&format=png&interpolation=RSP_BilinearInterpolation&f=image')
        subprocess.run(['curl', '-s', '-m', '180', '-o', tmp, u], check=True)
        A[i:i + h, j:j + w] = np.asarray(Image.open(tmp).convert('L').resize((w, h)))
os.remove(tmp); Image.fromarray(A).save(out)
json.dump({'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'w': W, 'h': H, 'svc': svc}, open(out.rsplit('.', 1)[0] + '.json', 'w')); print(out, W, H)
