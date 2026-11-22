# trees2d.py : procedural top-down tree sprites (lit from NW), RGBA float32
import numpy as np, math
from scipy import ndimage as ndi
L = np.array([-.55, -.55, .63]); L /= np.linalg.norm(L)   # light from top-left (x right, y down), z up
def _leafnoise(S, r, sc):
    n = ndi.gaussian_filter(r.standard_normal((S, S)), sc) ; return n / (n.std() + 1e-6)
def deciduous(seed, S=256, base=(.24, .40, .13), n=None, lumpy=.22):
    r = np.random.default_rng(seed); R = S * .40; c = S / 2
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    H = np.full((S, S), -1e9, np.float32); NX = np.zeros((S, S), np.float32); NY = np.zeros_like(NX); NZ = np.ones_like(NX); ID = np.zeros_like(NX)
    n = n or r.integers(45, 80)
    for i in range(n):
        a = r.uniform(0, 2 * math.pi); rad = R * math.sqrt(r.uniform(0, 1)) * .78
        cx, cy = c + rad * math.cos(a), c + rad * math.sin(a); cr = R * r.uniform(.16, .30) * (1.1 - .3 * rad / R)
        cz = (1 - (rad / R) ** 2) * R * .55
        dx, dy = xx - cx, yy - cy; d2 = dx * dx + dy * dy; inside = d2 < cr * cr
        z = np.sqrt(np.clip(cr * cr - d2, 0, None)) + cz
        k = inside & (z > H); H[k] = z[k]; NX[k] = dx[k] / cr; NY[k] = dy[k] / cr; NZ[k] = np.sqrt(np.clip(1 - (d2[k] / cr / cr), 0, 1)); ID[k] = i
    alpha = (H > -1e8).astype(np.float32)
    # leaf micro-normals
    ln1 = _leafnoise(S, r, .9); ln2 = _leafnoise(S, r, 2.2)
    gx = ndi.sobel(ln1, 1) * .08 + ndi.sobel(ln2, 1) * .05; gy = ndi.sobel(ln1, 0) * .08 + ndi.sobel(ln2, 0) * .05
    nx, ny, nz = NX + gx, NY + gy, NZ + .4; nn = np.sqrt(nx * nx + ny * ny + nz * nz)
    lam = np.clip((nx * L[0] + ny * L[1] + nz * L[2]) / nn, 0, 1)
    # self-occlusion: heights relative to neighbourhood
    Hn = np.where(alpha > 0, H, 0); ao = np.clip((Hn - ndi.uniform_filter(Hn, int(S * .06))) / (R * .12) + .75, .35, 1.1)
    shade = (.18 + .95 * lam ** 1.25) * ao * (1 + .10 * ln1 * (1 + ln2 * .3))
    hue = np.array(base, np.float32) * r.uniform(.88, 1.1, 3)
    # per-cluster hue variety
    var = 1 + lumpy * .25 * np.sin(ID * 2.3 + seed)[..., None] * np.array([.6, .3, .9])
    rgb = hue[None, None] * shade[..., None] * var
    hi = np.clip(lam - .82, 0, 1)[..., None] * np.array([.45, .5, .25]) * 1.2; rgb = rgb + hi
    # ragged leafy edge
    edge = ndi.gaussian_filter(alpha, 1.2); rag = np.clip((edge - .5) * 6 + ln1 * .9, 0, 1) * alpha
    return np.dstack([np.clip(rgb, 0, 1), rag]).astype(np.float32)
def conifer_old(seed, S=256):
    r = np.random.default_rng(seed); c = S / 2; yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) - c
    rr = np.hypot(xx, yy) / (S * .40); th = np.arctan2(yy, xx); n = r.integers(16, 24)
    al = np.zeros((S, S), np.float32); sh = np.zeros_like(al)
    for layer in range(4):                                   # tiers of radial branches, smaller toward top
        sc = 1 - layer * .22; ph = r.uniform(0, 6); k = n - layer * 3
        prof = sc * (.72 + .28 * np.abs(np.cos(k / 2 * th + ph)) ** 1.5)
        m = np.clip((prof - rr) * 14, 0, 1)
        # branch shading: lit on NW side of each spray
        lit = .5 + .5 * np.cos(th - math.radians(225)) * rr / sc
        s_ = (.45 + .45 * layer / 3 + .25 * lit) * (1 - .35 * (rr / sc) ** 2) * (.85 + .3 * np.abs(np.sin(k / 2 * th + ph)))
        sh = sh * (1 - m) + s_ * m; al = np.maximum(al, m)
    tex = 1 + .15 * ndi.gaussian_filter(r.standard_normal((S, S)), .8)
    base = np.array([.13, .25, .15]) * r.uniform(.85, 1.15)
    return np.dstack([np.clip(base[None, None] * (sh * tex)[..., None] * 1.3, 0, 1), al]).astype(np.float32)
def conifer(seed, S=256):
    t = deciduous(500 + seed, S, base=(.11, .22, .14), n=140, lumpy=.1)
    r = np.random.default_rng(seed); c = S / 2; yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) - c
    th = np.arctan2(yy, xx); rr = np.hypot(xx, yy) / (S * .40)
    streak = ndi.gaussian_filter(r.standard_normal(720), 1.5); streak /= streak.std()
    st = streak[((th + np.pi) / (2 * np.pi) * 719).astype(int)]
    t[..., :3] *= np.clip(1 + .22 * st * np.clip(rr, .2, 1), .6, 1.4)[..., None]
    t[..., 3] *= np.clip((1.02 + .08 * st - rr) * 10, 0, 1)   # spiky silhouette
    return t
def gorse(seed, S=256):   # links gorse/scrub: low, dense dark-olive bush with yellow flower specks, flatter shading
    t = deciduous(900 + seed, S, base=(.17, .25, .08), n=170, lumpy=.35)
    r = np.random.default_rng(seed); a = t[..., 3]
    fl = (ndi.gaussian_filter(r.standard_normal((S, S)), 1.2) > 1.9) & (a > .5)
    t[..., :3] = np.where(fl[..., None], np.array([.85, .72, .18]) * (.7 + .3 * t[..., 1:2] / max(1e-3, t[..., 1].max())), t[..., :3])
    t[..., :3] = .55 * t[..., :3] + .45 * t[..., :3].mean(axis=(0, 1), keepdims=True) * (a[..., None] > 0)   # less 3-D, reads as low scrub
    return t
PAL = [(.31, .50, .15), (.38, .55, .16), (.27, .46, .16), (.42, .57, .19), (.33, .48, .14), (.29, .44, .12), (.46, .58, .20)]
def library():
    lib = {'round': [deciduous(i, base=PAL[i % len(PAL)]) for i in range(14)],
           'big': [deciduous(100 + i, base=PAL[(i + 3) % len(PAL)], n=95) for i in range(8)],
           'conifer': [conifer(i) for i in range(8)], 'gorse': [gorse(i) for i in range(8)]}
    return lib
if __name__ == '__main__':
    from PIL import Image
    lib = library(); tiles = [t for k in ('round', 'big', 'conifer') for t in lib[k][:6]]
    sheet = np.ones((2 * 260, 9 * 260, 3), np.float32) * np.array([.42, .55, .25])
    for i, t in enumerate(tiles):
        y, x = divmod(i, 9); a = t[..., 3:]; sl = sheet[y * 260:y * 260 + 256, x * 260:x * 260 + 256]; sl[:] = sl * (1 - a) + t[..., :3] * a
    Image.fromarray((sheet * 255).astype(np.uint8)).save('/tmp/claude-0/osm/stamp/lib2.jpg')
