# holefind.py : work out which tee + green make each hole, in order, from the published scorecard (par + length per hole).
# Candidates = tee and green shapes (from OSM or found on the photo). The scorecard turns the guess into a check:
# hole i must run from a tee near the last green to a green whose distance fits the card length.
import json, sys, math
from shapely.geometry import Polygon, Point

def cands(d):
    G, T = [], []
    from shapely.ops import unary_union
    from shapely.prepared import prep
    fw = [Polygon(list(zip(c[0::2], c[1::2]))).buffer(0) for k, c in d['feats'] if k == 'fairway']
    global FW, BAD, FWL
    FWL = fw
    FW = prep(unary_union(fw).buffer(25)) if fw else None
    bad = [Polygon(list(zip(c[0::2], c[1::2]))).buffer(0) for k, c in d['feats'] if k in ('building', 'water')]
    BAD = prep(unary_union(bad)) if bad else None
    for k, c in d['feats']:
        if k not in ('green', 'tee'): continue
        p = Polygon(list(zip(c[0::2], c[1::2]))).buffer(0)
        if p.is_empty: continue
        mr = p.minimum_rotated_rectangle; xy = list(mr.exterior.coords)
        e1 = (xy[1][0] - xy[0][0], xy[1][1] - xy[0][1]); e2 = (xy[2][0] - xy[1][0], xy[2][1] - xy[1][1])
        ax = e1 if math.hypot(*e1) >= math.hypot(*e2) else e2; el = max(math.hypot(*e1), 1e-6) / max(math.hypot(*e2), 1e-6)
        el = max(el, 1 / el)   # tee boxes are long rectangles pointing down the hole
        (G if k == 'green' else T).append((p.centroid.x, p.centroid.y, p.area, math.atan2(ax[1], ax[0]), el))
    return G, T

def ratio_cost(dist, L, par):
    r = dist / max(L, 1)   # straight tee->green distance vs card length (doglegs make the card longer)
    lo = .93 if par == 3 else (.78 if par == 4 else .68)
    if r > 1.06: return (r - 1.06) * 40
    if r < lo: return (lo - r) * 25
    return abs(r - .98) * (12 if par == 3 else 6)

def line_cost(t, g, par):
    c = 0
    if False and t[4] > 1.6:   # tee box direction should match the hole
        b = math.atan2(g[1] - t[1], g[0] - t[0]); dd = abs((b - t[3] + math.pi / 2) % math.pi - math.pi / 2)
        c += max(0, dd - .25) * 3
    pts = [Point(t[0] + (g[0] - t[0]) * f, t[1] + (g[1] - t[1]) * f) for f in (.2, .35, .5, .65, .8)]
    if FWL and par > 3:   # a real par 4/5 has one fairway running from near the tee to near the green (doglegs allowed)
        T_, G_ = Point(t[:2]), Point(g[:2])
        if not any(f.distance(T_) < 110 and f.distance(G_) < 60 for f in FWL): c += 3
    return c

def solve(G, T, card, start=None, beam=int(__import__("os").environ.get("BEAM", "3000"))):
    """card: [(par, metres)] in hole order. Returns [(tee_xy, green_xy)] and cost."""
    D = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    states = [(0.0, [], frozenset(), frozenset(), start)]   # cost, holes, greens used, tees used, last green
    for i, (par, L) in enumerate(card):
        nxt = []
        for cost, path, used, usedt, here in states:
            tees = range(len(T)) if here is None else sorted(range(len(T)), key=lambda j: D(T[j], here))[:12]
            for ti in tees:
                t = T[ti]
                if ti in usedt: continue
                walk = 0 if here is None else D(t, here)
                if walk > 350: continue
                for gi, g in enumerate(G):
                    if gi in used: continue
                    dist = D(t, g)
                    if dist < .5 * L or dist > 1.2 * L: continue
                    c = cost + ratio_cost(dist, L, par) + max(0, walk - 70) / 50 + line_cost(t, g, par)
                    nxt.append((c, path + [(t[:2], g[:2])], used | {gi}, usedt | {ti}, g))
        if not nxt: return None, 1e9
        nxt.sort(key=lambda s: s[0])
        seen, states = set(), []
        for s in nxt:
            key = (s[4][:2], s[2])
            if key in seen: continue
            seen.add(key); states.append(s)
            if len(states) >= beam: break
    best = states[0]; return best[1], best[0]

if __name__ == '__main__':
    d = json.load(open(sys.argv[1]))
    H = [h for h in d['holes'] if len(h['pts']) > 1]
    card = [(int(h.get('par') or 4), sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(h['pts'], h['pts'][1:]))) for h in H]
    G, T = cands(d); path, cost = solve(G, T, card)
    if path is None: print('no solution'); sys.exit()
    ok = sum(1 for (t, g), h in zip(path, H) if math.hypot(g[0] - h['pts'][-1][0], g[1] - h['pts'][-1][1]) < 25)
    print(d['key'], f'{ok}/{len(H)} greens right', round(cost, 1))
