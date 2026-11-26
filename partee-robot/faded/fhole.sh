#!/bin/bash
# fhole.sh KEY REF OUTDIR [SIZE] [PPM] : one hole in the 4 faded styles (same recipe as the whole-course faded.sh, 25 Nov approved look).
# island has no flag: the designer + robot draw flag and shots on top using OUTDIR/KEY-REF-STYLE.t.json (island px -> poster px) and KEY-REF.geo.json (world -> island px)
set -o pipefail; K=$1; REF=$2; O=$3; SIZE=${4:-1800x2400}; PPM=${5:-3.0}; D=$(cd $(dirname $0); pwd); SIG=$D/sig
I=${INPUTS:-$D/inputs}/$K; C=$D/courses/$K.json; T=$O/tmp-$K-$REF; mkdir -p $T; cd $D
[ -n "$ALL" ] && [ -f "$ALL" ] && cp $ALL $T/all.json
[ -f $T/all.json ] || GATE_REPLAY=1 python3 course_input.py $C $T/all.json > /dev/null || exit 1
GATE_REPLAY=1 python3 hole_input.py $T/all.json $REF $T/h.json > /dev/null || { echo "FAIL hole_input $K $REF"; exit 1; }
python3 - $T/h.json $C <<'PY'
import json, sys; h = json.load(open(sys.argv[1])); c = json.load(open(sys.argv[2]))
h['pin0'] = h['pins']; h['pins'] = []; h['lat'], h['lon'] = c['lat'], c['lon']; json.dump(h, open(sys.argv[1], 'w'))
PY
GATE_REPLAY=1 ALPHA=1 NUMS=0 LIFT=0 python3 stamper.py $T/h.json $T/h.jpg $PPM > $T/st.log 2>&1 || { echo "FAIL stamper $K $REF"; tail -3 $T/st.log; exit 1; }
python3 - $T/h.json $T/h-frame.json $PPM $O/$K-$REF.geo.json <<'PY'
import json, sys, math; h = json.load(open(sys.argv[1])); f = json.load(open(sys.argv[2])); P = float(sys.argv[3])
r = h['rot']; cr, sr = math.cos(r), math.sin(r); wx, wy = h['holes'][0]['pts'][0]; tx, ty = f['tee_px'][0]
# world (course json, metres) -> rotated -> stamper canvas -> island png px
a, b = P * cr, -P * sr; cx = tx - P * wx - f['x0']
d_, e_ = -P * sr, -P * cr; cy = ty + P * wy - f['y0']
json.dump({'m': [a, b, cx, d_, e_, cy], 'ppm': P, 'pin': h['pin0'][0] if h['pin0'] else None, 'rot': r}, open(sys.argv[4], 'w'))
PY
E=$(python3 -c "print(int(380*$PPM))"); RP=$(python3 -c "print(int(300*$PPM))")
ST=${STYLES:-detailed architectural charcoal midnight}
for st in "ivory 244,241,233 detailed" "charcoal 34,37,32 charcoal" "midnight 70,76,60 midnight"; do set -- $st
  case " $ST " in *" $3 "*) ;; *) continue;; esac
  HALO=2 PAINT_S=12 SURR_PAPER=$2 python3 $SIG/surr2.py $T/h-island.png $T/h-frame.json $T/h.json $I/ctx.json ${LIDAR:-$I/lidar.png} $T/surr.png $E $PPM $RP > $T/surr.log 2>&1 || { echo "FAIL surr $K $REF $1"; continue; }
  TITLE="Hole $REF" SIZE=$SIZE SURR=$T/surr.png SURR_E=$E SURR_SHRINK=.9 TDUMP=$O/$K-$REF-$3.t.json MSG="$MSG" python3 $SIG/premium.py $T/h-island.png $T/h.json $1 $O/$K-$REF-$3.jpg port > $T/prem.log 2>&1 || echo "FAIL premium $K $REF $1"
done
rm -f $T/surr.png
LE=""; [ -f ${LIDAR:-$I/lidar.png} ] && LE="LIDAR=${LIDAR:-$I/lidar.png}"
case " $ST " in *" architectural "*) env HOLE=$REF NOPLAY=1 SIZE=$SIZE CTX=$I/ctx.json SURRCTX=$I/ctx.json SURR_R=.13 $LE TDUMP=$O/$K-$REF-architectural.t.json MSG="$MSG" python3 $SIG/plan.py $T/all.json $C $O/$K-$REF-architectural.jpg > $T/plan.log 2>&1 || { echo "FAIL plan $K $REF"; tail -3 $T/plan.log; };; esac
# one world (course metres) -> poster px affine per style, plus the pin (centre of the green nearest the hole's end)
python3 - $O $K $REF <<'PY'
import json, sys, os; O, K, R = sys.argv[1:4]; g = json.load(open(f'{O}/{K}-{R}.geo.json')); a = g['m']; out = {}
for st in ('detailed', 'charcoal', 'midnight', 'architectural'):
    p = f'{O}/{K}-{R}-{st}.t.json'
    if not os.path.exists(p): continue
    t = json.load(open(p))
    if st == 'architectural': out[st] = {'m': [round(v, 5) for v in t['m']], 'W': t['W'], 'H': t['H']}
    else:
        s = t['s']; bx = (t['offx']) * s + t['ox']; by = (t['offy']) * s + t['oy']
        out[st] = {'m': [round(v, 5) for v in (a[0] * s, a[1] * s, a[2] * s + bx, a[3] * s, a[4] * s, a[5] * s + by)], 'W': t['W'], 'H': t['H']}
    os.remove(p)
# pin back to world coords (hole files are rotated): R2 inverse
import math; r = g['rot']; cr, sr = math.cos(r), math.sin(r); pin = g['pin']
if pin: pin = [round(pin[0] * cr + pin[1] * sr, 2), round(-pin[0] * sr + pin[1] * cr, 2)]
json.dump({'t': out, 'pin': pin}, open(f'{O}/{K}-{R}.json', 'w')); os.remove(f'{O}/{K}-{R}.geo.json')
PY
[ -z "$KEEPTMP" ] && rm -rf $T
echo "done $K $REF"
