#!/bin/bash
# estate.sh KEY OUT : Estate style (27 Nov, engraved deep green on ivory) for a full course + every hole at 1800x2400.
# Drawn by heritage/estate.py from the traced vectors (plan.py GEODUMP), same satellite tree check as Heritage.
# OUT/f/KEY-estate.{webp,-s.webp}  OUT/fh/KEY-REF-estate.webp + .t.json
set -o pipefail; K=$1; OUT=$2; D=$(cd $(dirname $0); pwd); SIG=$D/sig; C=$D/courses/$K.json; I=$D/inputs/$K; T=$OUT/tmp/$K; mkdir -p $OUT/f $OUT/fh $T $I; cd $D
[ -f $I/ctx.json ] || python3 $SIG/ctx_fetch2.py $C $I/ctx.json 700 > $T/ctx.log 2>&1 || { echo "FAIL ctx $K"; exit 1; }
[ -f $I/lidar.png ] || python3 $SIG/lidar_fetch.py $C $I/lidar.png > $T/lidar.log 2>&1 || rm -f $I/lidar.png
GATE_REPLAY=1 python3 course_input.py $C $T/all.json > /dev/null || { echo "FAIL course_input $K"; exit 1; }
TF=$D/heritage/$K-treefix.json
if [ ! -f $TF ]; then
  for i in 1 2 3 4; do python3 $D/heritage/canopy.py $C $T/canopy.json $T/sat > $T/can.log 2>&1 && break; rm -rf $T/sat; sleep $((i * 20)); done
  [ -f $T/sat/sat.jpg ] && python3 $D/heritage/fixtrees.py $C $T/all.json $T/sat $T/treefix.json > $T/fix.log 2>&1 && TF=$T/treefix.json || { echo "no treefix $K"; TF=; }
fi
LE=""; [ -f $I/lidar.png ] && LE="LIDAR=$I/lidar.png"; TE=""; [ -n "$TF" ] && TE="TREEFIX=$TF"
E_="PLAN_STYLE=heritage STRIP_Z=55 $TE SIZE=1800x2400 CTX=$I/ctx.json SURRCTX=$I/ctx.json SURR_R=.13 $LE"
env $E_ GEODUMP=$T/geo.json python3 $SIG/plan.py $T/all.json $C $T/fin.jpg > $T/plan.log 2>&1 && env $E_ NOPLAY=2 python3 $SIG/plan.py $T/all.json $C $T/np.jpg > $T/plan2.log 2>&1 \
  && python3 $D/heritage/estate.py $T/geo.json $T/fin.jpg $T/np.jpg $OUT/f/$K-estate.jpg > /dev/null || { echo "FAIL course $K"; tail -3 $T/plan.log $T/plan2.log; }
for R in $(python3 -c "import json; d=json.load(open('$C')); print('' if d.get('preview') else ' '.join(str(h['ref']) for h in d['holes'] if len(h.get('pts',[]))>1))"); do
  env $E_ HOLE=$R NOPLAY=1 GEODUMP=$T/h.json TDUMP=$OUT/fh/$K-$R-estate.t.json python3 $SIG/plan.py $T/all.json $C $T/h.jpg > $T/h.log 2>&1 \
    && python3 $D/heritage/estate.py $T/h.json $T/h.jpg $T/h.jpg $OUT/fh/$K-$R-estate.jpg > /dev/null || { echo "FAIL hole $K $R"; tail -2 $T/h.log; }
done
python3 - $OUT $K <<'PY'
import glob, os, sys; from PIL import Image
O, K = sys.argv[1:3]
for p in glob.glob(f'{O}/fh/{K}-*.jpg'): Image.open(p).convert('RGB').save(p[:-4] + '.webp', quality=80, method=5); os.remove(p)
for p in glob.glob(f'{O}/f/{K}-*.jpg'):
    im = Image.open(p).convert('RGB'); w, h = im.size; im.save(p[:-4] + '.webp', quality=86, method=5)
    im.resize((1200, 1200 * h // w) if w > h else (1200 * w // h, 1200), Image.LANCZOS).save(p[:-4] + '-s.webp', quality=84, method=5)
PY
rm -rf $T $OUT/tmp; echo "estate done $K $(ls $OUT/fh | grep -c webp) hole images"
