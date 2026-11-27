#!/bin/bash
# heritage.sh KEY OUT : Heritage style (27 Nov, Shane's K Club reference) for a full course + every hole at 1800x2400, plus Architectural
# re-rendered with the same satellite tree check (island copses added, reed dots on lake banks dropped) and the tee-marker overlap fix.
# OUT/f/KEY-{heritage,architectural}.jpg  OUT/fh/KEY-REF-{heritage,architectural}.jpg + .t.json  OUT/treefix/KEY.json
set -o pipefail; K=$1; OUT=$2; D=$(cd $(dirname $0); pwd); SIG=$D/sig; C=$D/courses/$K.json; I=$D/inputs/$K; T=$OUT/tmp/$K; mkdir -p $OUT/f $OUT/fh $OUT/treefix $T $I; cd $D
[ -f $I/ctx.json ] || python3 $SIG/ctx_fetch2.py $C $I/ctx.json 700 > $T/ctx.log 2>&1 || { echo "FAIL ctx $K"; exit 1; }
[ -f $I/lidar.png ] || python3 $SIG/lidar_fetch.py $C $I/lidar.png > $T/lidar.log 2>&1 || rm -f $I/lidar.png
GATE_REPLAY=1 python3 course_input.py $C $T/all.json > /dev/null || { echo "FAIL course_input $K"; exit 1; }
TF=$D/heritage/$K-treefix.json
if [ ! -f $TF ]; then
  for i in 1 2 3 4; do python3 $D/heritage/canopy.py $C $T/canopy.json $T/sat > $T/can.log 2>&1 && break; rm -rf $T/sat; sleep $((i * 20)); done
  [ -f $T/sat/sat.jpg ] && python3 $D/heritage/fixtrees.py $C $T/all.json $T/sat $T/treefix.json > $T/fix.log 2>&1 && TF=$T/treefix.json || { echo "no treefix $K"; TF=; }
fi
[ -n "$TF" ] && cp $TF $OUT/treefix/$K.json && cat $T/fix.log 2>/dev/null
LE=""; [ -f $I/lidar.png ] && LE="LIDAR=$I/lidar.png"; TE=""; [ -n "$TF" ] && TE="TREEFIX=$TF"
for S in heritage architectural; do
  PS=""; [ $S = heritage ] && PS="PLAN_STYLE=heritage"
  env $PS $TE SIZE=1800x2400 CTX=$I/ctx.json SURRCTX=$I/ctx.json SURR_R=.13 $LE python3 $SIG/plan.py $T/all.json $C $OUT/f/$K-$S.jpg > $T/plan.log 2>&1 || { echo "FAIL plan $K $S"; tail -3 $T/plan.log; }
  for R in $(python3 -c "import json; d=json.load(open('$C')); print('' if d.get('preview') else ' '.join(str(h['ref']) for h in d['holes'] if len(h.get('pts',[]))>1))"); do
    env $PS $TE HOLE=$R NOPLAY=1 SIZE=1800x2400 CTX=$I/ctx.json SURRCTX=$I/ctx.json SURR_R=.13 $LE TDUMP=$OUT/fh/$K-$R-$S.t.json python3 $SIG/plan.py $T/all.json $C $OUT/fh/$K-$R-$S.jpg > $T/h.log 2>&1 || { echo "FAIL hole $K $R $S"; tail -2 $T/h.log; }
  done
done
python3 - $OUT $K <<'PY'
import glob, os, sys; from PIL import Image
O, K = sys.argv[1:3]
for p in glob.glob(f'{O}/fh/{K}-*.jpg'): Image.open(p).convert('RGB').save(p[:-4] + '.webp', quality=80, method=5); os.remove(p)
for p in glob.glob(f'{O}/f/{K}-*.jpg'):
    im = Image.open(p).convert('RGB'); w, h = im.size; im.save(p[:-4] + '.webp', quality=86, method=5)
    im.resize((1200, 1200 * h // w) if w > h else (1200 * w // h, 1200), Image.LANCZOS).save(p[:-4] + '-s.webp', quality=84, method=5)
PY
rm -rf $T $OUT/tmp; echo "heritage done $K $(ls $OUT/fh | grep -c webp) hole images"
