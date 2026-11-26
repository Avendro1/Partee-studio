#!/bin/bash
# promote.sh KEY OUT : a newly redrawn course -> faded whole-course previews (4 styles, 1800x2400) + every hole (1200x1600), saved OSM/LiDAR inputs.
# OUT/f/KEY-{detailed,architectural,charcoal,midnight}.jpg  OUT/fh/KEY-REF-STYLE.jpg + KEY-REF.json  OUT/inputs/KEY/{ctx.json,lidar.png,lidar.json}
set -o pipefail; K=$1; OUT=$2; D=$(cd $(dirname $0); pwd); SIG=$D/sig; C=$D/courses/$K.json; I=$D/inputs/$K; T=$OUT/tmp/$K; mkdir -p $OUT/f $OUT/fh $T $I; cd $D
[ -f $I/ctx.json ] || python3 $SIG/ctx_fetch2.py $C $I/ctx.json 700 > $T/ctx.log 2>&1 || { echo "FAIL ctx $K"; tail -3 $T/ctx.log; exit 1; }
[ -f $I/lidar.png ] || python3 $SIG/lidar_fetch.py $C $I/lidar.png > $T/lidar.log 2>&1 || { echo "no lidar $K"; rm -f $I/lidar.png; }
mkdir -p $OUT/inputs/$K; cp $I/* $OUT/inputs/$K/ 2>/dev/null
GATE_REPLAY=1 python3 course_input.py $C $T/all.json > /dev/null || { echo "FAIL course_input $K"; exit 1; }
PPM=2.0; L=$I/lidar.png; [ -f $L ] || L=-
for P in 2.0 1.7; do PPM=$P; GATE_REPLAY=1 ALPHA=1 NUMS=0 LIFT=0 python3 stamper.py $T/all.json $T/all.jpg $PPM > $T/st.log 2>&1 && break; done || { echo "FAIL stamper $K"; tail -3 $T/st.log; }
RP=$(python3 -c "print(int(300*$PPM))"); E=$(python3 -c "print(int(380*$PPM))")
for st in "ivory 244,241,233 detailed" "charcoal 34,37,32 charcoal" "midnight 70,76,60 midnight"; do set -- $st
  HALO=2 PAINT_S=12 SURR_PAPER=$2 python3 $SIG/surr2.py $T/all-island.png $T/all-frame.json $C $I/ctx.json $L $T/surr.png $E $PPM $RP > $T/surr.log 2>&1 || { echo "FAIL surr $K $1"; continue; }
  SIZE=1800x2400 SURR=$T/surr.png SURR_E=$E SURR_SHRINK=.9 python3 $SIG/premium.py $T/all-island.png $C $1 $OUT/f/$K-$3.jpg > $T/prem.log 2>&1 || echo "FAIL premium $K $1"
done
rm -f $T/surr.png
LE=""; [ -f $I/lidar.png ] && LE="LIDAR=$I/lidar.png"
env SIZE=1800x2400 CTX=$I/ctx.json SURRCTX=$I/ctx.json SURR_R=.13 $LE python3 $SIG/plan.py $T/all.json $C $OUT/f/$K-architectural.jpg > $T/plan.log 2>&1 || echo "FAIL plan $K"
for R in $(python3 -c "import json; d=json.load(open('$C')); print('' if d.get('preview') else ' '.join(str(h['ref']) for h in d['holes'] if len(h.get('pts',[]))>1))"); do
  ALL=$T/all.json bash $D/fhole.sh $K $R $OUT/fh 1200x1600 2.5 || ALL=$T/all.json bash $D/fhole.sh $K $R $OUT/fh 1200x1600 2.0 || echo "FAILED $K $R"
done
rm -rf $T; echo "promoted $K"
