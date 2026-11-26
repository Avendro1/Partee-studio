#!/bin/bash
# faded.sh KEY : the 25 Nov "faded surroundings" whole-course previews Shane approved, in all four shop styles.
# island (stamper, no numbers, replayed gate) -> real OSM surroundings (ctx_fetch2) + GSI LiDAR relief -> surr2 painted landscape
# feathering out from the course -> premium.py posters (ivory = Detailed, charcoal, midnight) + plan.py Architectural.
set -o pipefail; K=$1; R=/home/user/avendro-web-v2; S=/tmp/claude-0/osm/stamp; SIG=$R/partee-robot/studio/sig; C=$R/partee-site/courses/$K.json
PPM=${PPM:-2.0}; O=/tmp/claude-0/faded/$K; mkdir -p $O $S/$K; T0=$(date +%s)
cd $S
GATE_REPLAY=1 python3 course_input.py $C $K/all.json
GATE_REPLAY=1 ALPHA=1 NUMS=0 LIFT=0 python3 stamper.py $K/all.json $K/all.jpg $PPM > $K/all.log 2>&1 || { echo "FAIL stamper $K"; tail -5 $K/all.log; exit 1; }
python3 $SIG/ctx_fetch2.py $C $O/ctx.json 700 > $O/ctx.log 2>&1 || echo "WARN ctx $K"
python3 $SIG/lidar_fetch.py $C $O/lidar.png > $O/lidar.log 2>&1 || { echo "no lidar $K"; rm -f $O/lidar.png; }
L=$O/lidar.png; [ -f $L ] || L=-
RP=$(python3 -c "print(int(300*$PPM))"); E=$(python3 -c "print(int(380*$PPM))")
for st in "ivory 244,241,233" "charcoal 34,37,32" "midnight 70,76,60"; do set -- $st
  HALO=2 PAINT_S=12 SURR_PAPER=$2 python3 $SIG/surr2.py $K/all-island.png $K/all-frame.json $C $O/ctx.json $L $O/surr-$1.png $E $PPM $RP > $O/surr-$1.log 2>&1 || echo "FAIL surr $K $1"
  SIZE=1800x2400 SURR=$O/surr-$1.png SURR_E=$E SURR_SHRINK=.9 python3 $SIG/premium.py $K/all-island.png $C $1 $O/$K-$1.jpg > $O/prem-$1.log 2>&1 || echo "FAIL premium $K $1"
  rm -f $O/surr-$1.png
done
LIDARENV=""; [ -f $O/lidar.png ] && LIDARENV="LIDAR=$O/lidar.png"
env SIZE=1800x2400 CTX=$O/ctx.json SURRCTX=$O/ctx.json SURR_R=.13 $LIDARENV python3 $SIG/plan.py $K/all.json $C $O/$K-architectural.jpg > $O/plan.log 2>&1 || echo "FAIL plan $K"
cp $K/all-frame.json $O/frame.json
echo "done $K $(( $(date +%s)-T0 )) s"
