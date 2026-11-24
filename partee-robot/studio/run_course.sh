#!/bin/bash
# full studio pipeline for one course
set -eo pipefail; K=${1%%:*}; MODE=${1#*:}; R=/home/user/avendro-web-v2; S=/tmp/claude-0/osm/stamp; PY=/tmp/claude-0/samenv/bin/python
mkdir -p /tmp/claude-0/osm/teacher /tmp/claude-0/osm/qa $S/$K
cp $R/partee-site/courses/$K.json /tmp/claude-0/osm/$K.before-studio.json
T0=$(date +%s)
cd $R/partee-robot/tools
if [ "$MODE" = sig ]; then   # KEY:sig = Signature island only: no numbers, no lift, replayed gate decisions, PPM 2
cd /tmp/claude-0/osm/stamp; mkdir -p $K
GATE_REPLAY=1 python3 course_input.py $R/partee-site/courses/$K.json $K/all.json
GATE_REPLAY=1 ALPHA=1 NUMS=0 LIFT=0 python3 stamper.py $K/all.json $K/all.jpg 2.0 > $K/all.log 2>&1 || echo FAIL sig
echo "total $(( $(date +%s)-T0 )) s"; exit 0
fi
if [ "$MODE" = render ]; then   # KEY:render = already traced, just redraw the prints with the current studio code
PHOTO_ONLY=1 $PY photo_fix.py $K | tail -1
else
python3 verify.py $K --apply | tail -2
python3 osm_buildings.py $K | tail -1
$PY photo_fix.py $K | tail -2
cp $R/partee-site/courses/$K.json /tmp/claude-0/osm/teacher/$K.json
$PY seg.py $K | tail -1
fi
echo "traced in $(( $(date +%s)-T0 )) s"
cd $R/partee-robot/studio
if [ "$MODE" != render ]; then $PY qa_sat.py $K | tail -25; fi   # render-only reruns have no fresh trace to QA (last QA report stays)
cd $S
python3 course_input.py $R/partee-site/courses/$K.json $K/all.json
N=$(python3 -c "import json;print(' '.join(str(h['ref']) for h in json.load(open('$R/partee-site/courses/$K.json'))['holes'] if len(h['pts'])>1))")
for r in $N; do python3 hole_input.py $R/partee-site/courses/$K.json $r $K/$r.json > /dev/null; done
ALPHA=1 NUMS=1 SAT=1 python3 stamper.py $K/all.json $K/all.jpg 1.1 > $K/all.log 2>&1 || echo FAIL all
echo $N | tr ' ' '\n' | xargs -P3 -I{} sh -c "ALPHA=1 SAT=1 python3 stamper.py $K/{}.json $K/{}.jpg 3 > $K/{}.log 2>&1 || echo FAIL {}"
echo "total $(( $(date +%s)-T0 )) s"
