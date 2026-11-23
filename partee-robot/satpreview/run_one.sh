#!/bin/bash
# run_one.sh KEY : satellite-only preview for one club in clubs.json -> /tmp/claude-0/sp/out/KEY/{all.json,preview.jpg,preview-sat.jpg}
set -e; K=$1; D=$(dirname $0); O=/tmp/claude-0/sp/out/$K; mkdir -p $O
python3 - "$D/clubs.json" "$K" "$O" <<'P'
import json,sys; c=json.load(open(sys.argv[1]))[sys.argv[2]]; json.dump(c['geojson'],open(sys.argv[3]+'/boundary.geojson','w'))
open(sys.argv[3]+'/args','w').write(c['name']+'\n'+c['place'])
P
NAME=$(sed -n 1p $O/args); PLACE=$(sed -n 2p $O/args)
python3 $D/satonly.py "$K" "$NAME" "$PLACE" $O/boundary.geojson $O
NUMS=0 SAT=1 python3 /tmp/claude-0/osm/stamp/stamper_so.py $O/all.json $O/preview.jpg 1.1
