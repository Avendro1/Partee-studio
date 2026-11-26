#!/bin/bash
# holes.sh KEY OUT : re-render every hole of a full course at 1800x2400 (site art), then pack OUT/KEY-REF.webp strips (detailed|architectural|charcoal|midnight) + OUT/c/KEY.json
set -o pipefail; K=$1; OUT=$2; D=$(cd $(dirname $0); pwd); SIG=$D/sig; C=$D/courses/$K.json; I=$D/inputs/$K; T=$OUT/tmp/$K; mkdir -p $OUT/raw $OUT/c $T $I; cd $D
[ -f $I/ctx.json ] || python3 $SIG/ctx_fetch2.py $C $I/ctx.json 700 > $T/ctx.log 2>&1 || { echo "FAIL ctx $K"; exit 1; }
[ -f $I/lidar.png ] || python3 $SIG/lidar_fetch.py $C $I/lidar.png > $T/lidar.log 2>&1 || rm -f $I/lidar.png
GATE_REPLAY=1 python3 course_input.py $C $T/all.json > /dev/null || { echo "FAIL course_input $K"; exit 1; }
for R in $(python3 -c "import json; d=json.load(open('$C')); print('' if d.get('preview') else ' '.join(str(h['ref']) for h in d['holes'] if len(h.get('pts',[]))>1))"); do
  ALL=$T/all.json bash $D/fhole.sh $K $R $OUT/raw 1800x2400 3.0 || ALL=$T/all.json bash $D/fhole.sh $K $R $OUT/raw 1800x2400 2.6 || echo "FAILED $K $R"
done
python3 - $K $OUT <<'P'
import sys, os, json, glob; from PIL import Image
K, O = sys.argv[1:3]; ST = ('detailed', 'architectural', 'charcoal', 'midnight'); cj = {}; n = 0
for p in sorted(glob.glob(f'{O}/raw/{K}-*.json')):
    r = os.path.basename(p)[len(K) + 1:-5]; srcs = [f'{O}/raw/{K}-{r}-{s}.jpg' for s in ST]
    if not all(os.path.exists(s) for s in srcs): print('MISSING', K, r); continue
    ims = [Image.open(s).convert('RGB') for s in srcs]; w, h = ims[0].size; o = Image.new('RGB', (w * 4, h))
    for i, im in enumerate(ims): o.paste(im if im.size == (w, h) else im.resize((w, h), Image.LANCZOS), (i * w, 0))
    o.save(f'{O}/{K}-{r}.webp', 'WEBP', quality=80, method=4); cj[r] = json.load(open(p)); n += 1
json.dump(cj, open(f'{O}/c/{K}.json', 'w'), separators=(',', ':')); print('strips', K, n)
P
rm -rf $OUT/raw $OUT/tmp
