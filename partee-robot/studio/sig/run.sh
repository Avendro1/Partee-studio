#!/bin/bash
# Signature previews for one course: stamper island (ALPHA) -> premium.py in 3 colourways -> webp for the site.
K=$1; R=$(cd "$(dirname "$0")/../../.." && pwd); O=$R/out/sig; mkdir -p $O /tmp/sig
cd /tmp/claude-0/osm/stamp
python3 course_input.py $R/partee-site/courses/$K.json /tmp/sig/$K-in.json || exit 1
ALPHA=1 NUMS=0 LIFT=0 python3 stamper.py /tmp/sig/$K-in.json /tmp/sig/$K.jpg ${PPM:-1.1} || exit 1
for T in ivory charcoal navy; do
  python3 $R/partee-robot/studio/sig/premium.py /tmp/sig/$K-island.png $R/partee-site/courses/$K.json $T /tmp/sig/$K-$T.jpg > /tmp/sig/$K-$T.txt || exit 1
  python3 -c "
from PIL import Image; import sys
i=Image.open('/tmp/sig/$K-$T.jpg'); i.thumbnail((1600,1600)); i.save('$O/$K-sig-$T.webp',quality=86)
s=i.copy(); s.thumbnail((640,640)); s.save('$O/$K-sig-$T-s.webp',quality=82)"
done
cp /tmp/sig/$K-island.png $O/$K-island.png
awk '{print $2}' /tmp/sig/$K-ivory.txt > $O/$K-orient.txt
