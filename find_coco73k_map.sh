#!/usr/bin/env bash
set -e
find /mnt/work -type f 2>/dev/null | grep -Eia "coco.*73|73k|cocoid|coco_ids|imgids|stim.*coco" | head -n 200
