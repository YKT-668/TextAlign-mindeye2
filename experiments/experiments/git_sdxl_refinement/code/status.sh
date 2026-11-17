#!/usr/bin/env bash
# ConceptAlign 下载状态查看。
set -u
ROOT=/mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114
source "$ROOT/env.sh"
exec python "$ROOT/scripts/show_status.py" "$@"
