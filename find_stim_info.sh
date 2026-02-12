#!/usr/bin/env bash
set -e
find /mnt/work -type f 2>/dev/null | grep -Ei "nsd.*stim.*info|stim.*info.*nsd|nsd.*stim.*merged|stim_info_merged" | head -n 50
