#!/usr/bin/env bash
# ============================================================================
# 在“任意新实例”上一次性配置 /mnt/data 数据盘自动挂载（幂等、可重复执行）。
#
# 用法（新 A100 实例上，root 执行）：
#   - 若数据盘已挂载：  bash /mnt/data/CONCEPTALIGN_GIT_IMG2IMG_20261114/scripts/setup_data_disk_automount.sh
#   - 若数据盘未挂载： 先把本脚本内容复制到本地执行（脚本自带设备探测，不依赖数据盘已挂载）
#
# 它做四件事：
#   1. 自动探测含 marker 目录的 ext4 分区，取 UUID（不依赖 /dev/vdX 设备名）
#   2. 写入 /etc/fstab（UUID + nofail + noatime），删除旧的 /mnt/data 条目
#   3. 安装 systemd 重试服务（云盘启动后才出现时自动补挂）
#   4. 立即挂载并校验
# ============================================================================
set -euo pipefail

MARKER="CONCEPTALIGN_GIT_IMG2IMG_20261114"
MOUNTPOINT="/mnt/data"
FSTAB_OPTS="defaults,nofail,noatime,x-systemd.device-timeout=30s"

log() { echo "[setup-mount] $*"; }

# ---------- 1. 探测 UUID ----------
UUID=""
if mountpoint -q "$MOUNTPOINT" && [ -d "$MOUNTPOINT/$MARKER" ]; then
    SRC="$(findmnt -no SOURCE "$MOUNTPOINT")"
    UUID="$(blkid -s UUID -o value "$SRC")"
    log "数据盘已挂载：$SRC (UUID=$UUID)"
fi

if [ -z "$UUID" ]; then
    log "未检测到已挂载的数据盘，开始扫描 ext4 分区……"
    while read -r dev fstype; do
        [ "$fstype" = "ext4" ] || continue
        tmp="$(mktemp -d)"
        if mount -o ro "$dev" "$tmp" 2>/dev/null; then
            if [ -d "$tmp/$MARKER" ]; then
                UUID="$(blkid -s UUID -o value "$dev")"
                log "命中数据盘：$dev (UUID=$UUID)"
                umount "$tmp"; rmdir "$tmp"
                break
            fi
            umount "$tmp"
        fi
        rmdir "$tmp" 2>/dev/null || true
    done < <(lsblk -lnpo NAME,FSTYPE | awk '{print $1, $2}')
fi

if [ -z "$UUID" ]; then
    log "错误：找不到含 $MARKER 目录的 ext4 数据盘。请确认云盘已挂到本实例。"
    exit 1
fi

# ---------- 2. 写 fstab（幂等） ----------
mkdir -p "$MOUNTPOINT"
cp -a /etc/fstab "/etc/fstab.bak.$(date +%Y%m%d%H%M%S)"
sed -i '\#^[^#]*[[:space:]]/mnt/data[[:space:]]#d' /etc/fstab
printf 'UUID=%s %s ext4 %s 0 2\n' "$UUID" "$MOUNTPOINT" "$FSTAB_OPTS" >> /etc/fstab
log "已写入 /etc/fstab：UUID=$UUID $MOUNTPOINT ext4 $FSTAB_OPTS 0 2"

# ---------- 3. 安装重试服务（自包含，不依赖数据盘） ----------
install -m 755 /dev/stdin /usr/local/bin/ensure-mnt-data.sh <<'EOS'
#!/usr/bin/env bash
set -u
if mountpoint -q /mnt/data; then exit 0; fi
for i in $(seq 1 40); do
    if mountpoint -q /mnt/data; then exit 0; fi
    if mount /mnt/data 2>/dev/null; then
        logger -t ensure-mnt-data "mounted /mnt/data (attempt ${i})"
        exit 0
    fi
    sleep 15
done
logger -t ensure-mnt-data "FAILED to mount /mnt/data after 40 attempts"
exit 1
EOS

install -m 644 /dev/stdin /etc/systemd/system/mnt-data-ensure.service <<'EOS'
[Unit]
Description=Ensure /mnt/data data disk is mounted (retry for late-attached volume)
Documentation=man:fstab(5)
After=local-fs.target
ConditionPathIsMountPoint=!/mnt/data

[Service]
Type=oneshot
ExecStart=/usr/local/bin/ensure-mnt-data.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOS

systemctl daemon-reload
systemctl enable mnt-data-ensure.service >/dev/null 2>&1 || true
log "已安装并启用 mnt-data-ensure.service"

# ---------- 4. 立即挂载 + 校验 ----------
if ! mountpoint -q "$MOUNTPOINT"; then
    mount "$MOUNTPOINT" && log "已挂载 $MOUNTPOINT"
fi
findmnt --verify >/tmp/fstab_verify.txt 2>&1 || true
log "fstab 校验：$(head -1 /tmp/fstab_verify.txt)"
findmnt "$MOUNTPOINT" || { log "错误：挂载失败"; exit 1; }
log "完成。重启后 /etc/fstab 会自动挂载；若云盘延迟出现，mnt-data-ensure.service 会自动重试。"
