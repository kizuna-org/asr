#!/usr/bin/env bash

set -euo pipefail

SAMBA_SERVER=samba
SAMBA_SHARE=Data
SAMBA_USER=samba
SAMBA_PASS=secret

MOUNT_POINT=/mnt/samba

mount -t cifs "//$SAMBA_SERVER/$SAMBA_SHARE" "$MOUNT_POINT" \
  -o "username=$SAMBA_USER,password=$SAMBA_PASS,vers=3.0,uid=0,gid=0,iocharset=utf8,noperm"
