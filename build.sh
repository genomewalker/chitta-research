#!/usr/bin/env bash
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
unset CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER
export PATH="/usr/bin:$HOME/.rustup/toolchains/1.92.0-x86_64-unknown-linux-gnu/bin:$PATH"
# Build on local /var/tmp to avoid NFS write failures (libgit2-sys rlib truncation)
# /var/tmp is on /dev/mapper/rhel-root (local), persists across reboots unlike /tmp
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/var/tmp/cr-target}"
exec cargo "$@"
