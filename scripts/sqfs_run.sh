#!/bin/bash
# Run one command with the dataset squashfs images mounted read-only (2026-10-10).
#
#   PKABENCH_RUNTIME=<root> scripts/sqfs_run.sh <command> [args...]
#
# Mounts every <root>/images/<name>.sqfs at <mount root>/<name> (default /tmp/$USER-sqfs, PKABENCH_SQFS_MOUNT to
# override; must match what `dataset_transfer squash` linked) with squashfuse, inside a private user + mount namespace,
# then runs the command as the calling user. The mounts exist only for that command: other jobs on a shared node never
# see them, and they go away with it (no stale FUSE mounts when a job ends). Outside this wrapper the dataset
# directories are dangling symlinks, so a missing mount fails loudly. srun steps started by the command run outside
# the namespace; wrap the step itself (srun scripts/sqfs_run.sh ...).
# Needs unprivileged user namespaces (util-linux unshare >= 2.38 for --map-user) and squashfuse(_ll) + /dev/fuse.
set -euo pipefail
ROOT=${PKABENCH_RUNTIME:?set PKABENCH_RUNTIME}
MOUNT=${PKABENCH_SQFS_MOUNT:-/tmp/${USER:-$(id -u)}-sqfs}
SELF=$(readlink -f "$0")

if [ "${1:-}" = "--in-namespace" ]; then
  shift; uid=$1; gid=$2; shift 2
  fuse=$(command -v squashfuse_ll || command -v squashfuse)
  mounted=()
  cleanup() { for m in "${mounted[@]}"; do umount "$m" 2>/dev/null || fusermount3 -u "$m" 2>/dev/null || true; done; }
  trap cleanup EXIT
  shopt -s nullglob
  for image in "$ROOT"/images/*.sqfs; do
    name=$(basename "$image" .sqfs); point=$MOUNT/$name
    mkdir -p "$point"; "$fuse" -o ro "$image" "$point"; mounted+=("$point")
    [ -n "$(ls -A "$point")" ] || { echo "sqfs_run: $image mounted empty at $point" >&2; exit 1; }
  done
  [ ${#mounted[@]} -gt 0 ] || echo "sqfs_run: no images under $ROOT/images" >&2
  # back to the calling user's uid/gid for the command (a nested user namespace; the mounts stay visible)
  unshare --user --map-user="$uid" --map-group="$gid" -- "$@"
  exit $?
fi

[ $# -gt 0 ] || { echo "usage: PKABENCH_RUNTIME=<root> $0 <command> [args...]" >&2; exit 2; }
exec unshare --user --map-root-user --mount --propagation private -- bash "$SELF" --in-namespace "$(id -u)" "$(id -g)" "$@"
