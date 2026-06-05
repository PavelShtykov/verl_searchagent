#!/usr/bin/env bash
#
# Symlink the SearchAgent patch files from ./patches into the vendored verl
# agent_loop package so they are importable as
# verl.experimental.agent_loop.<module>.
#
# Re-runnable and idempotent. Relative symlinks are used so the links keep
# working regardless of where the repo is cloned. Any pre-existing *real*
# file at a destination (e.g. the vendored __init__.py) is backed up once to
# <name>.orig before being replaced by a symlink.
#
# Usage:  bash patches/link.sh   [--unlink]
#   --unlink  remove the symlinks and restore any .orig backups.

set -euo pipefail

PATCHES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PATCHES_DIR/.." && pwd)"
TARGET_DIR="$REPO_ROOT/verl/verl/experimental/agent_loop"

FILES=(
  __init__.py
  searchagent_context_manager.py
  searchagent_loop.py
  searchagent_session.py
  searchagent_tools.py
)

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "ERROR: target dir not found: $TARGET_DIR" >&2
  exit 1
fi

unlink_mode=0
[[ "${1:-}" == "--unlink" ]] && unlink_mode=1

for f in "${FILES[@]}"; do
  src="$PATCHES_DIR/$f"
  dst="$TARGET_DIR/$f"

  if [[ "$unlink_mode" -eq 1 ]]; then
    if [[ -L "$dst" ]]; then
      rm -f "$dst"
      echo "unlinked $f"
    fi
    if [[ -e "$dst.orig" ]]; then
      mv -f "$dst.orig" "$dst"
      echo "restored $f from $f.orig"
    fi
    continue
  fi

  if [[ ! -f "$src" ]]; then
    echo "ERROR: missing source file: $src" >&2
    exit 1
  fi

  # Back up an existing real file (not an already-created symlink) exactly once.
  if [[ -e "$dst" && ! -L "$dst" && ! -e "$dst.orig" ]]; then
    cp -p "$dst" "$dst.orig"
    echo "backed up existing $f -> $f.orig"
  fi

  rel="$(realpath --relative-to="$TARGET_DIR" "$src")"
  ln -sfn "$rel" "$dst"
  echo "linked $f -> $rel"
done

echo "done."
