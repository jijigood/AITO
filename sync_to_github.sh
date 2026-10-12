#!/usr/bin/env bash

# Usage:
#   cd /home/huangxiaolin/AITO
#   ./sync_to_github.sh
#
# What it does:
#   1. Copies the current AITO directory into a temporary Git worktree.
#   2. Excludes files and folders listed in .sync-exclude.
#   3. Creates a Git commit for the synced content.
#   4. Pushes the commit to the GitHub repository.
#
# Default target:
#   git@github.com:jijigood/AITO.git
#
# Common examples:
#   ./sync_to_github.sh
#   SYNC_COMMIT_MESSAGE="Update dataset" ./sync_to_github.sh
#   SYNC_BRANCH=main ./sync_to_github.sh
#   SYNC_REPO_URL=git@github.com:jijigood/AITO.git ./sync_to_github.sh
#
# Notes:
#   - Exclude rules are read from: /home/huangxiaolin/AITO/.sync-exclude
#   - The temporary sync workspace is: /tmp/AITO_repo_sync
#   - SSH access to GitHub must already be configured on this machine.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${SYNC_REPO_URL:-git@github.com:jijigood/AITO.git}"
BRANCH="${SYNC_BRANCH:-main}"
WORKTREE="${SYNC_WORKTREE:-/tmp/AITO_repo_sync}"
EXCLUDE_FILE="${SYNC_EXCLUDE_FILE:-$ROOT_DIR/.sync-exclude}"
COMMIT_MESSAGE="${SYNC_COMMIT_MESSAGE:-Sync from AITO on $(date '+%Y-%m-%d %H:%M:%S %z')}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd git
require_cmd rsync

if [[ ! -f "$EXCLUDE_FILE" ]]; then
  echo "Exclude file not found: $EXCLUDE_FILE" >&2
  exit 1
fi

case "$WORKTREE" in
  /tmp/*) ;;
  *)
    echo "SYNC_WORKTREE must stay under /tmp for safety. Current: $WORKTREE" >&2
    exit 1
    ;;
esac

echo "Preparing sync workspace: $WORKTREE"
rm -rf "$WORKTREE"
mkdir -p "$WORKTREE"

echo "Copying files from $ROOT_DIR"
rsync -a --delete \
  --exclude-from="$EXCLUDE_FILE" \
  "$ROOT_DIR"/ "$WORKTREE"/

if [[ ! -d "$WORKTREE/.git" ]]; then
  git -C "$WORKTREE" init -b "$BRANCH" >/dev/null
fi

git -C "$WORKTREE" config user.name "${SYNC_GIT_USER_NAME:-jijigood}"
git -C "$WORKTREE" config user.email "${SYNC_GIT_USER_EMAIL:-jijigood@users.noreply.github.com}"
git -C "$WORKTREE" remote remove origin >/dev/null 2>&1 || true
git -C "$WORKTREE" remote add origin "$REPO_URL"

if command -v git-lfs >/dev/null 2>&1; then
  git -C "$WORKTREE" lfs install --local >/dev/null
fi

git -C "$WORKTREE" add -A

if git -C "$WORKTREE" diff --cached --quiet; then
  echo "No changes to sync."
  exit 0
fi

git -C "$WORKTREE" commit -m "$COMMIT_MESSAGE"
git -C "$WORKTREE" push -u origin "$BRANCH"

echo "Sync complete."
