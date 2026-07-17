#!/usr/bin/env bash
# CI fixture check for .githooks/pre-commit: prove the secret-guard BLOCKS a
# planted secret / sensitive filename and PASSES clean content. Runs the real
# hook against a throwaway git repo so a regression in the guard is caught in CI,
# not only by a developer's local clone.
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOK="$REPO_ROOT/.githooks/pre-commit"

if [ ! -f "$HOOK" ]; then
  echo "FAIL: hook not found at $HOOK"; exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"
git init -q
git config user.email ci@example.com
git config user.name ci
mkdir -p .githooks
cp "$HOOK" .githooks/pre-commit
chmod +x .githooks/pre-commit
git config core.hooksPath .githooks

fails=0

# 1) A planted AWS-key-shaped secret must be BLOCKED. The key is assembled at
# runtime (adjacent string concatenation) so THIS script's own source doesn't trip
# the real repo's pre-commit scanner — leak.py still gets the full AKIA... string.
akia="AKIA""IOSFODNN7EXAMPLE"
printf 'aws_key = "%s"\n' "$akia" > leak.py
git add leak.py
if git commit -q -m "planted secret" 2>/dev/null; then
  echo "FAIL: hook did NOT block a planted secret"; fails=1
else
  echo "ok: hook blocked planted secret"
fi
git reset -q

# 2) A sensitive filename must be BLOCKED.
echo "x" > id_rsa
git add -f id_rsa
if git commit -q -m "sensitive filename" 2>/dev/null; then
  echo "FAIL: hook did NOT block a sensitive filename"; fails=1
else
  echo "ok: hook blocked sensitive filename"
fi
git reset -q
rm -f id_rsa

# 3) Clean content must PASS.
rm -f leak.py
echo "print('hello world')" > clean.py
git add clean.py
if git commit -q -m "clean content"; then
  echo "ok: hook passed clean content"
else
  echo "FAIL: hook wrongly blocked clean content"; fails=1
fi

if [ "$fails" -ne 0 ]; then
  echo "pre-commit hook fixture check FAILED"; exit 1
fi
echo "pre-commit hook fixture check passed"
