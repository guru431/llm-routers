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

# 2b) A sensitive file inside a directory whose NAME contains ".example." must
# still be BLOCKED. The template exemption used to be applied to the whole path,
# so any such directory exempted everything under it.
mkdir -p foo.example.bar
echo "x" > foo.example.bar/id_rsa
git add -f foo.example.bar/id_rsa
if git commit -q -m "sensitive file under .example. dir" 2>/dev/null; then
  echo "FAIL: hook did NOT block a sensitive file under a *.example.* directory"; fails=1
else
  echo "ok: hook blocked sensitive file under a *.example.* directory"
fi
git reset -q
rm -rf foo.example.bar

# 2c) Every sensitive basename must be blocked, not just id_rsa.
for name in .env vault.env server.pem server.key cert.p12 cert.pfx id_ed25519 id_dsa; do
  echo "x" > "$name"
  git add -f "$name"
  if git commit -q -m "sensitive $name" 2>/dev/null; then
    echo "FAIL: hook did NOT block $name"; fails=1
  else
    echo "ok: hook blocked $name"
  fi
  git reset -q
  rm -f "$name"
done

# 2d) …but a real template (`<name>.example`) must still PASS.
echo "TOKEN=" > .env.example
git add .env.example
if git commit -q -m "env template"; then
  echo "ok: hook allowed .env.example template"
else
  echo "FAIL: hook wrongly blocked a .env.example template"; fails=1
fi

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
