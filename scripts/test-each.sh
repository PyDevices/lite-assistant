#!/bin/bash
# Runs each test file in its own pytest process, so each file's models are
# freed before the next loads, which keeps a machine with little memory out of
# swap. Skips tests/test_live_server.py, which tests the running server, not
# this tree. Prints one line a file, and exits 1 if any failed.

cd "$(dirname "$0")/.."
python=.venv/bin/python
[ -x "$python" ] || python=python3

failed=0
for f in tests/test_*.py; do
    [ "$f" = tests/test_live_server.py ] && continue
    out=$(timeout 900 "$python" -m pytest -q -x "$f" 2>&1 | tail -1)
    echo "$f: $out"
    case "$out" in *failed*|*error*|"") failed=1;; esac
done
echo "done, failed=$failed"
exit $failed
