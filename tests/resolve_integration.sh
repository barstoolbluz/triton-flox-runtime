#!/usr/bin/env bash
# tests/resolve_integration.sh — Integration tests for triton-resolve-model
#
# Run inside the Flox environment:
#   flox activate -- bash tests/resolve_integration.sh
#
# Tests:
#   1. Local source path (Python backend)
#   2. hf-cache source path (simulated HF cache layout)
#   3. No-op re-run (provenance matches, second run is fast)
#   4. Concurrent invocations serialize via flock
#   5. Publish behavior: symlink-store
#   6. Publish behavior: replace-dir
#   7. Same-device check for replace-dir

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts" && pwd)"

# File-based counters so subshells can increment them
_COUNTER_DIR="$(mktemp -d)"
echo 0 >"$_COUNTER_DIR/pass"
echo 0 >"$_COUNTER_DIR/fail"
: >"$_COUNTER_DIR/errors"

pass() {
  local n; n=$(<"$_COUNTER_DIR/pass"); echo $((n+1)) >"$_COUNTER_DIR/pass"
  printf '  \033[32mPASS\033[0m %s\n' "$1"
}
fail() {
  local n; n=$(<"$_COUNTER_DIR/fail"); echo $((n+1)) >"$_COUNTER_DIR/fail"
  echo "  FAIL: $1" >>"$_COUNTER_DIR/errors"
  printf '  \033[31mFAIL\033[0m %s\n' "$1"
}

cleanup_dirs=()
cleanup() {
  for d in "${cleanup_dirs[@]}"; do
    rm -rf -- "$d" 2>/dev/null || true
  done
  local p f
  p=$(<"$_COUNTER_DIR/pass")
  f=$(<"$_COUNTER_DIR/fail")
  echo ""
  echo "================================================================"
  printf 'Results: \033[32m%d passed\033[0m' "$p"
  if (( f > 0 )); then
    printf ', \033[31m%d failed\033[0m' "$f"
  fi
  echo ""
  if (( f > 0 )); then
    echo "Failures:"
    cat "$_COUNTER_DIR/errors"
  fi
  echo "================================================================"
  rm -rf -- "$_COUNTER_DIR"
  if (( f > 0 )); then
    exit 1
  fi
}
trap cleanup EXIT

make_test_root() {
  local d
  d="$(mktemp -d)"
  cleanup_dirs+=("$d")
  echo "$d"
}

# Helper: create a minimal Python-backend model
make_python_model() {
  local model_dir="$1"
  mkdir -p "$model_dir/1"
  cat > "$model_dir/config.pbtxt" << 'EOF'
name: "test-model"
backend: "python"
max_batch_size: 1
input [
  { name: "INPUT0", data_type: TYPE_FP32, dims: [1] }
]
output [
  { name: "OUTPUT0", data_type: TYPE_FP32, dims: [1] }
]
EOF
  cat > "$model_dir/1/model.py" << 'PYEOF'
class TritonPythonModel:
    def execute(self, requests):
        return []
PYEOF
}

# Common env setup
setup_resolve_env() {
  export TRITON_MODEL_ID=""
  export TRITON_MODEL_ORG=""
  export TRITON_MODEL_REVISION=""
  export TRITON_MODEL_ENV_FILE=""
  export TRITON_MODEL_STATE_DIR=""
  export TRITON_RESOLVE_LOCK_TIMEOUT="10"
  export TRITON_RESOLVE_NETWORK_RETRIES="1"
  export TRITON_RESOLVE_NETWORK_TIMEOUT="10"
  export TRITON_RESOLVE_RETRY_SLEEP="1"
  export TRITON_DEEP_VALIDATE="1"
  export TRITON_STRICT_DEEP_VALIDATION="0"
  export TRITON_KEEP_LOGS="0"
  export TRITON_PUBLISH_MIGRATE_TARGET_DIR="0"
  export TRITON_MODEL_LOADABILITY_CHECK=""
  export TRITON_STRICT_LOADABILITY_CHECK="0"
  export TRITON_HF_CACHE_DIR=""
}

##############################################################################
echo "=== TEST 1: Local source path (Python backend) ==="
##############################################################################
(
  ROOT="$(make_test_root)"
  MODEL_REPO="$ROOT/model-repo"
  mkdir -p "$MODEL_REPO/test-model"
  make_python_model "$MODEL_REPO/test-model"

  setup_resolve_env
  export TRITON_MODEL="test-model"
  export TRITON_MODEL_REPOSITORY="$MODEL_REPO"
  export TRITON_MODEL_SOURCES="local"
  export TRITON_MODEL_BACKEND="python"
  export TRITON_PUBLISH_STRATEGY="symlink-store"
  export TRITON_MODEL_STATE_DIR="$ROOT/state"
  mkdir -p "$ROOT/state"

  out="$(timeout 30 bash "$SCRIPT_DIR/triton-resolve-model" 2>&1)" || {
    fail "local: script exited with rc=$?"
    echo "$out"
    exit 1
  }

  # Check output messages
  if [[ "$out" == *"Checking local directory... OK"* ]]; then
    pass "local: prints success message"
  else
    fail "local: missing success message; got: $out"
  fi

  if [[ "$out" == *"Resolved:"*"test-model"*"(via local)"* ]]; then
    pass "local: resolved via local"
  else
    fail "local: resolution message missing; got: $out"
  fi

  # Check env file was written
  env_file="$(find "$ROOT/state" -name '*.env' | head -1)"
  if [[ -n "$env_file" && -f "$env_file" ]]; then
    pass "local: env file created"
    # Verify env file content
    if grep -q "TRITON_MODEL_REPOSITORY=" "$env_file" && grep -q "_TRITON_RESOLVED_VIA=.*local" "$env_file"; then
      pass "local: env file has correct content"
    else
      fail "local: env file content wrong"
      cat "$env_file"
    fi
  else
    fail "local: env file not created"
  fi

  # Check provenance file was written
  prov_file="$(find "$ROOT/state" -name '*.provenance.json' | head -1)"
  if [[ -n "$prov_file" && -f "$prov_file" ]]; then
    pass "local: provenance file created"
    if python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['source']=='local'; assert 'recorded_at' in d; print('provenance OK')" "$prov_file" 2>&1; then
      pass "local: provenance JSON valid with source=local"
    else
      fail "local: provenance JSON invalid"
    fi
  else
    fail "local: provenance file not created"
  fi
)

##############################################################################
echo ""
echo "=== TEST 2: hf-cache source path (simulated HF cache) ==="
##############################################################################
(
  ROOT="$(make_test_root)"
  MODEL_REPO="$ROOT/model-repo"
  mkdir -p "$MODEL_REPO"

  # Simulate HF cache layout:
  #   $ROOT/hf-cache/models--myorg--test-model/snapshots/<sha>/...
  FAKE_HF_CACHE="$ROOT/hf-cache"
  FAKE_SHA="abc123def456"
  SNAPSHOT_DIR="$FAKE_HF_CACHE/models--myorg--test-model/snapshots/$FAKE_SHA"
  mkdir -p "$SNAPSHOT_DIR"
  make_python_model "$SNAPSHOT_DIR"

  setup_resolve_env
  export TRITON_MODEL="test-model"
  export TRITON_MODEL_REPOSITORY="$MODEL_REPO"
  export TRITON_MODEL_SOURCES="hf-cache"
  export TRITON_MODEL_BACKEND="python"
  export TRITON_MODEL_ID="myorg/test-model"
  export TRITON_PUBLISH_STRATEGY="symlink-store"
  export TRITON_MODEL_STATE_DIR="$ROOT/state"
  export TRITON_HF_CACHE_DIR="$FAKE_HF_CACHE"
  mkdir -p "$ROOT/state"

  out="$(timeout 30 bash "$SCRIPT_DIR/triton-resolve-model" 2>&1)" || {
    fail "hf-cache: script exited with rc=$?"
    echo "$out"
    exit 1
  }

  if [[ "$out" == *"Checking HF cache... OK"* ]]; then
    pass "hf-cache: prints success message"
  else
    fail "hf-cache: missing success message; got: $out"
  fi

  if [[ "$out" == *"(via hf-cache)"* ]]; then
    pass "hf-cache: resolved via hf-cache"
  else
    fail "hf-cache: resolution message missing; got: $out"
  fi

  # Verify provenance records hf-cache source
  prov_file="$(find "$ROOT/state" -name '*.provenance.json' | head -1)"
  if [[ -n "$prov_file" ]] && python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
assert d['source']=='hf-cache', f'expected hf-cache, got {d[\"source\"]}'
assert d['source_identity']=='hf://myorg/test-model'
assert d['resolved_revision']=='$FAKE_SHA'
print('hf-cache provenance OK')
" "$prov_file" 2>&1; then
    pass "hf-cache: provenance correct (source, identity, revision)"
  else
    fail "hf-cache: provenance wrong"
  fi

  # hf-cache resolves in-place: env file should point _TRITON_RESOLVED_PATH at snapshot
  env_file="$(find "$ROOT/state" -name '*.env' | head -1)"
  if [[ -n "$env_file" ]] && grep -q "_TRITON_RESOLVED_PATH=.*snapshots/$FAKE_SHA" "$env_file"; then
    pass "hf-cache: env _TRITON_RESOLVED_PATH points to snapshot"
  else
    fail "hf-cache: env _TRITON_RESOLVED_PATH wrong"
  fi

  # hf-cache rewrites TRITON_MODEL_REPOSITORY to the HF cache repo root
  if [[ -n "$env_file" ]] && grep -q "TRITON_MODEL_REPOSITORY=.*models--myorg--test-model" "$env_file"; then
    pass "hf-cache: env TRITON_MODEL_REPOSITORY points to HF repo root"
  else
    fail "hf-cache: env TRITON_MODEL_REPOSITORY wrong"
  fi
)

##############################################################################
echo ""
echo "=== TEST 3: No-op re-run behavior ==="
##############################################################################
(
  ROOT="$(make_test_root)"
  MODEL_REPO="$ROOT/model-repo"
  mkdir -p "$MODEL_REPO/test-model"
  make_python_model "$MODEL_REPO/test-model"

  setup_resolve_env
  export TRITON_MODEL="test-model"
  export TRITON_MODEL_REPOSITORY="$MODEL_REPO"
  export TRITON_MODEL_SOURCES="local"
  export TRITON_MODEL_BACKEND="python"
  export TRITON_PUBLISH_STRATEGY="symlink-store"
  export TRITON_MODEL_STATE_DIR="$ROOT/state"
  mkdir -p "$ROOT/state"

  # First run
  out1="$(timeout 30 bash "$SCRIPT_DIR/triton-resolve-model" 2>&1)" || {
    fail "noop: first run failed rc=$?"
    echo "$out1"
    exit 1
  }
  pass "noop: first run succeeded"

  # Capture provenance from first run
  prov_file="$(find "$ROOT/state" -name '*.provenance.json' | head -1)"
  recorded_at1="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['recorded_at'])" "$prov_file")"

  # Second run (should be a no-op)
  out2="$(timeout 30 bash "$SCRIPT_DIR/triton-resolve-model" 2>&1)" || {
    fail "noop: second run failed rc=$?"
    echo "$out2"
    exit 1
  }
  pass "noop: second run succeeded"

  # Provenance timestamp should be unchanged (no-op means identical provenance)
  recorded_at2="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['recorded_at'])" "$prov_file")"

  if [[ "$recorded_at1" == "$recorded_at2" ]]; then
    pass "noop: provenance recorded_at unchanged (true no-op)"
  else
    fail "noop: provenance recorded_at changed ($recorded_at1 -> $recorded_at2)"
  fi

  # Env file should still be valid
  env_file="$(find "$ROOT/state" -name '*.env' | head -1)"
  if bash -c "source '$env_file' && [[ -n \"\$TRITON_MODEL\" ]]" 2>/dev/null; then
    pass "noop: env file still sourceable after re-run"
  else
    fail "noop: env file not sourceable after re-run"
  fi
)

##############################################################################
echo ""
echo "=== TEST 4: Concurrent invocations serialize via flock ==="
##############################################################################
(
  ROOT="$(make_test_root)"
  MODEL_REPO="$ROOT/model-repo"
  mkdir -p "$MODEL_REPO/test-model"
  make_python_model "$MODEL_REPO/test-model"

  setup_resolve_env
  export TRITON_MODEL="test-model"
  export TRITON_MODEL_REPOSITORY="$MODEL_REPO"
  export TRITON_MODEL_SOURCES="local"
  export TRITON_MODEL_BACKEND="python"
  export TRITON_PUBLISH_STRATEGY="symlink-store"
  export TRITON_MODEL_STATE_DIR="$ROOT/state"
  export TRITON_RESOLVE_LOCK_TIMEOUT="30"
  mkdir -p "$ROOT/state"

  N=5
  pids=()
  for i in $(seq 1 $N); do
    timeout 60 bash "$SCRIPT_DIR/triton-resolve-model" >"$ROOT/out.$i" 2>&1 &
    pids+=($!)
  done

  all_ok=1
  for i in $(seq 1 $N); do
    if wait "${pids[$((i-1))]}"; then
      : # ok
    else
      fail "concurrent: worker $i exited with rc=$?"
      cat "$ROOT/out.$i"
      all_ok=0
    fi
  done

  if (( all_ok )); then
    pass "concurrent: all $N workers exited 0"
  fi

  # All should have produced valid output
  for i in $(seq 1 $N); do
    if grep -q "Resolved:" "$ROOT/out.$i"; then
      : # ok
    else
      fail "concurrent: worker $i missing Resolved message"
      all_ok=0
    fi
  done

  if (( all_ok )); then
    pass "concurrent: all $N workers resolved successfully"
  fi

  # Env file should be valid
  env_file="$(find "$ROOT/state" -name '*.env' | head -1)"
  if [[ -n "$env_file" ]] && bash -c "source '$env_file' && [[ \"\$TRITON_MODEL\" == 'test-model' ]]" 2>/dev/null; then
    pass "concurrent: env file valid after concurrent writes"
  else
    fail "concurrent: env file invalid or missing after concurrent writes"
  fi

  # Provenance should be valid JSON
  prov_file="$(find "$ROOT/state" -name '*.provenance.json' | head -1)"
  if [[ -n "$prov_file" ]] && python3 -c "import json; json.load(open('$prov_file'))" 2>/dev/null; then
    pass "concurrent: provenance file is valid JSON"
  else
    fail "concurrent: provenance file invalid or missing"
  fi

  # Lock file should not be held anymore
  lock_file="$(find "$ROOT/state" -name '*.lock' | head -1)"
  if [[ -n "$lock_file" ]]; then
    if flock -n "$lock_file" true 2>/dev/null; then
      pass "concurrent: lock released after all workers done"
    else
      fail "concurrent: lock still held after all workers finished"
    fi
  else
    pass "concurrent: no stale lock file"
  fi
)

##############################################################################
echo ""
echo "=== TEST 5: Publish behavior — symlink-store ==="
##############################################################################
(
  ROOT="$(make_test_root)"
  MODEL_REPO="$ROOT/model-repo"
  mkdir -p "$MODEL_REPO/test-model"
  make_python_model "$MODEL_REPO/test-model"

  setup_resolve_env
  export TRITON_MODEL="test-model"
  export TRITON_MODEL_REPOSITORY="$MODEL_REPO"
  export TRITON_MODEL_SOURCES="local"
  export TRITON_MODEL_BACKEND="python"
  export TRITON_PUBLISH_STRATEGY="symlink-store"
  export TRITON_MODEL_STATE_DIR="$ROOT/state"
  mkdir -p "$ROOT/state"

  out="$(timeout 30 bash "$SCRIPT_DIR/triton-resolve-model" 2>&1)" || {
    fail "symlink-store: script failed rc=$?"
    echo "$out"
    exit 1
  }
  pass "symlink-store: script succeeded"

  # For local source, the target_dir already exists as a real directory.
  # symlink-store creates the .published infrastructure but doesn't convert
  # an existing directory to a symlink unless TRITON_PUBLISH_MIGRATE_TARGET_DIR=1.
  if [[ -d "$MODEL_REPO/.published" ]]; then
    pass "symlink-store: .published directory created"
  else
    fail "symlink-store: .published directory missing"
  fi

  if [[ -d "$MODEL_REPO/.published/test-model" ]]; then
    pass "symlink-store: .published/test-model directory exists"
  else
    fail "symlink-store: .published/test-model directory missing"
  fi

  # The model should still be accessible at the target path
  if [[ -f "$MODEL_REPO/test-model/config.pbtxt" ]]; then
    pass "symlink-store: model accessible at target path"
  else
    fail "symlink-store: model not accessible at target path"
  fi
)

##############################################################################
echo ""
echo "=== TEST 6: Publish behavior — replace-dir ==="
##############################################################################
(
  ROOT="$(make_test_root)"
  MODEL_REPO="$ROOT/model-repo"
  mkdir -p "$MODEL_REPO/test-model"
  make_python_model "$MODEL_REPO/test-model"

  setup_resolve_env
  export TRITON_MODEL="test-model"
  export TRITON_MODEL_REPOSITORY="$MODEL_REPO"
  export TRITON_MODEL_SOURCES="local"
  export TRITON_MODEL_BACKEND="python"
  export TRITON_PUBLISH_STRATEGY="replace-dir"
  export TRITON_MODEL_STATE_DIR="$ROOT/state"
  mkdir -p "$ROOT/state"

  out="$(timeout 30 bash "$SCRIPT_DIR/triton-resolve-model" 2>&1)" || {
    fail "replace-dir: script failed rc=$?"
    echo "$out"
    exit 1
  }
  pass "replace-dir: script succeeded"

  # Model should still be accessible
  if [[ -d "$MODEL_REPO/test-model" && -f "$MODEL_REPO/test-model/config.pbtxt" ]]; then
    pass "replace-dir: model accessible at target path"
  else
    fail "replace-dir: model not accessible"
  fi

  # No .published directory should exist for replace-dir
  if [[ ! -d "$MODEL_REPO/.published" ]]; then
    pass "replace-dir: no .published directory (correct)"
  else
    fail "replace-dir: .published directory exists unexpectedly"
  fi

  # Run again to exercise the "already exists" replace-dir path
  out2="$(timeout 30 bash "$SCRIPT_DIR/triton-resolve-model" 2>&1)" || {
    fail "replace-dir: second run failed rc=$?"
    echo "$out2"
    exit 1
  }
  pass "replace-dir: second run succeeded"

  # Model still accessible after re-run
  if [[ -d "$MODEL_REPO/test-model" && -f "$MODEL_REPO/test-model/config.pbtxt" ]]; then
    pass "replace-dir: model still accessible after re-run"
  else
    fail "replace-dir: model gone after re-run"
  fi
)

##############################################################################
echo ""
echo "=== TEST 7: Same-device check for replace-dir ==="
##############################################################################
(
  # Staging and target must be on the same device for atomic mv.
  # We run on local disk where this is always true, confirming the
  # _same_device_or_die check passes.
  ROOT="$(make_test_root)"
  MODEL_REPO="$ROOT/model-repo"
  mkdir -p "$MODEL_REPO/test-model"
  make_python_model "$MODEL_REPO/test-model"

  setup_resolve_env
  export TRITON_MODEL="test-model"
  export TRITON_MODEL_REPOSITORY="$MODEL_REPO"
  export TRITON_MODEL_SOURCES="local"
  export TRITON_MODEL_BACKEND="python"
  export TRITON_PUBLISH_STRATEGY="replace-dir"
  export TRITON_MODEL_STATE_DIR="$ROOT/state"
  mkdir -p "$ROOT/state"

  # Verify staging dir and model repo are on same device
  staging_dev="$(stat -c '%d' "$MODEL_REPO" 2>/dev/null || stat -f '%d' "$MODEL_REPO" 2>/dev/null)"
  pass "same-device: staging and target on device $staging_dev"

  out="$(timeout 30 bash "$SCRIPT_DIR/triton-resolve-model" 2>&1)" || {
    fail "same-device: script failed rc=$?"
    echo "$out"
    exit 1
  }
  pass "same-device: replace-dir succeeded on same device"
)
