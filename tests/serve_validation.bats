#!/usr/bin/env bats
# tests/serve_validation.bats — Error paths and validation tests.

setup() {
  source "$(dirname "$BATS_TEST_FILENAME")/test_helper.bash"
  common_setup
  make_mock_tritonserver
  setup_serve_env
}

teardown() {
  common_teardown
}

@test "validation: TRITON_OPENAI_FRONTEND=banana fails" {
  export TRITON_OPENAI_FRONTEND=banana
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"must be true/false/1/0/yes/no"* ]]
}

@test "validation: TRITON_OPENAI_PORT=0 in openai mode fails" {
  export TRITON_OPENAI_FRONTEND=true
  export TRITON_OPENAI_PORT=0
  TRITON_OPENAI_MAIN="$(make_mock_openai_main)"
  export TRITON_OPENAI_MAIN
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"must be > 0"* ]]
}

@test "validation: TRITON_OPENAI_PORT=abc in openai mode fails" {
  export TRITON_OPENAI_FRONTEND=true
  export TRITON_OPENAI_PORT=abc
  TRITON_OPENAI_MAIN="$(make_mock_openai_main)"
  export TRITON_OPENAI_MAIN
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"must be a positive integer"* ]]
}

@test "validation: auto-discovery finds main.py relative to mock tritonserver" {
  export TRITON_OPENAI_FRONTEND=true
  export TRITON_OPENAI_MAIN=""
  # Create main.py relative to mock tritonserver: bin/../python/openai/main.py
  mkdir -p "$TEST_TMPDIR/python/openai"
  cat > "$TEST_TMPDIR/python/openai/main.py" <<'STUB'
#!/usr/bin/env python3
import sys; sys.exit(0)
STUB
  chmod +x "$TEST_TMPDIR/python/openai/main.py"
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
  [[ "$output" == *"main.py"* ]]
}

@test "validation: auto-discovery fails with clear error when main.py absent" {
  export TRITON_OPENAI_FRONTEND=true
  export TRITON_OPENAI_MAIN=""
  # tritonserver is on PATH but no main.py exists relative to it
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"not found"* ]]
}

@test "validation: TRITON_OPENAI_MAIN=/nonexistent/path fails" {
  export TRITON_OPENAI_FRONTEND=true
  export TRITON_OPENAI_MAIN="/nonexistent/path/main.py"
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"not found at"* ]]
}

@test "validation: standard mode requires tritonserver on PATH" {
  # Remove mock tritonserver
  rm -f "$TEST_TMPDIR/bin/tritonserver"
  # Also clear bash hash table
  hash -d tritonserver 2>/dev/null || true
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"tritonserver"* ]]
  [[ "$output" == *"not found"* ]]
}

# --- Path derivation tests ---

@test "path derivation: finds env file via FLOX_ENV_CACHE" {
  setup_serve_env_no_explicit_env_file
  make_mock_tritonserver
  local env_file
  env_file="$(make_resolve_model_env_file "$TEST_TMPDIR/cache" "$TEST_TMPDIR/model-repo" "my-model")"
  export FLOX_ENV_CACHE="$TEST_TMPDIR/cache"
  unset TRITON_MODEL_STATE_DIR
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
}

@test "path derivation: finds env file via TRITON_MODEL_STATE_DIR" {
  setup_serve_env_no_explicit_env_file
  make_mock_tritonserver
  local state_dir="$TEST_TMPDIR/custom-state"
  local env_file
  env_file="$(make_resolve_model_env_file "$state_dir" "$TEST_TMPDIR/model-repo" "my-model")"
  export TRITON_MODEL_STATE_DIR="$state_dir"
  unset FLOX_ENV_CACHE
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
}

@test "path derivation: falls back to repo/.triton-resolve-state" {
  setup_serve_env_no_explicit_env_file
  make_mock_tritonserver
  local env_file
  env_file="$(make_resolve_model_env_file "$TEST_TMPDIR/model-repo/.triton-resolve-state" "$TEST_TMPDIR/model-repo" "my-model")"
  unset FLOX_ENV_CACHE
  unset TRITON_MODEL_STATE_DIR
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
}

@test "path derivation: TRITON_MODEL_STATE_DIR takes priority over FLOX_ENV_CACHE" {
  setup_serve_env_no_explicit_env_file
  make_mock_tritonserver
  local state_dir="$TEST_TMPDIR/priority-state"
  local env_file
  env_file="$(make_resolve_model_env_file "$state_dir" "$TEST_TMPDIR/model-repo" "my-model")"
  export TRITON_MODEL_STATE_DIR="$state_dir"
  # Also set FLOX_ENV_CACHE to a different dir with NO env file — if priority is wrong, it will fail
  export FLOX_ENV_CACHE="$TEST_TMPDIR/wrong-cache"
  mkdir -p "$TEST_TMPDIR/wrong-cache"
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
}

@test "path derivation: TRITON_MODEL_ID uses slug from model_id but hash from repo/model" {
  setup_serve_env_no_explicit_env_file
  make_mock_tritonserver
  export TRITON_MODEL_ID="custom-org/custom-id"
  # Hash is still based on repo/model, but slug comes from TRITON_MODEL_ID
  local env_file
  env_file="$(make_resolve_model_env_file "$TEST_TMPDIR/cache" "$TEST_TMPDIR/model-repo" "my-model" "custom-org/custom-id")"
  export FLOX_ENV_CACHE="$TEST_TMPDIR/cache"
  unset TRITON_MODEL_STATE_DIR
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
}

@test "path derivation: TRITON_MODEL_ORG derives model_id as org/model" {
  setup_serve_env_no_explicit_env_file
  make_mock_tritonserver
  export TRITON_MODEL_ORG="myorg"
  # model_id becomes "myorg/my-model", hash is still repo/model
  local env_file
  env_file="$(make_resolve_model_env_file "$TEST_TMPDIR/cache" "$TEST_TMPDIR/model-repo" "my-model" "myorg/my-model")"
  export FLOX_ENV_CACHE="$TEST_TMPDIR/cache"
  unset TRITON_MODEL_STATE_DIR
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
}

@test "path derivation: fails when TRITON_MODEL unset" {
  setup_serve_env_no_explicit_env_file
  make_mock_tritonserver
  unset TRITON_MODEL
  export FLOX_ENV_CACHE="$TEST_TMPDIR/cache"
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"TRITON_MODEL is required"* ]]
}

@test "path derivation: fails when state dir doesn't exist" {
  setup_serve_env_no_explicit_env_file
  make_mock_tritonserver
  export TRITON_MODEL_STATE_DIR="$TEST_TMPDIR/nonexistent-state-dir"
  unset FLOX_ENV_CACHE
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -ne 0 ]
  [[ "$output" == *"State directory does not exist"* ]]
}

@test "path derivation: legacy unhashed env file fallback" {
  setup_serve_env_no_explicit_env_file
  make_mock_tritonserver
  local state_dir="$TEST_TMPDIR/legacy-state"
  mkdir -p "$state_dir"
  # Compute slug independently for the legacy (unhashed) filename
  local slug
  slug="$(printf '%s' "my-model" | tr -cs 'A-Za-z0-9._-' '-' | sed -e 's/^-*//' -e 's/-*$//')"
  local legacy_path="$state_dir/triton-model.${slug}.env"
  cat > "$legacy_path" <<EOF
export _TRITON_RESOLVED_PATH='$TEST_TMPDIR/resolved'
export TRITON_MODEL_REPOSITORY='$TEST_TMPDIR/model-repo'
EOF
  export TRITON_MODEL_STATE_DIR="$state_dir"
  unset FLOX_ENV_CACHE
  run "$SCRIPTS_DIR/triton-serve" --dry-run
  [ "$status" -eq 0 ]
}
