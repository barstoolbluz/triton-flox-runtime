#!/usr/bin/env bats
# tests/lib.bats — Unit tests for _lib.sh functions.

setup() {
  source "$(dirname "$BATS_TEST_FILENAME")/test_helper.bash"
  common_setup
  # Unset the guard so we can re-source _lib.sh
  unset _LIB_SH_LOADED
  source "$SCRIPTS_DIR/_lib.sh"
}

teardown() {
  common_teardown
}

# --- lib::is_truthy ---

@test "is_truthy: true returns 0" {
  lib::is_truthy "true"
}

@test "is_truthy: 1 returns 0" {
  lib::is_truthy "1"
}

@test "is_truthy: yes returns 0" {
  lib::is_truthy "yes"
}

@test "is_truthy: TRUE returns 0" {
  lib::is_truthy "TRUE"
}

@test "is_truthy: false returns 1" {
  run lib::is_truthy "false"
  [ "$status" -eq 1 ]
}

@test "is_truthy: 0 returns 1" {
  run lib::is_truthy "0"
  [ "$status" -eq 1 ]
}

@test "is_truthy: no returns 1" {
  run lib::is_truthy "no"
  [ "$status" -eq 1 ]
}

@test "is_truthy: empty returns 1" {
  run lib::is_truthy ""
  [ "$status" -eq 1 ]
}

@test "is_truthy: banana returns 1" {
  run lib::is_truthy "banana"
  [ "$status" -eq 1 ]
}

@test "is_truthy: on returns 0" {
  lib::is_truthy "on"
}

@test "is_truthy: ON returns 0" {
  lib::is_truthy "ON"
}

@test "is_truthy: off returns 1" {
  run lib::is_truthy "off"
  [ "$status" -eq 1 ]
}

@test "is_truthy: OFF returns 1" {
  run lib::is_truthy "OFF"
  [ "$status" -eq 1 ]
}

# --- lib::require_bool ---

@test "require_bool: accepts true" {
  TESTVAR=true
  lib::require_bool TESTVAR
}

@test "require_bool: accepts false" {
  TESTVAR=false
  lib::require_bool TESTVAR
}

@test "require_bool: accepts 1" {
  TESTVAR=1
  lib::require_bool TESTVAR
}

@test "require_bool: accepts 0" {
  TESTVAR=0
  lib::require_bool TESTVAR
}

@test "require_bool: accepts yes" {
  TESTVAR=yes
  lib::require_bool TESTVAR
}

@test "require_bool: accepts no" {
  TESTVAR=no
  lib::require_bool TESTVAR
}

@test "require_bool: rejects banana" {
  TESTVAR=banana
  run lib::require_bool TESTVAR
  [ "$status" -ne 0 ]
}

@test "require_bool: rejects empty" {
  TESTVAR=""
  run lib::require_bool TESTVAR
  [ "$status" -ne 0 ]
}

@test "require_bool: accepts on" {
  TESTVAR=on
  lib::require_bool TESTVAR
}

@test "require_bool: accepts off" {
  TESTVAR=off
  lib::require_bool TESTVAR
}

@test "require_bool: accepts ON" {
  TESTVAR=ON
  lib::require_bool TESTVAR
}

@test "require_bool: accepts OFF" {
  TESTVAR=OFF
  lib::require_bool TESTVAR
}

# --- lib::require_pos_int ---

@test "require_pos_int: accepts 1" {
  TESTVAR=1
  lib::require_pos_int TESTVAR
}

@test "require_pos_int: accepts 9000" {
  TESTVAR=9000
  lib::require_pos_int TESTVAR
}

@test "require_pos_int: rejects 0" {
  TESTVAR=0
  run lib::require_pos_int TESTVAR
  [ "$status" -ne 0 ]
}

@test "require_pos_int: rejects negative" {
  TESTVAR=-1
  run lib::require_pos_int TESTVAR
  [ "$status" -ne 0 ]
}

@test "require_pos_int: rejects non-integer" {
  TESTVAR=abc
  run lib::require_pos_int TESTVAR
  [ "$status" -ne 0 ]
}

@test "require_pos_int: rejects empty" {
  TESTVAR=""
  run lib::require_pos_int TESTVAR
  [ "$status" -ne 0 ]
}

# --- lib::slugify ---

@test "slugify: slashes become hyphens" {
  result="$(lib::slugify "org/model")"
  [ "$result" = "org-model" ]
}

@test "slugify: preserves dots and underscores" {
  result="$(lib::slugify "my_model.v2")"
  [ "$result" = "my_model.v2" ]
}

@test "slugify: collapses special chars" {
  result="$(lib::slugify "a@#\$b")"
  [ "$result" = "a-b" ]
}

# --- lib::normalize_bool_var ---

@test "normalize_bool_var: TRUE becomes true" {
  X=TRUE
  lib::normalize_bool_var X
  [ "$X" = "true" ]
}

@test "normalize_bool_var: YES becomes true" {
  X=YES
  lib::normalize_bool_var X
  [ "$X" = "true" ]
}

@test "normalize_bool_var: on becomes true" {
  X=on
  lib::normalize_bool_var X
  [ "$X" = "true" ]
}

@test "normalize_bool_var: OFF becomes false" {
  X=OFF
  lib::normalize_bool_var X
  [ "$X" = "false" ]
}

@test "normalize_bool_var: 0 becomes false" {
  X=0
  lib::normalize_bool_var X
  [ "$X" = "false" ]
}

@test "normalize_bool_var: empty becomes false" {
  X=""
  lib::normalize_bool_var X
  [ "$X" = "false" ]
}

@test "normalize_bool_var: custom values true->1 false->0" {
  X=true
  lib::normalize_bool_var X 0 1
  [ "$X" = "1" ]
  X=false
  lib::normalize_bool_var X 0 1
  [ "$X" = "0" ]
}

@test "normalize_bool_var: rejects banana" {
  X=banana
  run lib::normalize_bool_var X
  [ "$status" -ne 0 ]
  [[ "$output" == *"must be true/false/1/0/yes/no/on/off"* ]]
}

# --- lib::slugify_nonempty ---

@test "slugify_nonempty: valid input works" {
  result="$(lib::slugify_nonempty "hello-world")"
  [ "$result" = "hello-world" ]
}

@test "slugify_nonempty: empty input dies" {
  run lib::slugify_nonempty ""
  [ "$status" -ne 0 ]
  [[ "$output" == *"Slugified value is empty"* ]]
}

@test "slugify_nonempty: all-special-chars input dies" {
  run lib::slugify_nonempty "@#\$%"
  [ "$status" -ne 0 ]
  [[ "$output" == *"Slugified value is empty"* ]]
}

# --- lib::load_env_safe ---

@test "load_env_safe: basic single export" {
  local envf="$TEST_TMPDIR/basic.env"
  echo "export FOO='bar'" > "$envf"
  lib::load_env_safe "$envf"
  [ "$FOO" = "bar" ]
}

@test "load_env_safe: multiple variables" {
  local envf="$TEST_TMPDIR/multi.env"
  cat > "$envf" <<'EOF'
export AAA='one'
export BBB='two'
export CCC='three'
EOF
  lib::load_env_safe "$envf"
  [ "$AAA" = "one" ]
  [ "$BBB" = "two" ]
  [ "$CCC" = "three" ]
}

@test "load_env_safe: unquoted values" {
  local envf="$TEST_TMPDIR/unquoted.env"
  echo "export XVAL=hello" > "$envf"
  lib::load_env_safe "$envf"
  [ "$XVAL" = "hello" ]
}

@test "load_env_safe: ANSI-C quoting with newline" {
  local envf="$TEST_TMPDIR/ansic.env"
  printf "export NLVAL=\$'line1\\\\nline2'\n" > "$envf"
  lib::load_env_safe "$envf"
  local expected
  expected="$(printf 'line1\nline2')"
  [ "$NLVAL" = "$expected" ]
}

@test "load_env_safe: comments and blank lines are skipped" {
  local envf="$TEST_TMPDIR/comments.env"
  cat > "$envf" <<'EOF'
# This is a comment
export CVAL='yes'

# Another comment
export DVAL='no'
EOF
  lib::load_env_safe "$envf"
  [ "$CVAL" = "yes" ]
  [ "$DVAL" = "no" ]
}

@test "load_env_safe: rejects symlinks" {
  local real="$TEST_TMPDIR/real.env"
  local link="$TEST_TMPDIR/link.env"
  echo "export SYM='bad'" > "$real"
  ln -s "$real" "$link"
  run lib::load_env_safe "$link"
  [ "$status" -ne 0 ]
}

@test "load_env_safe: rejects missing file" {
  run lib::load_env_safe "$TEST_TMPDIR/nonexistent.env"
  [ "$status" -ne 0 ]
}
