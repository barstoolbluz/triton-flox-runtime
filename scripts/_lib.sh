#!/usr/bin/env bash
# _lib.production.sh — Shared library for Triton operational scripts.
#
# Source this file; do not execute it directly.
#
# Portability:
#   - Bash 4.1+
#   - GNU/Linux userland assumptions for stat/mktemp/mv/find semantics
#   - Requires python3 for selected helpers
#
# Public API:
#   lib::log, lib::warn, lib::err, lib::die
#   lib::require_var, lib::require_pos_int, lib::require_bool
#   lib::is_int, lib::is_unsigned_decimal, lib::is_num (compat alias)
#   lib::normalize_bool_var, lib::is_truthy
#   lib::sha256_12
#   lib::slugify, lib::slugify_nonempty
#   lib::flock_open_or_die, lib::flock_close, lib::flock_or_die
#   lib::write_env_atomically
#   lib::load_env_safe, lib::load_env_trusted
#   lib::replace_dir_with_backup, lib::restore_backup, lib::atomic_swap_dir (compat alias)
#   lib::validate_model_name
#   lib::run_to_log
#
# Notes:
#   - write_env_atomically and run_to_log use same-directory temp files, fsync the
#     temp file, rename into place, and fsync the parent directory.
#   - replace_dir_with_backup is recoverable but not a single-step atomic swap from
#     a reader's point of view when a prior target directory exists.
#   - slugify may return an empty string if all characters are filtered away.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '%s\n' "This file is a library. Source it from another script." >&2
  exit 1
fi

[[ -n "${_LIB_SH_LOADED:-}" ]] && return 0
readonly _LIB_SH_LOADED=1

declare -Ag _LIB_LOCK_HELPER_DIRS=()

# 0=quiet  1=normal  2=verbose
TRITON_VERBOSITY="${TRITON_VERBOSITY:-1}"

# ----------------------------
# Internal helpers
# ----------------------------

lib::_have_cmd() {
  command -v -- "$1" >/dev/null 2>&1
}

lib::_require_cmd() {
  lib::_have_cmd "$1" || lib::die "Required command not found in PATH: $1"
}

lib::_validate_identifier() {
  [[ "${1-}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]
}

lib::_require_identifier() {
  local name="${1-}"
  lib::_validate_identifier "$name" || lib::die "Invalid shell identifier: $name"
}

lib::_restore_shell_flags() {
  local had_a="${1-}" had_u="${2-}"
  [[ "$had_a" == "1" ]] && set -a || set +a
  [[ "$had_u" == "1" ]] && set -u || set +u
}

lib::_reject_symlink_if_exists() {
  local path="${1-}"
  [[ -n "$path" ]] || lib::die "Path is empty"
  if [[ -e "$path" || -L "$path" ]]; then
    [[ ! -L "$path" ]] || lib::die "Path is a symlink: $path"
  fi
}

lib::_prepare_parent_dir() {
  local path="${1-}"
  [[ -n "$path" ]] || lib::die "Path is empty"

  local parent
  parent="$(dirname -- "$path")"

  if [[ -e "$parent" || -L "$parent" ]]; then
    [[ -d "$parent" ]] || lib::die "Parent path is not a directory: $parent"
    [[ ! -L "$parent" ]] || lib::die "Parent directory is a symlink: $parent"
  else
    mkdir -p -- "$parent" || lib::die "Failed to create parent directory: $parent"
  fi

  [[ -w "$parent" ]] || lib::die "Parent directory is not writable: $parent"
}

lib::_fsync_file() {
  local path="${1-}"
  [[ -n "$path" ]] || lib::die "fsync_file path is empty"
  lib::_require_cmd python3
  python3 - "$path" <<'PY'
import os, sys
path = sys.argv[1]
fd = os.open(path, os.O_RDONLY)
try:
    os.fsync(fd)
finally:
    os.close(fd)
PY
}

lib::_fsync_dir() {
  local path="${1-}"
  [[ -n "$path" ]] || lib::die "fsync_dir path is empty"
  lib::_require_cmd python3
  python3 - "$path" <<'PY'
import os, sys
path = sys.argv[1]
fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
try:
    os.fsync(fd)
finally:
    os.close(fd)
PY
}

lib::_snapshot_regular_file_nofollow() {
  local src="${1-}" dst="${2-}"
  [[ -n "$src" && -n "$dst" ]] || lib::die "snapshot_regular_file_nofollow requires <src> <dst>"
  lib::_require_cmd python3
  python3 - "$src" "$dst" <<'PY'
import os
import stat
import sys

src, dst = sys.argv[1], sys.argv[2]
parent, base = os.path.split(src)
if not base:
    raise SystemExit(f"Source file path has no basename: {src}")
if not parent:
    parent = "."

dir_flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0)
file_flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)

dirfd = os.open(parent, dir_flags)
try:
    fd = os.open(base, file_flags, dir_fd=dirfd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SystemExit(f"Source file is not a regular file: {src}")
        with open(dst, 'wb') as outf:
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                outf.write(chunk)
            outf.flush()
            os.fsync(outf.fileno())
    finally:
        os.close(fd)
finally:
    os.close(dirfd)
PY
}

lib::_same_device_or_die() {
  local left="${1-}" right="${2-}"
  [[ -n "$left" && -n "$right" ]] || lib::die "Device check requires two paths"
  local left_dev right_dev
  left_dev="$(stat -c '%d' -- "$left")" || lib::die "stat failed: $left"
  right_dev="$(stat -c '%d' -- "$right")" || lib::die "stat failed: $right"
  [[ "$left_dev" == "$right_dev" ]] || lib::die "Paths must be on the same filesystem: $left vs $right"
}

lib::_pick_newest_backup() {
  local target="${1-}"
  local newest='' newest_mtime=''
  local candidate mtime
  local had_nullglob=0
  shopt -q nullglob && had_nullglob=1
  shopt -s nullglob
  for candidate in "$target".bak.*; do
    [[ -e "$candidate" ]] || continue
    [[ ! -L "$candidate" ]] || continue
    mtime="$(stat -c '%Y' -- "$candidate")" || continue
    if [[ -z "$newest" || "$mtime" -gt "$newest_mtime" ]]; then
      newest="$candidate"
      newest_mtime="$mtime"
    fi
  done
  if (( had_nullglob )); then
    shopt -s nullglob
  else
    shopt -u nullglob
  fi
  printf '%s\n' "$newest"
}

# ----------------------------
# Logging
# ----------------------------

lib::log() {
  (( TRITON_VERBOSITY >= 1 )) && printf '%s\n' "$*" >&2 || true
}

lib::verbose() {
  (( TRITON_VERBOSITY >= 2 )) && printf '%s\n' "$*" >&2 || true
}

lib::warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

lib::err() {
  printf 'ERROR: %s\n' "$*" >&2
}

lib::die() {
  lib::err "$@"
  exit 1
}

# ----------------------------
# Type checks and validation
# ----------------------------

lib::is_int() {
  [[ "${1-}" =~ ^[0-9]+$ ]]
}

lib::is_unsigned_decimal() {
  [[ "${1-}" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

# Compatibility alias: historically this meant "unsigned decimal string".
lib::is_num() {
  lib::is_unsigned_decimal "${1-}"
}

lib::require_var() {
  local name="${1-}"
  lib::_require_identifier "$name"
  local val="${!name-}"
  [[ -n "$val" ]] || lib::die "Required env var is empty or unset: $name"
}

lib::require_pos_int() {
  local name="${1-}"
  lib::_require_identifier "$name"
  local val="${!name-}"
  [[ -n "$val" ]] || lib::die "Required env var is empty or unset: $name"
  lib::is_int "$val" || lib::die "$name must be a positive integer, got: $val"
  (( val > 0 )) || lib::die "$name must be > 0, got: $val"
}

lib::require_bool() {
  local name="${1-}"
  lib::_require_identifier "$name"
  local val="${!name-}"
  case "${val,,}" in
    true|false|1|0|yes|no|on|off) return 0 ;;
    *) lib::die "$name must be true/false/1/0/yes/no/on/off, got: $val" ;;
  esac
}

lib::is_truthy() {
  local val="${1-}"
  case "${val,,}" in
    true|1|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

lib::normalize_bool_var() {
  local name="${1-}"
  local false_value="${2:-false}"
  local true_value="${3:-true}"
  lib::_require_identifier "$name"
  local val="${!name-}"
  case "${val,,}" in
    1|true|yes|on) printf -v "$name" '%s' "$true_value" ;;
    0|false|no|off|'') printf -v "$name" '%s' "$false_value" ;;
    *) lib::die "$name must be true/false/1/0/yes/no/on/off, got: ${!name-}" ;;
  esac
}

# ----------------------------
# Hashing and path helpers
# ----------------------------

lib::sha256_12() {
  lib::_require_cmd python3
  python3 -c "import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode('utf-8')).hexdigest()[:12])" "$1"
}

lib::slugify() {
  printf '%s' "${1-}" | tr -cs 'A-Za-z0-9._-' '-' | sed -e 's/^-*//' -e 's/-*$//'
}

lib::slugify_nonempty() {
  local out
  out="$(lib::slugify "${1-}")"
  [[ -n "$out" ]] || lib::die "Slugified value is empty"
  printf '%s\n' "$out"
}

lib::validate_model_name() {
  local name="${1-}"
  [[ -n "$name" ]] || lib::die "Model name is empty"
  [[ "$name" != '.' && "$name" != '..' ]] || lib::die "Model name must not be '.' or '..': $name"
  [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || lib::die "Model name must be a single safe ASCII path element: $name"
}

# ----------------------------
# Locking
# ----------------------------

lib::flock_open_or_die() {
  local __out_fd_var="${1-}"
  local lockfile="${2-}"
  local timeout="${3:-10}"
  local label="${4:-}"

  lib::_require_identifier "$__out_fd_var"
  [[ -n "$lockfile" ]] || lib::die "Lockfile path is empty"
  lib::is_int "$timeout" || lib::die "Lock timeout must be a non-negative integer, got: $timeout"
  lib::_require_cmd python3

  local parent
  parent="$(dirname -- "$lockfile")"
  [[ -d "$parent" ]] || lib::die "Lockfile parent directory must already exist: $parent"
  [[ ! -L "$parent" ]] || lib::die "Lockfile parent directory is a symlink: $parent"
  [[ -w "$parent" ]] || lib::die "Lockfile parent directory is not writable: $parent"

  local tmpdir status_file pid deadline status_line
  tmpdir="$(mktemp -d -t triton_lock_helper.XXXXXX)" || lib::die "mktemp failed for lock helper"
  status_file="$tmpdir/status"

  python3 - "$lockfile" "$timeout" "$status_file" <<'PY' &
import ctypes
import fcntl
import os
import signal
import stat
import sys
import time

lockfile, timeout_s, status_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
parent, base = os.path.split(lockfile)
if not base:
    msg = f"Lockfile path has no basename: {lockfile}"
    with open(status_path, 'w', encoding='utf-8') as fh:
        fh.write(f"ERROR:{msg}\n")
    raise SystemExit(1)
if not parent:
    parent = "."

def write_status(line: str) -> None:
    with open(status_path, 'w', encoding='utf-8') as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())

def set_pdeathsig() -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        PR_SET_PDEATHSIG = 1
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
    except Exception:
        pass

set_pdeathsig()

dir_flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0)
file_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0)

dirfd = None
fd = None
try:
    dirfd = os.open(parent, dir_flags)
    fd = os.open(base, file_flags, 0o600, dir_fd=dirfd)
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        raise RuntimeError(f"Lockfile is not a regular file: {lockfile}")

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Lock timeout after {timeout_s}s: {lockfile}")
            time.sleep(0.05)

    write_status("READY")

    def _stop(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while True:
        signal.pause()
except BaseException as exc:
    try:
        write_status(f"ERROR:{exc}")
    except Exception:
        pass
    raise
finally:
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
    if dirfd is not None:
        try:
            os.close(dirfd)
        except OSError:
            pass
PY
  pid=$!

  deadline=$((SECONDS + timeout + 2))
  while :; do
    if [[ -s "$status_file" ]]; then
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    (( SECONDS <= deadline )) || break
    sleep 0.05
  done

  if [[ ! -s "$status_file" ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    rm -rf -- "$tmpdir"
    lib::die "Lock helper did not report status for: ${label:-$lockfile}"
  fi

  IFS= read -r status_line <"$status_file" || status_line=''
  if [[ "$status_line" == READY ]]; then
    _LIB_LOCK_HELPER_DIRS["$pid"]="$tmpdir"
    eval "$__out_fd_var=\$pid"
    return 0
  fi

  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  rm -rf -- "$tmpdir"
  status_line="${status_line#ERROR:}"
  [[ -n "$status_line" ]] || status_line="Failed to acquire lock: ${label:-$lockfile}"
  lib::die "$status_line"
}

lib::flock_close() {
  local fd="${1-}"
  lib::is_int "$fd" || lib::die "flock_close requires a numeric lock handle, got: $fd"

  if [[ -n "${_LIB_LOCK_HELPER_DIRS[$fd]+x}" ]]; then
    local tmpdir
    tmpdir="${_LIB_LOCK_HELPER_DIRS[$fd]}"
    kill "$fd" 2>/dev/null || true
    wait "$fd" 2>/dev/null || true
    rm -rf -- "$tmpdir"
    unset '_LIB_LOCK_HELPER_DIRS[$fd]'
    return 0
  fi

  eval "exec ${fd}>&-"
}

# Compatibility helper for callers that want one ambient lock handle.
lib::flock_or_die() {
  LIB_FLOCK_FD=''
  lib::flock_open_or_die LIB_FLOCK_FD "$@"
}

# ----------------------------
# Env file I/O
# ----------------------------

lib::write_env_atomically() {
  local env_file="${1-}"
  shift || true
  [[ -n "$env_file" ]] || lib::die "Env file path is empty"

  lib::_prepare_parent_dir "$env_file"
  if [[ -e "$env_file" || -L "$env_file" ]]; then
    [[ ! -L "$env_file" ]] || lib::die "Env file is a symlink: $env_file"
    [[ -f "$env_file" ]] || lib::die "Env file is not a regular file: $env_file"
  fi

  local env_dir env_base old_umask tmp
  env_dir="$(dirname -- "$env_file")"
  env_base="$(basename -- "$env_file")"

  old_umask="$(umask)"
  umask 077
  tmp="$(mktemp --tmpdir="$env_dir" ".${env_base}.tmp.XXXXXX")" || {
    umask "$old_umask"
    lib::die "mktemp failed for env file: $env_file"
  }
  umask "$old_umask"

  {
    printf '%s\n' '# generated by triton library'
    local pair key val
    for pair in "$@"; do
      [[ "$pair" == *=* ]] || {
        rm -f -- "$tmp"
        lib::die "Env assignment must be KEY=VALUE, got: $pair"
      }
      key="${pair%%=*}"
      val="${pair#*=}"
      lib::_validate_identifier "$key" || {
        rm -f -- "$tmp"
        lib::die "Invalid env key: $key"
      }
      printf 'export %s=%q\n' "$key" "$val"
    done
  } >"$tmp" || {
    rm -f -- "$tmp"
    lib::die "Failed to write temp env file: $tmp"
  }

  chmod 600 -- "$tmp" 2>/dev/null || true
  lib::_fsync_file "$tmp" || {
    rm -f -- "$tmp"
    lib::die "fsync failed for temp env file: $tmp"
  }
  mv -f -- "$tmp" "$env_file" || {
    rm -f -- "$tmp"
    lib::die "Failed to move temp env file into place: $env_file"
  }
  lib::_fsync_dir "$env_dir" || lib::die "fsync failed for env directory: $env_dir"
}

lib::load_env_safe() {
  local f="${1-}"
  [[ -n "$f" ]] || lib::die "Env file path is empty"

  local tmp had_a had_u rc source_rc
  tmp="$(mktemp -t triton_env_exports.XXXXXX)" || lib::die "mktemp failed for safe env loading"

  rc=0
  python3 - "$f" >"$tmp" <<'PY' || rc=$?
import os
import re
import shlex
import stat
import sys

path = sys.argv[1]
parent, base = os.path.split(path)
if not base:
    raise SystemExit(f"Env file path has no basename: {path}")
if not parent:
    parent = "."

dir_flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0)
file_flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
line_re = re.compile(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$')
out = []


def parse_ansi_c_word(text: str):
    if not text.startswith("$'"):
        raise ValueError('not ANSI-C quoted')
    i = 2
    buf = []
    while i < len(text):
        c = text[i]
        if c == "'":
            return ''.join(buf), text[i + 1:]
        if c != '\\':
            buf.append(c)
            i += 1
            continue
        i += 1
        if i >= len(text):
            raise ValueError('unterminated ANSI-C escape')
        e = text[i]
        if e == 'n':
            buf.append('\n')
            i += 1
        elif e == 'r':
            buf.append('\r')
            i += 1
        elif e == 't':
            buf.append('\t')
            i += 1
        elif e == 'a':
            buf.append('\a')
            i += 1
        elif e == 'b':
            buf.append('\b')
            i += 1
        elif e == 'f':
            buf.append('\f')
            i += 1
        elif e == 'v':
            buf.append('\v')
            i += 1
        elif e in ('\\', "'", '"'):
            buf.append(e)
            i += 1
        elif e == 'x':
            hexchars = text[i + 1:i + 3]
            if len(hexchars) != 2 or any(ch not in '0123456789abcdefABCDEF' for ch in hexchars):
                raise ValueError('bad \\x escape')
            buf.append(chr(int(hexchars, 16)))
            i += 3
        elif e in '01234567':
            octchars = e
            j = i + 1
            while j < len(text) and len(octchars) < 3 and text[j] in '01234567':
                octchars += text[j]
                j += 1
            buf.append(chr(int(octchars, 8)))
            i = j
        else:
            buf.append(e)
            i += 1
    raise ValueError('unterminated ANSI-C quoted string')


dirfd = os.open(parent, dir_flags)
try:
    fd = os.open(base, file_flags, dir_fd=dirfd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SystemExit(f"Env file is not a regular file: {path}")
        with os.fdopen(fd, 'r', encoding='utf-8') as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.rstrip('\n')
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                m = line_re.match(line)
                if not m:
                    raise SystemExit(f"Unsupported env line at {lineno}: {line}")
                key, rest = m.group(1), m.group(2)
                rest = rest.strip()
                if rest == '' or rest.startswith('#'):
                    val = ''
                elif rest.startswith("$'"):
                    try:
                        val, tail = parse_ansi_c_word(rest)
                    except ValueError as exc:
                        raise SystemExit(f"Unsupported ANSI-C env value at {lineno}: {exc}: {line}")
                    tail = tail.strip()
                    if tail and not tail.startswith('#'):
                        raise SystemExit(f"Unexpected trailing chars at {lineno}: {line}")
                else:
                    lexer = shlex.shlex(rest, posix=True)
                    lexer.whitespace_split = True
                    lexer.commenters = '#'
                    tokens = list(lexer)
                    if len(tokens) != 1:
                        raise SystemExit(f"Unsupported env value at {lineno}: {line}")
                    val = tokens[0]
                out.append(f"export {key}={shlex.quote(val)}")
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
finally:
    os.close(dirfd)

sys.stdout.write("\n".join(out) + ("\n" if out else ""))
PY

  if (( rc != 0 )); then
    rm -f -- "$tmp"
    return "$rc"
  fi

  had_a=0
  had_u=0
  [[ $- == *a* ]] && had_a=1
  [[ $- == *u* ]] && had_u=1
  set -a
  set +u
  # shellcheck disable=SC1090
  source "$tmp"
  source_rc=$?
  lib::_restore_shell_flags "$had_a" "$had_u"
  rm -f -- "$tmp"
  return "$source_rc"
}

lib::load_env_trusted() {
  local f="${1-}"
  [[ -n "$f" ]] || lib::die "Env file path is empty"

  local tmp had_a had_u rc
  tmp="$(mktemp -t triton_env_trusted.XXXXXX)" || lib::die "mktemp failed for trusted env loading"
  chmod 600 -- "$tmp" 2>/dev/null || true
  lib::_snapshot_regular_file_nofollow "$f" "$tmp" || {
    rm -f -- "$tmp"
    lib::die "Failed to open trusted env file without following symlinks: $f"
  }

  had_a=0
  had_u=0
  [[ $- == *a* ]] && had_a=1
  [[ $- == *u* ]] && had_u=1
  set -a
  set +u
  # shellcheck disable=SC1090
  source "$tmp"
  rc=$?
  lib::_restore_shell_flags "$had_a" "$had_u"
  rm -f -- "$tmp"
  return "$rc"
}

# ----------------------------
# Recoverable directory replacement
# ----------------------------

lib::restore_backup() {
  local target="${1-}"
  [[ -n "$target" ]] || lib::die "Target path is empty"

  if [[ -e "$target" || -L "$target" ]]; then
    [[ ! -L "$target" ]] || lib::die "Target path is a symlink: $target"
    return 0
  fi

  local newest parent
  newest="$(lib::_pick_newest_backup "$target")"
  if [[ -n "$newest" ]]; then
    parent="$(dirname -- "$target")"
    lib::warn "Target missing; restoring backup $(basename -- "$newest")"
    mv -T -- "$newest" "$target" || lib::die "Failed to restore backup: $newest -> $target"
    lib::_fsync_dir "$parent" || lib::die "fsync failed for directory: $parent"
  fi
}

lib::replace_dir_with_backup() {
  local staged="${1-}"
  local target="${2-}"
  [[ -n "$staged" && -n "$target" ]] || lib::die "replace_dir_with_backup requires <staged> <target>"
  [[ -d "$staged" ]] || lib::die "Staged path is not a directory: $staged"
  [[ ! -L "$staged" ]] || lib::die "Staged path is a symlink: $staged"

  local target_parent staged_parent backup target_base
  target_parent="$(dirname -- "$target")"
  staged_parent="$(dirname -- "$staged")"
  target_base="$(basename -- "$target")"

  lib::_prepare_parent_dir "$target"
  [[ ! -L "$target_parent" ]] || lib::die "Target parent directory is a symlink: $target_parent"
  [[ -d "$staged_parent" ]] || lib::die "Staged parent directory is missing: $staged_parent"
  [[ ! -L "$staged_parent" ]] || lib::die "Staged parent directory is a symlink: $staged_parent"

  lib::_same_device_or_die "$staged_parent" "$target_parent"

  if [[ -e "$target" || -L "$target" ]]; then
    [[ ! -L "$target" ]] || lib::die "Target path is a symlink: $target"
    [[ -d "$target" ]] || lib::die "Target path is not a directory: $target"
  fi

  lib::restore_backup "$target"

  backup="${target_parent}/${target_base}.bak.${RANDOM}${RANDOM}"
  if [[ -e "$target" ]]; then
    mv -T -- "$target" "$backup" || lib::die "Failed to move target to backup: $target"
    lib::_fsync_dir "$target_parent" || lib::die "fsync failed for directory: $target_parent"
  fi

  if mv -T -- "$staged" "$target"; then
    lib::_fsync_dir "$target_parent" || lib::die "fsync failed for directory: $target_parent"
    if [[ -e "$backup" ]]; then
      rm -rf -- "$backup" || lib::warn "Failed to remove backup directory: $backup"
      lib::_fsync_dir "$target_parent" || lib::warn "fsync failed after backup removal: $target_parent"
    fi
    return 0
  fi

  if [[ -e "$backup" ]]; then
    if ! mv -T -- "$backup" "$target"; then
      lib::die "Rollback failed after replacement error: $backup -> $target"
    fi
    lib::_fsync_dir "$target_parent" || lib::die "fsync failed after rollback in directory: $target_parent"
  fi
  return 1
}

# Compatibility alias retained for existing callers. This is recoverable directory
# replacement, not a single-step atomic swap when a prior target exists.
lib::atomic_swap_dir() {
  lib::replace_dir_with_backup "$@"
}

# ----------------------------
# Logging command output to a file
# ----------------------------

lib::run_to_log() {
  local log="${1-}"
  shift || true
  [[ -n "$log" ]] || lib::die "Log path is empty"
  (($# > 0)) || lib::die "run_to_log requires a command"

  lib::_prepare_parent_dir "$log"
  if [[ -e "$log" || -L "$log" ]]; then
    [[ ! -L "$log" ]] || lib::die "Log path is a symlink: $log"
    [[ -f "$log" ]] || lib::die "Log path is not a regular file: $log"
  fi

  local log_dir log_base old_umask tmp rc
  log_dir="$(dirname -- "$log")"
  log_base="$(basename -- "$log")"

  old_umask="$(umask)"
  umask 077
  tmp="$(mktemp --tmpdir="$log_dir" ".${log_base}.tmp.XXXXXX")" || {
    umask "$old_umask"
    lib::die "mktemp failed for log file: $log"
  }
  umask "$old_umask"

  rc=0
  "$@" >"$tmp" 2>&1 || rc=$?

  chmod 600 -- "$tmp" 2>/dev/null || true
  lib::_fsync_file "$tmp" || {
    rm -f -- "$tmp"
    lib::die "fsync failed for temp log file: $tmp"
  }
  mv -f -- "$tmp" "$log" || {
    rm -f -- "$tmp"
    lib::die "Failed to move temp log into place: $log"
  }
  lib::_fsync_dir "$log_dir" || lib::die "fsync failed for log directory: $log_dir"
  return "$rc"
}
