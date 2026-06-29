#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

if [[ "${TRACE-0}" == "1" ]]; then
  set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
  echo 'usage: sync.sh [-h]

Sync the local workspace to remote and the remote logs to local (ignoring logs
that are newer on the receiver).
'
  exit 1
fi

# Get the absolute path to the script's directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find repo root by looking for pyproject.toml starting from script_dir
repo_root="$script_dir"
while [ "$(find "$repo_root" -maxdepth 1 -name pyproject.toml | wc -l)" -ne 1 ]; do
  repo_root="$(dirname "$repo_root")"
  # Prevent infinite loop if pyproject.toml not found
  if [ "$repo_root" = "/" ]; then
    echo "[error] Could not find pyproject.toml. Are you running this from the verl repository?"
    exit 1
  fi
done

main() {
  logs_exclude_patterns=("/debug/" "/slurm/" "/tests/")
  workspace_exclude_patterns=(
    ".env"
    ".cache"
    ".venv"
    ".venv-grpo"
    ".pytest_cache"
    ".vscode"
    "__pycache__"
    "/data/"
    "/libs/"
    "/models/"
    "/logs/"
    "/wandb/"
    "*.db"
    "/notebooks/"
    "/plots/"
    "tests/"
    "/arch_outputs/"
  )

  # Read remotes from configuration file
  config_file="$repo_root/configs/sync.conf"
  if [ ! -f "$config_file" ]; then
    echo "[error] Configuration file not found at $config_file"
    exit 1
  fi

  # Sync local workspace to each remote
  workspace_exclude_opts=()
  for pattern in "${workspace_exclude_patterns[@]}"; do
    workspace_exclude_opts+=("--exclude" "$pattern")
  done
  while IFS= read -r remote || [ -n "$remote" ]; do
    # Skip empty lines and comments
    [[ -z "$remote" || "$remote" =~ ^[[:space:]]*# ]] && continue
    echo "[info] Syncing $repo_root to $remote..."
    rsync -azhv "${workspace_exclude_opts[@]}" "$repo_root/" "$remote/"
  done <"$config_file"

  # Sync remote logs to local
  logs_exclude_opts=()
  for pattern in "${logs_exclude_patterns[@]}"; do
    logs_exclude_opts+=("--exclude" "$pattern")
  done
  while IFS= read -r remote || [ -n "$remote" ]; do
    # Skip empty lines and comments
    [[ -z "$remote" || "$remote" =~ ^[[:space:]]*# ]] && continue
    echo "[info] Syncing $remote/logs/ to $repo_root/logs ..."
    rsync --update -azhv "${logs_exclude_opts[@]}" "$remote/logs/" "$repo_root/logs/"
  done <"$config_file"
}

main "$@"
