#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WATCHDOG_LOG="${SCRIPT_DIR}/logs/watchdog_main_tokmem_fixed_split.log"
TARGET_SESSION="tokmem_atomic_run"
TARGET_COMMAND="python -u ${SCRIPT_DIR}/main_in_domain_fixed_split.py"
CHECK_INTERVAL_SECONDS=300

mkdir -p "${SCRIPT_DIR}/logs"
touch "${WATCHDOG_LOG}"

exec >> "${WATCHDOG_LOG}" 2>&1

echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') watchdog started ====="
echo "Target tmux session: ${TARGET_SESSION}"
echo "Target command: ${TARGET_COMMAND}"

while true; do
    timestamp="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    active_lines="$(pgrep -af "${TARGET_COMMAND}" || true)"
    latest_training_log="$(ls -1t "${SCRIPT_DIR}"/logs/training_*.log 2>/dev/null | head -n 1 || true)"

    if [ -n "${latest_training_log}" ] && rg -q "Task learning pipeline completed!" "${latest_training_log}"; then
        echo "[${timestamp}] target run completed; no relaunch needed"
    elif [ -n "${active_lines}" ]; then
        echo "[${timestamp}] target run active"
        echo "${active_lines}"
    else
        echo "[${timestamp}] target run missing and not completed; relaunching in detached tmux"
        if tmux has-session -t "${TARGET_SESSION}" 2>/dev/null; then
            tmux kill-session -t "${TARGET_SESSION}" 2>/dev/null || true
        fi
        tmux new-session -d -s "${TARGET_SESSION}" "bash ${SCRIPT_DIR}/main_tokmem_fixed_split.sh"
        echo "[${timestamp}] relaunched ${TARGET_SESSION}"
    fi

    sleep "${CHECK_INTERVAL_SECONDS}"
done
