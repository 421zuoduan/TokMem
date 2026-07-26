#!/bin/sh
set -eu

umask 077

package_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
bin_dir="$package_dir/artifacts/rootless-docker/bin"
helper_dir="$package_dir/rootless_helpers"
runtime_home="$package_dir/artifacts/rootless-docker/home"
data_root="$package_dir/artifacts/rootless-docker/singleuid-data"
runtime_dir=${TOKMEM_ROOTLESS_RUNTIME_DIR:-"/tmp/tokmem-intercode-rootless-$(id -u)"}
docker_socket="$runtime_dir/docker.sock"
launcher_pid_file="$runtime_dir/rootlesskit.pid"
daemon_pid_file="$runtime_dir/dockerd.pid"
daemon_log="$runtime_dir/dockerd.log"
docker_cli="$bin_dir/docker"

usage() {
    echo "usage: $0 doctor|start|foreground|status|env|smoke|stop" >&2
    exit 2
}

ensure_layout() {
    mkdir -p "$runtime_dir" "$runtime_home" "$data_root"
    chmod 700 "$runtime_dir" "$runtime_home" "$data_root"
}

docker_command() {
    DOCKER_HOST="unix://$docker_socket" "$docker_cli" "$@"
}

daemon_is_ready() {
    [ -S "$docker_socket" ] &&
        docker_command info >/dev/null 2>&1
}

doctor() {
    if [ "$(id -u)" -eq 0 ]; then
        echo "refusing to run the single-UID fallback as root" >&2
        exit 1
    fi
    for executable in \
        dockerd dockerd-rootless.sh docker containerd runc rootlesskit vpnkit
    do
        if [ ! -x "$bin_dir/$executable" ]; then
            echo "missing executable: $bin_dir/$executable" >&2
            exit 1
        fi
    done
    for executable in newuidmap newgidmap runc_singleuid.py
    do
        if [ ! -x "$helper_dir/$executable" ]; then
            echo "missing helper: $helper_dir/$executable" >&2
            exit 1
        fi
    done
    download_dir="$package_dir/artifacts/rootless-docker/downloads"
    if [ ! -f "$download_dir/SHA256SUMS" ]; then
        echo "missing download checksum manifest" >&2
        exit 1
    fi
    (
        cd "$download_dir"
        sha256sum --check --status SHA256SUMS
    ) || {
        echo "rootless Docker download checksum mismatch" >&2
        exit 1
    }
    if ! unshare -Ur true; then
        echo "unprivileged user namespaces are unavailable" >&2
        exit 1
    fi
    ensure_layout
    echo "ordinary_user_uid=$(id -u)"
    echo "runtime_dir=$runtime_dir"
    echo "data_root=$data_root"
    df -Pk "$data_root" | tail -n 1
}

start() {
    doctor
    if daemon_is_ready; then
        echo "rootless Docker is already running"
        status
        return
    fi
    if [ -f "$launcher_pid_file" ]; then
        stale_pid=$(cat "$launcher_pid_file")
        if kill -0 "$stale_pid" 2>/dev/null; then
            echo "managed launcher PID $stale_pid exists but Docker is not ready" >&2
            exit 1
        fi
        rm -f "$launcher_pid_file"
    fi

    nohup env \
        PATH="$helper_dir:$bin_dir:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        HOME="$runtime_home" \
        XDG_RUNTIME_DIR="$runtime_dir" \
        DOCKERD_ROOTLESS_ROOTLESSKIT_NET=vpnkit \
        DOCKERD_ROOTLESS_ROOTLESSKIT_DISABLE_HOST_LOOPBACK=true \
        "$bin_dir/dockerd-rootless.sh" \
        --host="unix://$docker_socket" \
        --group=root \
        --data-root="$data_root" \
        --exec-root="$runtime_dir/docker-exec" \
        --pidfile="$daemon_pid_file" \
        --storage-driver=vfs \
        --add-runtime="singleuid=$helper_dir/runc_singleuid.py" \
        --default-runtime=singleuid \
        --log-level=info \
        >"$daemon_log" 2>&1 </dev/null &
    launcher_pid=$!
    echo "$launcher_pid" >"$launcher_pid_file"

    attempts=0
    while ! daemon_is_ready; do
        attempts=$((attempts + 1))
        if ! kill -0 "$launcher_pid" 2>/dev/null; then
            echo "rootless Docker exited during startup; log follows" >&2
            tail -n 80 "$daemon_log" >&2
            exit 1
        fi
        if [ "$attempts" -ge 30 ]; then
            echo "rootless Docker did not become ready in 30 seconds" >&2
            tail -n 80 "$daemon_log" >&2
            exit 1
        fi
        sleep 1
    done
    status
}

foreground() {
    doctor
    if daemon_is_ready; then
        echo "rootless Docker is already running" >&2
        exit 1
    fi
    exec env \
        PATH="$helper_dir:$bin_dir:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        HOME="$runtime_home" \
        XDG_RUNTIME_DIR="$runtime_dir" \
        DOCKERD_ROOTLESS_ROOTLESSKIT_NET=vpnkit \
        DOCKERD_ROOTLESS_ROOTLESSKIT_DISABLE_HOST_LOOPBACK=true \
        "$bin_dir/dockerd-rootless.sh" \
        --host="unix://$docker_socket" \
        --group=root \
        --data-root="$data_root" \
        --exec-root="$runtime_dir/docker-exec" \
        --pidfile="$daemon_pid_file" \
        --storage-driver=vfs \
        --add-runtime="singleuid=$helper_dir/runc_singleuid.py" \
        --default-runtime=singleuid \
        --log-level=info
}

status() {
    if ! daemon_is_ready; then
        echo "rootless Docker is not ready at unix://$docker_socket" >&2
        exit 1
    fi
    docker_command version --format 'server_version={{.Server.Version}}'
    docker_command info --format \
        'driver={{.Driver}} default_runtime={{.DefaultRuntime}} security={{json .SecurityOptions}}'
    echo "docker_host=unix://$docker_socket"
}

print_env() {
    echo "export DOCKER_HOST=unix://$docker_socket"
    echo "export PATH=$bin_dir:\$PATH"
}

smoke() {
    status
    for fs_number in 1 2 3 4
    do
        image="intercode-nl2bash-fs$fs_number:latest"
        docker_command image inspect "$image" >/dev/null
        docker_command run --rm "$image" /bin/bash -lc \
            'set -e; test -z "$(git status --porcelain)"; command -v bash python3 git find jq tree >/dev/null; printf "fs%s smoke-ok\n" "$file_system_version"'
    done
}

stop() {
    if [ ! -f "$launcher_pid_file" ]; then
        echo "no managed launcher PID file: $launcher_pid_file" >&2
        exit 1
    fi
    launcher_pid=$(cat "$launcher_pid_file")
    case "$launcher_pid" in
        *[!0-9]*|'')
            echo "invalid launcher PID: $launcher_pid" >&2
            exit 1
            ;;
    esac
    if ! kill -0 "$launcher_pid" 2>/dev/null; then
        rm -f "$launcher_pid_file"
        echo "managed rootless Docker is already stopped"
        return
    fi
    owner_uid=$(stat -c %u "/proc/$launcher_pid")
    command_line=$(tr '\000' ' ' <"/proc/$launcher_pid/cmdline")
    case "$command_line" in
        *dockerd-rootless.sh*"$data_root"*) ;;
        *)
            echo "refusing to stop unexpected PID $launcher_pid: $command_line" >&2
            exit 1
            ;;
    esac
    if [ "$owner_uid" -ne "$(id -u)" ]; then
        echo "refusing to stop PID owned by UID $owner_uid" >&2
        exit 1
    fi
    kill -TERM "$launcher_pid"
    attempts=0
    while kill -0 "$launcher_pid" 2>/dev/null; do
        attempts=$((attempts + 1))
        if [ "$attempts" -ge 30 ]; then
            echo "launcher did not exit in 30 seconds; refusing to force-kill it" >&2
            exit 1
        fi
        sleep 1
    done
    rm -f "$launcher_pid_file"
    echo "managed rootless Docker stopped"
}

case "${1:-}" in
    doctor) doctor ;;
    start) start ;;
    foreground) foreground ;;
    status) status ;;
    env) print_env ;;
    smoke) smoke ;;
    stop) stop ;;
    *) usage ;;
esac
