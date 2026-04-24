#!/bin/bash
# Watchdog wrapper: 自动重启 run_qwen3-4b_grpo.sh，直到收到停止信号

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/run_qwen3-4b_grpo.sh"

STOP_FILE="/tmp/stop_grpo_watchdog"
PID_FILE="/tmp/grpo_watchdog.pid"
LOG_FILE="/primus_xpfs_workspace_T04/txy/projects/verl/watchdog_grpo.log"

RETRY_INTERVAL=5   # 重启前等待秒数
CHILD_PID=""

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

cleanup() {
    log "收到停止信号，正在退出..."
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        log "终止子进程 PID=$CHILD_PID"
        kill -TERM "$CHILD_PID"
        wait "$CHILD_PID" 2>/dev/null
    fi
    rm -f "$PID_FILE"
    log "Watchdog 已退出。"
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

# 写入自身 PID，方便外部 kill
echo $$ > "$PID_FILE"
log "Watchdog 启动，PID=$$，PID 文件：$PID_FILE"
log "停止方法："
log "  方法1（推荐）: touch ${STOP_FILE}  然后等当前轮次结束"
log "  方法2（立即）: kill \$(cat ${PID_FILE})  或  kill $$"
log "  方法3（立即）: kill -9 \$(cat ${PID_FILE})"

# 清除可能遗留的停止文件
rm -f "$STOP_FILE"

RUN_COUNT=0

while true; do
    # 检查停止文件
    if [ -f "$STOP_FILE" ]; then
        log "检测到停止文件 ${STOP_FILE}，不再重启，退出。"
        rm -f "$STOP_FILE" "$PID_FILE"
        exit 0
    fi

    RUN_COUNT=$((RUN_COUNT + 1))
    log "===== 第 ${RUN_COUNT} 次启动 ====="

    bash "$TARGET_SCRIPT" &
    CHILD_PID=$!
    log "子进程启动，PID=$CHILD_PID"

    wait "$CHILD_PID"
    EXIT_CODE=$?
    CHILD_PID=""

    log "子进程退出，退出码=$EXIT_CODE"

    # 再次检查停止文件（子进程运行期间可能放入）
    if [ -f "$STOP_FILE" ]; then
        log "检测到停止文件 ${STOP_FILE}，不再重启，退出。"
        rm -f "$STOP_FILE" "$PID_FILE"
        exit 0
    fi

    log "等待 ${RETRY_INTERVAL} 秒后重启..."
    sleep "$RETRY_INTERVAL"
done
