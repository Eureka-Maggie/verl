#!/bin/bash
cd /primus_xpfs_workspace_T04/txy/projects/verl
source /primus_xpfs_workspace_T04/txy/venv/verl/bin/activate

while true; do
    # 等 GPU 空闲
    echo "监控中，等待 GPU 进程释放..."
    while true; do
        count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')
        if [ "$count" -eq 0 ]; then
            echo ""
            echo "空了！！！！ 时间: $(date)"
            break
        fi
        echo -ne "\r占用进程数: $count    "
        sleep 3
    done

    # 启动训练，等它结束（正常结束或崩溃）
    echo "启动训练... 时间: $(date)"
    bash /primus_xpfs_workspace_T04/txy/projects/verl/examples/grpo_trainer/run_qwen3-4b_grpo_dy.sh
    echo ""
    echo "训练进程退出，退出码=$?，时间: $(date)"
    echo "等待 10s 后重新检测 GPU 状态..."
    sleep 10
done
