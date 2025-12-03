#!/bin/bash

: << 'COMMENT_BLOCK'
cd /home/scuee_user06/myh/电池/test-7.3/累计放电量/调参用

tmux new -s myh_search

tmux attach -t my_search


添加权限
chmod +x run_search.sh
执行搜索
./run_search.sh cyclenet.py
退出
按下组合键 Ctrl + b，然后松开，再按一下 d 键。
COMMENT_BLOCK

# --- 1. 接收并检查参数 ---
# 检查用户是否提供了第一个参数（即Python脚本的文件名）
if [ -z "$1" ]; then
    # 如果 $1 为空（-z 表示 zero-length），则打印错误信息
    echo "错误：缺少要执行的 Python 脚本名。"
    echo "用法: ./run_search.sh [你的python脚本.py]"
    # 退出脚本，返回错误码 1
    exit 1
fi

# 将第一个参数赋值给一个可读性更好的变量（可选，但推荐）
PYTHON_SCRIPT_TO_RUN=$1
echo "准备使用脚本 '${PYTHON_SCRIPT_TO_RUN}' 进行超参数搜索..."

# 定义要搜索的超参数范围
# 你可以根据需要修改这些数组
D_MODELS=(128 256 512)
D_FFS=(256 512 1024)
DROPOUTS=(0.1 0.2 0.3)
WEIGHT_DECAYS=(1e-4 1e-5 5e-5)
BATCH_SIZES=(64 128 256 512)
LEARNING_RATES=(0.001 0.002 0.005)
PATIENCES=(15 25)

# GPU数量
NUM_GPUS=3
# 计数器，用于分配GPU和控制并行任务数量
COUNTER=0

# 遍历所有超参数组合 (Grid Search)
for D_MODEL in "${D_MODELS[@]}"; do
for D_FF in "${D_FFS[@]}"; do
for DROPOUT in "${DROPOUTS[@]}"; do
for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
for PATIENCE in "${PATIENCES[@]}"; do

    # 计算当前任务应该在哪块GPU上运行
    GPU_ID=$((COUNTER % NUM_GPUS))

    echo "==========================================================="
    echo "启动任务 ${COUNTER} 在 GPU ${GPU_ID} 上"
    echo "参数: --d_model=${D_MODEL} --d_ff=${D_FF} --dropout=${DROPOUT} --weight_decay=${WEIGHT_DECAY} --batch_size=${BATCH_SIZE} --learning_rate=${LEARNING_RATE} --patience=${PATIENCE}"

    # 使用 CUDA_VISIBLE_DEVICES 来指定GPU，并在后台运行python脚本 (&)
    # 将写死的 'cyclenet3.3-forcyclenum.py' 替换为我们从参数中获取的变量
    CUDA_VISIBLE_DEVICES=$GPU_ID python ${PYTHON_SCRIPT_TO_RUN} \
        --d_model=${D_MODEL} \
        --d_ff=${D_FF} \
        --dropout=${DROPOUT} \
        --weight_decay=${WEIGHT_DECAY} \
        --batch_size=${BATCH_SIZE} \
        --learning_rate=${LEARNING_RATE} \
        --patience=${PATIENCE} > "log_gpu_${GPU_ID}_task_${COUNTER}.txt" 2>&1 &

    COUNTER=$((COUNTER + 1))

    if (( COUNTER % NUM_GPUS == 0 )); then
        echo "--------- 等待当前批次的 ${NUM_GPUS} 个任务完成... ---------"
        wait
        echo "--------- 当前批次任务已完成，继续... ---------"
    fi

done
done
done
done
done
done
done

# 等待最后一批任务完成
echo "--------- 等待最后一批任务完成... ---------"
wait

echo "所有超参数搜索任务已完成！"
