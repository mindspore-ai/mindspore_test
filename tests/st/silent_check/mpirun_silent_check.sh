#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash $0"
echo "=============================================================================================================="

LOG_PATH="ascend_log"
export ASCEND_PROCESS_LOG_PATH=`pwd`/${LOG_PATH}
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_GLOBAL_EVENT_ENABLE=0

export ASCEND_RT_VISIABLE_DEVICES=0,1,2,3
export MS_DEV_DUMP_IR_PASSES="insert_silent_check_v2,graph_build,print_insert_placeholder_for_tensor_name"

export HCCL_EXEC_TIMEOUT=8

# NOTE: environment variable `NPU_ASD_ENABLE` is set in python script

rm -rf ${LOG_PATH}
rm -rf ms_graphs
rm -rf log_output

dir_name=$(dirname $0)
abs_path=$(cd $dir_name; pwd)

mpirun -n 4 --output-filename log_output --merge-stderr-to-stdout python ${abs_path}/silent_check.py
