#!/bin/bash
rm -rf ./*.log
CURDIR=$(pwd)

HOST_IP=$(hostname -I | awk '{print $1}')
rank_table_file="/home/workspace/mindspore_config/hccl/rank_table_8p.json"
DEVICE_IP=$(cat $rank_table_file | grep -E "device_ip" | head -1 | awk '{print $2}' | awk -F "[\"\"]" '{print $2}')
echo "host ip is: $HOST_IP"
echo "device_ip is: $DEVICE_IP"

RANK_TABLE_FILE=${CURDIR}/rank_table_file.json
ESCLUSTER_CONFIG_PATH=${CURDIR}/escluster_config.json
echo $RANK_TABLE_FILE
echo $ESCLUSTER_CONFIG_PATH

sed -i "s/10.155.170.21/${HOST_IP}/g" ${RANK_TABLE_FILE} ${ESCLUSTER_CONFIG_PATH}
sed -i "s/192.168.100.101/${DEVICE_IP}/g" ${RANK_TABLE_FILE} ${ESCLUSTER_CONFIG_PATH}

export RANK_TABLE_FILE
export ESCLUSTER_CONFIG_PATH
export MS_DISABLE_REF_MODE=1
export JOB_ID=10087
export MS_DEV_JIT_SYNTAX_LEVEL=0

msrun --worker_num=1 --local_worker_num=1 --join=True python ${CURDIR}/test_es_external_api.py

result=$?
if [ $result != 0 ]; then
    echo "msrun test_es_external_api fail!"
    exit 1
fi
echo "msrun test_es_external_api succ!"
exit 0
