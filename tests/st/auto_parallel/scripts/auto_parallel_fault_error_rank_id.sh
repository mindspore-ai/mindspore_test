#!/bin/bash
rm -rf device
mkdir device

# start a scheduler process
export MS_WORKER_NUM=8
export MS_SCHED_HOST=127.0.0.1
export MS_SCHED_PORT=8118
export MS_ROLE=MS_SCHED
python ./auto_parallel_fault_error_rank_id_net.py > device/scheduler.log 2>&1 &

# start all worker processes
for ((i=0;i<8;i++));
do
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=127.0.0.1
    export MS_SCHED_PORT=8118
    export MS_ROLE=MS_WORKER
    export MS_NODE_ID=$i
    python ./auto_parallel_fault_error_rank_id_net.py > device/worker_$i.log 2>&1 &
done

wait
status=`echo $?`
if [ "${status}" != "0" ]; then
  echo "[ERROR] test_error_rank_id failed, failed to wait all processes end, status: ${status}."
  exit 1
fi

rm -rf MNIST_Data.zip mnist

exit 0
