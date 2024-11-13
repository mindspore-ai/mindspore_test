#!/bin/bash

cd tests/st/silent_check
MS_ALLOC_CONF="enable_vmm:False" NPU_ASD_ENABLE=3 ENABLE_SILENT_CHECK_V3=1 VLOG_v=(50001,) bash singlerun_silent_check.sh silent_check_last_grad2.py
