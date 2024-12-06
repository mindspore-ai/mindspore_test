#!/bin/bash

script_path=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
ms_path=$(realpath "${script_path}/../../")
echo "The ms project is located in: ${ms_path}"
ascendc_path=${ms_path}/mindspore/ops/kernel/ascend/ascendc
op_host_ori_path=${ascendc_path}/op_host
op_kernel_ori_path=${ascendc_path}/op_kernel
custom_compiler_path=${ms_path}/mindspore/python/mindspore/custom_compiler
echo "Custom compiler path: ${custom_compiler_path}"

ws_path=${ms_path}/custom_workspace
if [ ! -d "${ws_path}" ]; then
  mkdir -p "${ws_path}"
fi

cp ${op_host_ori_path} ${ws_path} -r
cp ${op_kernel_ori_path} ${ws_path} -r
cp ${custom_compiler_path} ${ws_path} -r
op_host_path=${ws_path}/op_host
op_kernel_path=${ws_path}/op_kernel
tmp_compiler_path=${ws_path}/custom_compiler

cd ${tmp_compiler_path} || exit
sed -i '/mindspore/ s/^/#/' setup.py
sed -i 's/\blogger.info\b/print/g' setup.py
sed -i 's/\blogger.error\b/print/g' setup.py
sed -i 's/\blogger.warning\b/print/g' setup.py

vendor_name="custom_ascendc_910b"
if [ ! -d "${ascendc_path}/${vendor_name}" ]; then
  python setup.py -o ${op_host_path} -k ${op_kernel_path} \
    --soc_version="ascend910b;ascend310p" \
    --vendor_name=${vendor_name}

  mapfile -t result_array < <(find ${tmp_compiler_path} -name ${vendor_name})
  if [ ${#result_array[@]} -gt 0 ]; then
    ascendc_result=${result_array[0]}
    cp -r ${ascendc_result} ${ascendc_path}
  fi
fi

vendor_name="custom_ascendc_910"
if [ ! -d "${ascendc_path}/${vendor_name}" ]; then
  rm -rf ${op_host_path}/all_finite*
  rm -rf ${op_kernel_path}/all_finite*
  python setup.py -o ${op_host_path} -k ${op_kernel_path} \
    --soc_version="ascend910" \
    --vendor_name=${vendor_name}

  mapfile -t result_array < <(find ${tmp_compiler_path} -name ${vendor_name})
  if [ ${#result_array[@]} -gt 0 ]; then
    ascendc_result=${result_array[0]}
    cp -r ${ascendc_result} ${ascendc_path}
  fi
fi
