#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
llm_path=$(dirname "$script_path")
script_dir=$(dirname "$llm_path")
networks_dir=$(dirname "$script_dir")
MindSpeed_Core_MS_PATH="$networks_dir"/MindSpeed-Core-MS

src_tools="$MindSpeed_Core_MS_PATH/tools"
dst_tools="$llm_path/tools"
rm -rf "$dst_tools"
cp -r "$src_tools" "$dst_tools"

#MindSpeed-LLM
rm -rf MindSpeed-LLM/
git clone https://gitee.com/ascend/MindSpeed-LLM.git -b master
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed-LLM"
    exit 1
fi
cd MindSpeed-LLM
git checkout 95117b0d91514ed05d4565a2c67ee92063ae8620 #0612
rm -rf tests
cd ..
echo "------------------------------------done MindSpeed-LLM"

#MindSpeed
rm -rf MindSpeed/
git clone https://gitee.com/ascend/MindSpeed.git -b core_r0.8.0
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed"
    exit 1
fi
cd MindSpeed
git checkout 5b710acbb87004ffa8d341dc6b427037d9f76d2d #0612
rm -rf tests_extend
cd ..
echo "...............................................done MindSpeed"

#Megatron-LM
rm -rf Megatron-LM/
git clone https://gitee.com/mirrors/Megatron-LM.git -b core_r0.8.0
if [ $? -ne 0 ]; then
    echo "Error: git clone Megatron-LM"
    exit 1
fi
rm -rf Megatron-LM/tests
echo "..............................................done Megatron-LM"

#msadapter
rm -rf msadapter
git clone https://gitee.com/mindspore/msadapter.git -b master
if [ $? -ne 0 ]; then
    echo "Error: git clone msadapter"
    exit 1
fi
cd msadapter
rm -rf tests
cd ..
echo "..............................................done msadapter"

#transformers
rm -rf transformers/
git clone https://gitee.com/mirrors/huggingface_transformers.git -b v4.47.0
if [ $? -ne 0 ]; then
    echo "Error: git clone msadaptor"
    exit 1
fi
mv huggingface_transformers transformers
cd transformers
git apply ../tools/rules/transformers.diff
rm -rf tests
cd ..
echo "..............................................done apply transformers"

#accelerate
rm -rf accelerate/
git clone https://gitee.com/modelee/accelerate.git -b v1.6.0
if [ $? -ne 0 ]; then
    echo "Error: git clone accelerate"
    exit 1
fi
cd accelerate
git apply ../tools/rules/accelerate.diff
rm -rf tests
cd ..
echo "..............................................done apply accelerate"

echo "..............................................start code_convert"
MindSpeed_Core_MS_PATH=$(pwd)
echo ${MindSpeed_Core_MS_PATH}

python3 tools/transfer.py \
--megatron_path ${MindSpeed_Core_MS_PATH}/Megatron-LM/megatron/ \
--mindspeed_path ${MindSpeed_Core_MS_PATH}/MindSpeed/mindspeed/ \
--mindspeed_llm_path ${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/ \

export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/mindtorch:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:${MindSpeed_Core_MS_PATH}/transformers/src/:${MindSpeed_Core_MS_PATH}/accelerate/src/:$PYTHONPATH
echo $PYTHONPATH
echo "..............................................done code_convert"
