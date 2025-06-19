#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
rl_path=$(dirname "$script_path")
script_dir=$(dirname "$rl_path")
networks_dir=$(dirname "$script_dir")
MindSpeed_Core_MS_PATH="$networks_dir"/MindSpeed-Core-MS

src_tools="$MindSpeed_Core_MS_PATH/tools"
dst_tools="$rl_path/tools"
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
git checkout 71c5af4d72078d826fd93fec6980004f0de51132
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
git checkout 31aaf3d4ca86234b15f4a5d3af20bd6df06e7d45
rm -rf tests_extend
cd ..
echo "...............................................done MindSpeed"

#MindSpeed-RL
rm -rf MindSpeed-RL/
git clone https://gitee.com/ascend/MindSpeed-RL.git
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed-RL"
    exit 1
fi
cd MindSpeed-RL
echo "...............................................MindSpeed-RL GongKa"
git checkout 0707949f152599862f0a28cb155681599659dc00
rm -rf tests
cd ..
echo "...............................................done MindSpeed-RL"

#Megatron-LM
rm -rf Megatron-LM/
git clone https://gitee.com/mirrors/Megatron-LM.git
if [ $? -ne 0 ]; then
    echo "Error: git clone Megatron-LM"
    exit 1
fi
cd Megatron-LM
git checkout core_r0.8.0
rm -rf tests
cd ..
echo "..............................................done Megatron-LM"

#msadapter
rm -rf msadapter
git clone https://gitee.com/mindspore/msadapter.git
cd msadapter
rm -rf tests
cd ..
if [ $? -ne 0 ]; then
    echo "Error: git clone msadapter"
    exit 1
fi
echo "..............................................done msadapter"

#vllm
rm -rf vllm
git clone https://gitee.com/mirrors/vllm.git
cd vllm
git checkout v0.7.3
rm -rf tests
if [ $? -ne 0 ]; then
    echo "Error: git clone vllm"
    exit 1
fi
cd ..
echo "..............................................done vllm"


#vllm-ascend
rm -rf vllm-ascend
git clone https://gitee.com/mirrors/vllm-ascend.git
cd vllm-ascend
git checkout 0713836e95fe993feefe334945b5b273e4add1f1
rm -rf tests
if [ $? -ne 0 ]; then
    echo "Error: git clone vllm-ascend"
    exit 1
fi
cd ..
echo "..............................................done vllm-ascend"

#transformers
rm -rf transformers/
git clone https://gitee.com/mirrors/huggingface_transformers.git -b v4.47.0
if [ $? -ne 0 ]; then
    echo "Error: git clone huggingface_transformers"
    exit 1
fi
mv huggingface_transformers transformers
cd transformers
git apply ../tools/rules/transformers.diff
rm -rf tests
cd ..
echo "..............................................done apply transformers"


echo "..............................................start code_convert"
MindSpeed_Core_MS_PATH=$PWD
echo ${MindSpeed_Core_MS_PATH}

python3 tools/transfer.py --is_rl_gongka \
--megatron_path ${MindSpeed_Core_MS_PATH}/Megatron-LM/megatron/ \
--mindspeed_path ${MindSpeed_Core_MS_PATH}/MindSpeed/mindspeed/ \
--mindspeed_llm_path ${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/ \
--mindspeed_rl_path ${MindSpeed_Core_MS_PATH}/MindSpeed-RL/ \
--vllm_path ${MindSpeed_Core_MS_PATH}/vllm/ \
--vllm_ascend_path ${MindSpeed_Core_MS_PATH}/vllm-ascend/

export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/mindtorch:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:${MindSpeed_Core_MS_PATH}/transformers/src/:${MindSpeed_Core_MS_PATH}/vllm/:${MindSpeed_Core_MS_PATH}/vllm-ascend/:${MindSpeed_Core_MS_PATH}/MindSpeed-RL/:$PYTHONPATH
echo $PYTHONPATH
echo "..............................................done code_convert"
