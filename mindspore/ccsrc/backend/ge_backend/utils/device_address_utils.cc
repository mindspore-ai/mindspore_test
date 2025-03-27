/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/ge_backend/utils/device_address_utils.h"

#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include "ops/op_def.h"
#include "ir/tensor.h"
#include "include/common/utils/ms_device_shape_transfer.h"
#include "frontend/ir/tensor_py.h"
#include "runtime/pipeline/pipeline.h"
#include "runtime/device/res_manager/utils/utils.h"
#include "runtime/device/res_manager/hal_res_manager.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "utils/ms_context.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
namespace {
device::DeviceAddressPtr CreateDeviceAddressForScalarAndString(const ValueNodePtr &value_node,
                                                               const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto node_target = AnfAlgo::FetchDeviceTarget(value_node, graph.get());
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::ResKey res_key{device::GetDeviceTypeByName(node_target), device_id};
  auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);

  device::DeviceAddressPtr address = nullptr;
  const auto &node_value = value_node->value();
  MS_EXCEPTION_IF_NULL(node_value);
  if (node_value->isa<StringImm>()) {
    auto value = GetValue<std::string>(node_value);
    // Allocate one more byte to '/0'
    size_t tensor_size = value.size() + 1;
    if (node_target == kAscendDevice) {
      // size of ge::StringHead which defined in Ascend/latest.aarch64-linux/include/types.h
      constexpr size_t GE_STRING_HEAD_SIZE = 16;
      // NOTE: on Ascend, string type need a head of type ge::StringHead
      tensor_size += GE_STRING_HEAD_SIZE;
    }
    const auto &kernel_tensor =
      AnfAlgo::CreateOutputKernelTensorWithDeviceInfo({value_node, 0}, nullptr, tensor_size, kOpFormat_DEFAULT,
                                                      kObjectTypeString, ShapeVector(), node_target, device_id);
    kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(value_node));
    address = res_manager->CreateDeviceAddress(kernel_tensor);
  } else if (node_value->isa<Scalar>()) {
    auto scalar_value = node_value->cast<ScalarPtr>();
    MS_EXCEPTION_IF_NULL(scalar_value);
    TypePtr data_type = scalar_value->type();
    MS_EXCEPTION_IF_NULL(data_type);
    TypeId type_id = data_type->type_id();
    const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      {value_node, 0}, nullptr, GetTypeByte(TypeIdToType(type_id)), kOpFormat_DEFAULT, type_id, ShapeVector(),
      node_target, device_id);
    kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(value_node));
    address = res_manager->CreateDeviceAddress(kernel_tensor);
  } else if (node_value->isa<None>()) {
    const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      {value_node, 0}, nullptr, 0, kOpFormat_DEFAULT, kTypeNone->type_id(), ShapeVector(), node_target, device_id);
    kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(value_node));
    address = res_manager->CreateDeviceAddress(kernel_tensor);
  }

  return address;
}

mindspore::HashSet<mindspore::AnfNodePtr> FetchValueNodesNeedDevicePtr(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  mindspore::HashSet<mindspore::AnfNodePtr> nodes;
  auto topo_nodes = TopoSort(graph->get_return());
  for (auto const &n : topo_nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }
    auto node = n->cast<CNodePtr>();
    auto op_name = common::AnfAlgo::GetCNodeName(node);
    auto input_num = common::AnfAlgo::GetInputTensorNum(node);
    mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(op_name);
    if (op_def == nullptr) {
      MS_LOG(DEBUG) << op_name << " is not found in OpDef.";
      for (size_t i = 0; i < input_num; i++) {
        auto input = common::AnfAlgo::GetInputNode(node, i);
        (void)nodes.insert(input);
      }
      continue;
    }
    auto args = op_def->args_;
    if (input_num != args.size()) {
      int input_with_init_args = std::count_if(args.begin(), args.end(), [](auto arg) { return arg.as_init_arg_; });
      size_t total = input_num - IntToSize(input_with_init_args);
      for (size_t i = 0; i < total; i++) {
        (void)nodes.insert(common::AnfAlgo::GetInputNode(node, i));
      }
      MS_LOG(DEBUG) << "Node " << op_name << ", has " << input_num << " inputs, but has " << args.size()
                    << " inputs in op_def, it means allsame input, input with init args number: "
                    << input_with_init_args;
      continue;
    }
    for (size_t i = 0; i < input_num; i++) {
      if (args[i].as_init_arg_ == 0) {
        auto input = common::AnfAlgo::GetInputNode(node, i);
        (void)nodes.insert(input);
      }
    }
  }
  return nodes;
}

device::DeviceAddressPtr CreateDeviceAddressForTypeValue(const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(value_node->func_graph());
  auto node_target = AnfAlgo::FetchDeviceTarget(value_node, kernel_graph.get());
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::ResKey res_key{device::GetDeviceTypeByName(node_target), device_id};
  auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);

  const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
    {value_node, 0}, nullptr, 0, kOpFormat_DEFAULT, kMetaTypeTypeType, {}, node_target, device_id);
  kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(value_node));
  device::DeviceAddressPtr address = res_manager->CreateDeviceAddress(kernel_tensor);
  MS_LOG(DEBUG) << "Create addr for node:" << value_node->DebugString() << " addr:" << address;
  MS_EXCEPTION_IF_NULL(address);
  address->set_from_persistent_mem(true);
  AnfAlgo::SetOutputAddr(address, 0, value_node.get());
  return address;
}
}  // namespace

bool DeviceAddressUtils::NodeDeviceAddressExist(const device::DeviceType &node_device_type, const AnfNodePtr &node,
                                                size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  if (AnfAlgo::OutputAddrExist(node, index)) {
    const auto address = AnfAlgo::GetMutableOutputAddr(node, index, false);
    MS_EXCEPTION_IF_NULL(address);
    CreateKernelTensor(address, session::AnfRuntimeAlgorithm::GetNodeAbstractByIndex(node, index));
    return address->GetDeviceType() == node_device_type;
  }
  return false;
}

void DeviceAddressUtils::CreateDeviceAddressByMapTensorNode(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &abstract_base = AnfAlgo::GetNodeAbstractByIndex(node, index);
  MS_EXCEPTION_IF_NULL(abstract_base);
  if (!abstract_base->isa<abstract::AbstractMapTensor>()) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Parameter:" << node->DebugString() << " is not a map tensor type.";
  }

  const auto &abstract = abstract_base->cast<abstract::AbstractMapTensorPtr>();
  MS_EXCEPTION_IF_NULL(abstract);

  // Parse attrs for user data by abstract.
  const auto &value_shape = abstract->value_shape();
  MS_EXCEPTION_IF_NULL(value_shape);
  const auto &shape_vector = value_shape->shape();
  const auto &map_tensor_type = abstract->map_tensor_type();
  MS_EXCEPTION_IF_NULL(map_tensor_type);
  MS_EXCEPTION_IF_NULL(map_tensor_type->key_dtype());
  MS_EXCEPTION_IF_NULL(map_tensor_type->value_dtype());

  auto user_data = std::make_shared<UserData>();
  user_data->set(kUserDataType, std::make_shared<UserDataType>(UserDataType::kUserTypeHashTable));
  user_data->set(kHashTableKeyType, std::make_shared<TypeId>(map_tensor_type->key_dtype()->type_id()));
  user_data->set(kHashTableValueType, std::make_shared<TypeId>(map_tensor_type->value_dtype()->type_id()));
  user_data->set(kHashTableShapeVector, std::make_shared<ShapeVector>(shape_vector));
  user_data->set(kHashTableDefaultValue, abstract->default_value());
  user_data->set(kHashTablePermitFilter, abstract->permit_filter_value());
  user_data->set(kHashTableEvictFilter, abstract->evict_filter_value());

  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(node->func_graph());
  auto node_target = AnfAlgo::FetchDeviceTarget(node, kernel_graph.get());
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::ResKey res_key{device::GetDeviceTypeByName(node_target), device_id};
  auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);

  // Create device for map tensor node and the ptr size is 1 byte.
  const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
    {node, index}, nullptr, 1, kOpFormat_DEFAULT, TypeId::kObjectTypeMapTensorType, ShapeVector(), node_target,
    device_id, user_data);
  kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(node));
  auto device_address = res_manager->CreateDeviceAddress(kernel_tensor);
  MS_LOG(DEBUG) << "Create device tensor:" << device_address << " type:" << device_address->type_id();
  AnfAlgo::SetOutputAddr(device_address, index, node.get());
}

void DeviceAddressUtils::CreateParameterDeviceAddress(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> graph_inputs = graph->inputs();
  const std::vector<bool> &graph_valid_input = graph->valid_inputs();
  (void)graph_inputs.insert(graph_inputs.end(), graph->child_graph_result().begin(), graph->child_graph_result().end());

  // Anf nodes which need create device address.
  std::vector<AnfNodePtr> nodes_list;
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    AnfNodePtr item = graph_inputs[i];
    MS_EXCEPTION_IF_NULL(item);
    if (i < graph_valid_input.size() && !graph_valid_input[i]) {
      continue;
    }

    auto node_target = device::GetDeviceTypeByName(AnfAlgo::FetchDeviceTarget(item, graph.get()));

    if (common::AnfAlgo::CheckPrimitiveType(item, prim::kPrimMakeTuple)) {
      std::vector<AnfNodePtr> outs = common::AnfAlgo::GetAllOutput(item);
      for (const auto &out : outs) {
        MS_EXCEPTION_IF_NULL(out);
        if (!out->isa<Parameter>() || NodeDeviceAddressExist(node_target, out, 0)) {
          continue;
        }
        nodes_list.push_back(out);
      }
    }
    if (!item->isa<Parameter>() || NodeDeviceAddressExist(node_target, item, 0)) {
      continue;
    }
    nodes_list.push_back(item);
  }

  // Create device address for anf node in nodes_list
  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  for (const auto &item : nodes_list) {
    MS_EXCEPTION_IF_NULL(item);
    auto node_target = AnfAlgo::FetchDeviceTarget(item, graph.get());
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::ResKey res_key{device::GetDeviceTypeByName(node_target), device_id};
    auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
    MS_EXCEPTION_IF_NULL(res_manager);

    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      const auto &abstract = AnfAlgo::GetNodeAbstractByIndex(item, index);
      if (abstract != nullptr && abstract->isa<abstract::AbstractMapTensor>()) {
        CreateDeviceAddressByMapTensorNode(item, index);
        continue;
      }

      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      if (output_type_id == kTypeUnknown) {
        output_type_id = common::AnfAlgo::GetOutputInferDataType(item, index);
      }

      size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(item, index);
      const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
        {item, index}, nullptr, tensor_size, AnfAlgo::GetOutputFormat(item, index), output_type_id,
        AnfAlgo::GetRuntimePaddingShape(item, index), node_target, device_id);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(item));
      auto device_address = res_manager->CreateDeviceAddress(kernel_tensor);

      MS_EXCEPTION_IF_NULL(device_address);
      MS_LOG(DEBUG) << "Create device address:" << device_address << " for item:" << item->DebugString();
      // Set the flag of no user parameter.
      if (item->isa<Parameter>()) {
        auto input_param = item->cast<ParameterPtr>();
        MS_EXCEPTION_IF_NULL(input_param);
        // Unused address will not alloc memory, which is easy to cause problems for weight node, so skip weight node.
        if (!common::AnfAlgo::IsParameterWeight(input_param) &&
            !input_param->IsUsedByRealKernelInGraph(graph->graph_id())) {
          MS_LOG(INFO) << "Node:" << item->fullname_with_scope() << " debug name:" << item->DebugString()
                       << " is not used in the graph " << graph->graph_id();
          device_address->UpdateFlag(device::kDeviceAddressFlagNotUsed);
        }
      }
      device_address->SetNodeIndex(item, index);
      device_address->set_from_persistent_mem(item->isa<Parameter>());
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(item)
                    << " addr:" << device_address << " type:" << device_address->type_id();
      AnfAlgo::SetOutputAddr(device_address, index, item.get());
    }
  }
}

void DeviceAddressUtils::UpdateDeviceAddressHostInfoByNode(const device::DeviceAddressPtr &addr, const AnfNodePtr &node,
                                                           size_t output_idx) {
  MS_EXCEPTION_IF_NULL(addr);
  CreateKernelTensor(addr, session::AnfRuntimeAlgorithm::GetNodeAbstractByIndex(node, output_idx));
}

device::DeviceAddressPtrList DeviceAddressUtils::CreateDeviceAddressForTensorValue(const ValuePtr &node_value,
                                                                                   size_t output_idx,
                                                                                   const ValueNodePtr &value_node,
                                                                                   const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(node_value);
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  auto node_target = AnfAlgo::FetchDeviceTarget(value_node, graph.get());

  device::DeviceAddressPtrList address_list;
  if (node_value->isa<tensor::BaseTensor>()) {
    auto tensor = node_value->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto output_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (output_address != nullptr) {
      if (output_address->GetDeviceType() == device::GetDeviceTypeByName(node_target)) {
        // We need to set tensor->device_address to ValueNode even if the tensor is a forward_output tensor
        // in PyNative Bprop graph. ValueNode device_address is necessary for GraphSchedule::Transform.
        UpdateDeviceAddressHostInfoByNode(output_address, value_node, output_idx);
        AnfAlgo::SetOutputAddr(std::static_pointer_cast<device::DeviceAddress>(tensor->device_address()), output_idx++,
                               value_node.get());
        (void)address_list.emplace_back(output_address);
        return address_list;
      }
      tensor->data_sync();
    }
  }

  size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(value_node, output_idx);
  TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(value_node, output_idx);
  if (output_type_id == kTypeUnknown) {
    output_type_id = common::AnfAlgo::GetOutputInferDataType(value_node, output_idx);
    if (output_type_id == kTypeUnknown && value_node->value() != nullptr && value_node->value()->isa<ValueTuple>() &&
        value_node->value()->cast<ValueTuplePtr>()->size() == 0) {
      MS_LOG(DEBUG) << "Set int64 type for empty value tuple node:" << value_node->DebugString();
      output_type_id = TypeId::kNumberTypeInt64;
    }
  }
  std::string output_format = AnfAlgo::GetOutputFormat(value_node, output_idx);

  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::ResKey res_key{device::GetDeviceTypeByName(node_target), device_id};
  auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);

  const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
    {value_node, output_idx}, nullptr, tensor_size, output_format, output_type_id, {}, node_target, device_id);
  kernel_tensor->set_host_shape(kernel_tensor->GetShapeVector());
  kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(value_node));
  device::DeviceAddressPtr address = res_manager->CreateDeviceAddress(kernel_tensor);
  MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(value_node) << " addr:" << address
                << " size:" << tensor_size << " format:" << output_format << " type:" << output_type_id
                << " shape:" << kernel_tensor->GetShapeVector();
  MS_EXCEPTION_IF_NULL(address);
  address->set_from_persistent_mem(true);
  AnfAlgo::SetOutputAddr(address, output_idx++, value_node.get());
  (void)address_list.emplace_back(address);
  return address_list;
}

void DeviceAddressUtils::CreateValueNodeDeviceAddress(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  // store node without init args, means need device addr
  auto value_nodes_without_init_args = FetchValueNodesNeedDevicePtr(graph);
  for (const ValueNodePtr &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    auto node_target = device::GetDeviceTypeByName(AnfAlgo::FetchDeviceTarget(value_node, graph.get()));
    if (NodeDeviceAddressExist(node_target, value_node, 0)) {
      continue;
    }

    const auto &abstract = value_node->abstract();
    if (abstract != nullptr && abstract->isa<abstract::AbstractMapTensor>()) {
      CreateDeviceAddressByMapTensorNode(value_node, 0);
      continue;
    }
    const auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    if (node_value->isa<tensor::BaseTensor>() || node_value->isa<ValueSequence>()) {
      auto address_list = CreateDeviceAddressForTensorValue(node_value, 0, value_node, graph);
      // Deal with tensor and tuple
      if (value_nodes_without_init_args.find(value_node) == value_nodes_without_init_args.end()) {
        for (const auto &address : address_list) {
          MS_EXCEPTION_IF_NULL(address);
          address->UpdateFlag(device::kDeviceAddressFlagIgnoreDevicePtr);
          MS_LOG(DEBUG) << "Find node " << value_node->DebugString() << " has init args";
        }
      }
      continue;
    } else if (node_value->isa<Type>()) {
      CreateDeviceAddressForTypeValue(value_node);
      continue;
    }

    device::DeviceAddressPtr address = CreateDeviceAddressForScalarAndString(value_node, graph);
    // Deal with string and scalar; Address will be nullptr if the input is a type.
    if (address && (value_nodes_without_init_args.find(value_node) == value_nodes_without_init_args.end())) {
      address->UpdateFlag(device::kDeviceAddressFlagIgnoreDevicePtr);
      MS_LOG(DEBUG) << "Find node " << value_node->DebugString() << " has init args";
    }
    if (address != nullptr) {
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(value_node)
                    << " addr:" << address;
      address->set_from_persistent_mem(true);
      AnfAlgo::SetOutputAddr(address, 0, value_node.get());
    } else {
      MS_LOG(INFO) << "No device address for value node:" << value_node->fullname_with_scope()
                   << ", debug name:" << common::AnfAlgo::GetNodeDebugString(value_node);
    }
  }
}

device::DeviceAddressPtr DeviceAddressUtils::CloneEmptyDeviceAddress(
  const device::DeviceAddressPtr &old_device_address) {
  MS_EXCEPTION_IF_NULL(old_device_address);
  const auto &kernel_tensor = old_device_address->kernel_tensor();
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto new_kernel_tensor = kernel_tensor->CloneKernelTensor();
  MS_EXCEPTION_IF_NULL(new_kernel_tensor);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  new_kernel_tensor->set_device_name(device_name);
  new_kernel_tensor->set_device_id(device_id);
  new_kernel_tensor->set_device_ptr(nullptr);

  device::ResKey res_key{device::GetDeviceTypeByName(device_name), device_id};
  auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);
  auto new_device_address = res_manager->CreateDeviceAddress(new_kernel_tensor);
  MS_EXCEPTION_IF_NULL(new_device_address);
  MS_LOG(DEBUG) << "Create device tensor:" << new_device_address << " type:" << new_device_address->type_id();

  new_device_address->set_original_ref_count(old_device_address->original_ref_count());
  new_device_address->ResetRefCount();
  auto node = old_device_address->GetNodeIndex();
  new_device_address->SetNodeIndex(node.first, node.second);
  new_device_address->set_padding_type(old_device_address->padding_type());
  return new_device_address;
}

void DeviceAddressUtils::CreateKernelTensor(const device::DeviceAddressPtr &device_address,
                                            const tensor::BaseTensor *tensor) {
  MS_EXCEPTION_IF_NULL(device_address);
  MS_EXCEPTION_IF_NULL(tensor);
  if (device_address->kernel_tensor() != nullptr) {
    return;
  }
  {
    GilReleaseWithCheck no_gil;
    // DeviceAddress is used by backebd queue.
    runtime::Pipeline::Get().backend_stage()->Wait();
  }
  const auto &address_common = device_address->address_common();
  MS_EXCEPTION_IF_NULL(address_common);
  auto real_kernel_tensor = std::make_shared<kernel::KernelTensor>(
    address_common, std::make_shared<abstract::TensorShape>(tensor->shape()),
    std::make_shared<TensorType>(TypeIdToType(tensor->data_type())), nullptr, tensor->shape());
  device_address->set_kernel_tensor(real_kernel_tensor);
  device_address->DeviceSynchronizerInit();
}

void DeviceAddressUtils::CreateKernelTensor(const ValuePtr &input_value) {
  MS_EXCEPTION_IF_NULL(input_value);
  if (input_value->isa<tensor::BaseTensor>()) {
    auto tensor = input_value->cast<tensor::BaseTensorPtr>();
    if (tensor->device_address() != nullptr) {
      auto device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
      MS_EXCEPTION_IF_NULL(device_address);
      CreateKernelTensor(device_address, tensor.get());
    }
  }
}

void DeviceAddressUtils::CreateKernelTensor(const tensor::TensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  if (input_tensor->device_address() != nullptr) {
    auto device_address = std::static_pointer_cast<device::DeviceAddress>(input_tensor->device_address());
    MS_EXCEPTION_IF_NULL(device_address);
    CreateKernelTensor(device_address, input_tensor.get());
  }
}

void DeviceAddressUtils::CreateKernelTensor(const device::DeviceAddressPtr &device_address,
                                            const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(device_address);
  if (device_address->kernel_tensor() != nullptr) {
    return;
  }
  {
    GilReleaseWithCheck no_gil;
    // DeviceAddress is used by backebd queue.
    runtime::Pipeline::Get().backend_stage()->Wait();
  }
  const auto address_common = device_address->address_common();
  MS_EXCEPTION_IF_NULL(address_common);
  MS_EXCEPTION_IF_NULL(abs);
  const auto &shape = abs->GetShape();
  const auto &type = abs->GetType();
  auto real_kernel_tensor =
    std::make_shared<kernel::KernelTensor>(address_common, shape, type, nullptr, shape->GetShapeVector());
  device_address->set_kernel_tensor(real_kernel_tensor);
  device_address->DeviceSynchronizerInit();
}

bool DeviceAddressUtils::IsContiguousTensor(const tensor::BaseTensorPtr &tensor) {
  if (tensor == nullptr || tensor->storage_info() == nullptr) {
    MS_LOG(INFO) << "It is a contiguous tensor, tensor: " << tensor;
    return true;
  }

  auto device_address = tensor->device_address();
  const auto &old_storage_info = device_address->GetTensorStorageInfo();
  if (old_storage_info == nullptr) {
    MS_LOG(INFO) << "It is a contiguous tensor, tensor: " << tensor;
    return true;
  }

  auto new_storage_info = tensor->storage_info();
  if (new_storage_info->shape == new_storage_info->ori_shape) {
    MS_LOG(INFO) << "It is a contiguous tensor, tensor: " << tensor;
    return true;
  }

  if (new_storage_info->storage_offset == 0 && new_storage_info->is_contiguous) {
    MS_LOG(INFO) << "It is a contiguous tensor, tensor: " << tensor;
    return true;
  }

  MS_LOG(ERROR) << "It is not a contiguous tensor, tensor: " << tensor
                << ", new_storage_info: " << new_storage_info->ToString()
                << ", old_storage_info: " << old_storage_info->ToString();
  return false;
}
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
