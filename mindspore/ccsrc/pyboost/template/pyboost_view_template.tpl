  MS_LOG(DEBUG) << "View ${op_name} Call start";
  auto storage_info_list = ops::${storage_calc}Calc(primitive_, ${calc_func_args});
  if (!storage_info_list.empty()) {
    auto op = get_op();
    // Create device address for input tensors
    PyBoostUtils::PrepareOpInputs(device_context_, op->stream_id(), ${call_tensors});
    PyBoostUtils::CreateOutputTensor(device_context_, ${input}, storage_info_list, &outputs_);
    ProfileTrackerTask();
    // Async
    PyBoostUtils::DispatchRun(
      std::make_shared<runtime::PyBoostDeviceTask>(
        [op, ${call_tensors}](){
          MS_LOG(DEBUG) << "View device task ${op_name} start";
          auto device_context = op->device_context();
          PyBoostUtils::MallocOpInputsForView(device_context, ${call_tensors});
          MS_LOG(DEBUG) << "View device task ${op_name} end";
        }
      )
    );
    ProfileTrackerInput(${call_args});
    ProfileTrackerOutput(${return_values});
  } else {
    MS_LOG_EXCEPTION << "View unsupported:" << primitive_->name() <<" or input ERROR";
  }
  get_op()->CreateOutputSimpleInfo();
  MS_LOG(DEBUG) << "View ${op_name} Call end";
  return ${return_values};
