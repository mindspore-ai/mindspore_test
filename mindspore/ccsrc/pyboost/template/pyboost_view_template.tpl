  MS_LOG(DEBUG) << "View ${op_name} Call start";
  auto view_info = ops::${storage_calc}BasicTypeCalc(${call_args});
  auto op = get_op();
  // Create device address for input tensors
  PyBoostUtils::PrepareOpInputs(device_context_, op->stream_id(), ${call_tensors});
  PyBoostUtils::CreateOutputTensor(device_context_, ${input}, view_info, &outputs_);
  ProfileTrackerTask();
  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>(
      [op, ${call_tensors}](){
        MS_LOG(DEBUG) << "View device task ${op_name} start";
        auto device_context = op->device_context();
        PyBoostUtils::MallocOpInputs(device_context, ${call_tensors});
        MS_LOG(DEBUG) << "View device task ${op_name} end";
      }
    )
  );
  ProfileTrackerInput(${call_args});
  ProfileTrackerOutput(${return_values});

  get_op()->CreateOutputSimpleInfo();
  MS_LOG(DEBUG) << "View ${op_name} Call end";
  return ${return_values};
