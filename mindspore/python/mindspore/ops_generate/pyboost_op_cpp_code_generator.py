# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
This module defines several classes and functions for generating C++ code for PyBoost operations,
including function headers, source files, and registration code. It handles the generation of code
for different devices (Ascend, CPU, GPU) and manages residual files associated with operator prototypes.
"""

import os
import re

from pyboost_utils import is_cube, AclnnUtils, get_return_type, merge_strings_by_chunk_size, is_op_multi_output
import template
import gen_constants as K
from gen_utils import save_file
from op_proto import OpProto
from op_template_parser import OpTemplateParser
from base_generator import BaseGenerator


class PyboostCommonOpHeaderGenerator(BaseGenerator):
    """
    Generates common C++ headers for PyBoost operations.

    This class processes operator prototypes and generates header files containing function definitions
    based on templates provided. It specifically generates the headers that define operations for PyBoost.
    """

    def __init__(self):
        self.pyboost_op_header_str = template.PYBOOST_BASE_OP_DEFINE_TEMPLATE

    def generate(self, work_path, op_protos):
        """
        Generates header files for the provided operator prototypes.

        Args:
            work_path (str): The directory path where the header files will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators.

        Returns:
            None
        """
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            op_parser = OpTemplateParser(op_proto)
            op_name_str = op_proto.op_class.name
            call_args_with_type = op_parser.parse_call_args_with_types()
            cpp_func_return = _generate_cpp_func_return(op_proto)
            output_is_tuple = "bool output_is_tuple() const override { return true; }" \
                if is_op_multi_output(op_proto.op_returns) else ''
            pyboost_op_header_str = template.PYBOOST_BASE_OP_DEFINE_TEMPLATE.replace(op_name=op_name_str,
                                                                                     op_name_upper=op_name_str.upper(),
                                                                                     call_args=call_args_with_type,
                                                                                     return_type=cpp_func_return,
                                                                                     output_is_tuple=output_is_tuple)
            save_path = os.path.join(work_path, f"{K.MS_COMMON_PYBOOST_KERNEL_PATH}/auto_generate/")
            file_name = f"{op_proto.op_name}.h"
            save_file(save_path, file_name, pyboost_op_header_str)


class PyboostOpHeaderGenerator(BaseGenerator):
    """
    Generates device-specific C++ headers for PyBoost operations.

    This class generates header files for different devices (Ascend, CPU, GPU) and defines
    the operation functions accordingly.
    """

    def __init__(self, device):
        """
        Initializes the PyboostOpHeaderGenerator with the appropriate templates for the specified device.

        Args:
            device (str): The target device (ascend, gpu, or cpu).

        Raises:
            ValueError: If the device is not supported.
        """
        template_dict = {"ascend": template.PYBOOST_ASCEND_OP_HEADER_TEMPLATE,
                         "gpu": template.PYBOOST_GPU_OP_HEADER_TEMPLATE,
                         "cpu": template.PYBOOST_CPU_OP_HEADER_TEMPLATE}
        if device not in template_dict:
            raise ValueError(f"Device must be ascend, gpu, or cpu, {device} is not supported")
        self.PYBOOST_OP_HEADER_TEMPLATE = template_dict[device]
        self.code_generate_path = f"{K.MS_OPS_KERNEL_PATH}/{device}/pyboost/auto_generate/"
        self.device = device

    def generate(self, work_path, op_protos):
        """
        Generates header files for the provided operator prototypes based on the device.

        Args:
            work_path (str): The directory path where the header files will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators.

        Returns:
            None
        """
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            if getattr(op_proto.op_dispatch, self.device) == 'None':
                continue
            op_parser = OpTemplateParser(op_proto)
            op_name_str = op_proto.op_class.name
            call_args_with_type = op_parser.parse_call_args_with_types()
            cpp_func_return = _generate_cpp_func_return(op_proto)

            pyboost_op_str = self.PYBOOST_OP_HEADER_TEMPLATE.replace(op_name=op_name_str,
                                                                     op_name_upper=op_name_str.upper(),
                                                                     operator_name=op_proto.op_name,
                                                                     call_args_with_type=call_args_with_type,
                                                                     return_type=cpp_func_return)

            save_path = os.path.join(work_path, self.code_generate_path)
            file_name = f"{op_proto.op_name}.h"
            save_file(save_path, file_name, pyboost_op_str)


class PyboostOpCppGenerator:
    """
    Generates C++ source files for PyBoost operations.

    This class generates the implementation of operations for different devices, handling function calls
    and registering custom kernels as necessary.
    """

    def __init__(self, device):
        """
        Initializes the PyboostOpCppGenerator with the appropriate templates for the specified device.

        Args:
            device (str): The target device (ascend, gpu, or cpu).

        Raises:
            ValueError: If the device is not supported.
        """
        if device == 'ascend':
            PYBOOST_CUSTOMIZE_CALL_TEMPLATE = template.PYBOOST_ASCEND_CUSTOMIZE_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_HEADER_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/ascend/pyboost/auto_generate/"
        elif device == 'cpu':
            PYBOOST_CUSTOMIZE_CALL_TEMPLATE = template.PYBOOST_CPU_CUSTOMIZE_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.PYBOOST_CPU_SINGLE_OP_HEADER_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_CPU_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/cpu/pyboost/auto_generate/"
        elif device == 'gpu':
            PYBOOST_CUSTOMIZE_CALL_TEMPLATE = template.PYBOOST_GPU_CUSTOMIZE_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.PYBOOST_GPU_SINGLE_OP_HEADER_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_GPU_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/gpu/pyboost/auto_generate/"
        else:
            raise ValueError(f"Device must be ascend, gpu, or cpu, {device} is not supported")
        self.PYBOOST_CUSTOMIZE_CALL_TEMPLATE = PYBOOST_CUSTOMIZE_CALL_TEMPLATE
        self.PYBOOST_SINGLE_OP_HEADER_TEMPLATE = PYBOOST_SINGLE_OP_HEADER_TEMPLATE
        self.PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = PYBOOST_SINGLE_OP_SOURCE_TEMPLATE
        self.gen_path = gen_path
        self.device = device

    def generate_customize_op_cpp_code(self, op_protos, merge_op_header, merge_op_function):
        """
        Generate C++ code for PyBoost operations using the provided operation prototypes.

        This method processes a list of operation prototypes, generates customized function call
        implementations, and updates the merged headers and functions for the specified device.

        Args:
            op_protos (list): A list of operation prototypes to process. Each prototype contains
                              metadata about the operation, including dispatch settings and arguments.
            merge_op_header (list): A list to store the generated C++ header code for operations.
            merge_op_function (list): A list to store the generated C++ source code for operations.
        """
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            if getattr(op_proto.op_dispatch, self.device) == 'default':
                continue
            if getattr(op_proto.op_dispatch, self.device) == 'None':
                continue
            op_parser = OpTemplateParser(op_proto)
            call_args = op_parser.parse_original_call_args(op_proto.op_args)
            call_args_with_type = op_parser.parse_call_args_with_types()
            _, call_func_outputs = op_parser.generate_pyboost_outputs()
            operator_name = op_proto.op_name
            op_name_str = op_proto.op_class.name
            check_inplace_func = ''
            for arg in op_proto.op_returns:
                if arg.inplace != '':
                    check_inplace_func = f'ThrowExpectionWhenInternalOverlap({arg.inplace}_tensor);'
                    break
            call_impl = self.PYBOOST_CUSTOMIZE_CALL_TEMPLATE.replace(
                call_args=call_args,
                return_values=call_func_outputs,
                customize_func=getattr(op_proto.op_dispatch, self.device) + "Customize",
                check_expression=check_inplace_func,
            )
            customize_include = \
                f'#include "{K.MS_OPS_KERNEL_PATH}/{self.device}/pyboost/customize/{operator_name.lower()}.h"\n'
            register_custom = self._get_register_custom_kernel(op_proto)
            cpp_func_return = _generate_cpp_func_return(op_proto)
            merge_op_header.append(self.PYBOOST_SINGLE_OP_HEADER_TEMPLATE.replace(operator_name=operator_name,
                                                                                  customize_include=customize_include))
            merge_op_function.append(
                self.PYBOOST_SINGLE_OP_SOURCE_TEMPLATE.replace(op_name=op_name_str,
                                                               call_args_with_type=call_args_with_type,
                                                               return_type=cpp_func_return, call_impl=call_impl,
                                                               register_custom_kernel=register_custom))

    def _get_register_custom_kernel(self, op_proto: OpProto):
        """
        Generates the registration code for custom kernels based on the device.

        Args:
            op_proto (OpProto): The operator prototype to generate registration for.

        Returns:
            str: The registration code for the custom kernel.
        """
        if self.device == 'ascend':
            register_custom_kernel = ''
        elif self.device == 'cpu':
            register_custom_kernel = f"MS_REG_PYBOOST_CPU_CUSTOM_KERNEL({op_proto.op_class.name});"
        elif self.device == 'gpu':
            register_custom_kernel = f"MS_REG_PYBOOST_GPU_CUSTOM_KERNEL({op_proto.op_class.name});"
        else:
            raise ValueError(f"Device must be ascend, gpu, or cpu, {self.device} is not supported")
        return register_custom_kernel


class PyboostViewOpCppGenerator:
    """
    Generates C++ source files for view operations in PyBoost.

    This class handles the generation of source files for view operations, which have special handling
    compared to regular operations.
    """

    def __init__(self, device):
        """
        Initializes the PyboostViewOpCppGenerator with the appropriate templates for the specified device.

        Args:
            device (str): The target device (ascend, gpu, or cpu).

        Raises:
            ValueError: If the device is not supported.
        """
        if device == 'ascend':
            PYBOOST_VIEW_CALL_TEMPLATE = template.PYBOOST_ASCEND_VIEW_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_HEADER_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/ascend/pyboost/auto_generate/"
        elif device == 'cpu':
            PYBOOST_VIEW_CALL_TEMPLATE = template.PYBOOST_CPU_VIEW_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.PYBOOST_CPU_SINGLE_OP_HEADER_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_CPU_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/cpu/pyboost/auto_generate/"
        elif device == 'gpu':
            PYBOOST_VIEW_CALL_TEMPLATE = template.PYBOOST_GPU_VIEW_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.PYBOOST_GPU_SINGLE_OP_HEADER_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_GPU_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/gpu/pyboost/auto_generate/"
        else:
            raise ValueError(f"Device must be ascend, gpu, or cpu, {device} is not supported")
        self.PYBOOST_VIEW_CALL_TEMPLATE = PYBOOST_VIEW_CALL_TEMPLATE
        self.PYBOOST_SINGLE_OP_HEADER_TEMPLATE = PYBOOST_SINGLE_OP_HEADER_TEMPLATE
        self.PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = PYBOOST_SINGLE_OP_SOURCE_TEMPLATE
        self.gen_path = gen_path
        self.device = device

    def generate_view_op_cpp_code(self, op_protos, merge_op_header, merge_op_function):
        """
        Generate C++ code for view operations in PyBoost.

        This method processes a list of operation prototypes (`op_protos`) and generates C++ code
        for view operations where `op_view` is set to `True` and the dispatch setting for the target
        device is `'default'`.

        Args:
            op_protos (list): A list of operation prototypes to process. Each prototype includes
                              metadata such as dispatch settings, arguments, and view-specific attributes.
            merge_op_header (list): A list to store the generated C++ header code for view operations.
            merge_op_function (list): A list to store the generated C++ source code for view operations.
        """
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            if getattr(op_proto.op_dispatch, self.device) != 'default':
                continue
            if getattr(op_proto.op_dispatch, self.device) == 'None':
                continue
            if not op_proto.op_view:
                continue

            op_parser = OpTemplateParser(op_proto)
            call_args_tensor = op_parser.get_call_args_tensor()
            call_args = op_parser.parse_original_call_args(op_proto.op_args)
            call_args_with_type = op_parser.parse_call_args_with_types()
            _, call_func_outputs = op_parser.generate_pyboost_outputs()
            call_impl = self.PYBOOST_VIEW_CALL_TEMPLATE.replace(op_name=op_proto.op_class.name,
                                                                call_args=call_args,
                                                                call_tensors=call_args_tensor,
                                                                return_values=call_func_outputs,
                                                                input=call_args[0])
            customize_include = f'#include "{K.MS_OPS_VIEW_PATH}/{op_proto.op_name}_strides_calc.h"\n'
            cpp_func_return = _generate_cpp_func_return(op_proto)
            merge_op_header.append(self.PYBOOST_SINGLE_OP_HEADER_TEMPLATE.replace(operator_name=op_proto.op_name,
                                                                                  customize_include=customize_include))

            merge_op_function.append(
                self.PYBOOST_SINGLE_OP_SOURCE_TEMPLATE.replace(op_name=op_proto.op_class.name,
                                                               call_args_with_type=call_args_with_type,
                                                               return_type=cpp_func_return,
                                                               call_impl=call_impl,
                                                               register_custom_kernel=""))


class AclnnOpCppCodeGenerator:
    """
    Generates C++ source files for ACLNN operations in PyBoost.

    This class handles the generation of source files for operations that utilize the ACLNN framework,
    including customized calls and tensor management.

    Attributes:
        PYBOOST_CALL_TEMPLATE (Template): Template for generating ACLNN operation calls.
        PYBOOST_OP_SOURCE_TEMPLATE (Template): Template for generating operation source files.
        gen_path (str): Path for saving the generated C++ source files.
        device (str): The target device (ascend, cpu, or gpu).
    """

    def __init__(self, device):
        """
        Initializes the AclnnOpCppCodeGenerator with the appropriate templates for the specified device.

        Args:
            device (str): The target device (ascend, gpu, or cpu).

        Raises:
            ValueError: If the device is not supported.
        """
        if device == 'ascend':
            PYBOOST_CALL_TEMPLATE = template.PYBOOST_ASCEND_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/ascend/pyboost/auto_generate/"
        elif device == 'cpu':
            PYBOOST_CALL_TEMPLATE = template.PYBOOST_CPU_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_CPU_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/cpu/pyboost/auto_generate/"
        elif device == 'gpu':
            PYBOOST_CALL_TEMPLATE = template.PYBOOST_GPU_CALL_TEMPLATE
            PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = template.PYBOOST_GPU_SINGLE_OP_SOURCE_TEMPLATE
            gen_path = f"{K.MS_OPS_KERNEL_PATH}/gpu/pyboost/auto_generate/"
        else:
            raise ValueError(f"Device must be ascend, gpu, or cpu, {device} is not supported")
        self.PYBOOST_CALL_TEMPLATE = PYBOOST_CALL_TEMPLATE
        self.PYBOOST_SINGLE_OP_HEADER_TEMPLATE = template.Template(
            '#include "kernel/${device}/pyboost/auto_generate/${operator_name}.h"\n'
        )
        self.PYBOOST_SINGLE_OP_SOURCE_TEMPLATE = PYBOOST_SINGLE_OP_SOURCE_TEMPLATE
        self.gen_path = gen_path
        self.device = device

    def generate_aclnn_op_cpp_code(self, op_protos, merge_op_header, merge_op_function):
        """
        Generate C++ code for ACLNN operations in PyBoost.

        This method processes a list of operation prototypes (`op_protos`) and generates C++ code
        for aclnn operations. The method filters the operation
        prototypes based on their dispatch and view settings, and then uses templates and metadata
        to generate the necessary implementation and header files.

        Args:
            op_protos (list): A list of operation prototypes. Each prototype includes metadata
                              such as operation name, dispatch settings, view attributes, and arguments.
            merge_op_header (list): A list to store the generated C++ header code for ACLNN operations.
            merge_op_function (list): A list to store the generated C++ source code for ACLNN operations.
        """
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            if getattr(op_proto.op_dispatch, self.device) != 'default':
                continue
            if getattr(op_proto.op_dispatch, self.device) == 'None':
                continue
            if op_proto.op_view:
                continue

            op_parser = OpTemplateParser(op_proto)
            aclnn_name = AclnnUtils.get_aclnn_interface(op_proto.op_class.name)

            call_args_tensor = op_parser.get_call_args_tensor()
            create_input_address = self._generate_create_input_address(op_parser)
            malloc_inputs = self._generate_malloc_input(op_parser)
            op_outputs, call_func_outputs = op_parser.generate_pyboost_outputs()
            get_inputs_kernel_tensors = self._generate_get_inputs_kernel_tensors(op_parser)

            cube_math_type, get_cube_math_type = '', ''
            if self.device == 'ascend' and is_cube(op_proto.op_class.name):
                get_cube_math_type = f'// cubeMathType: 0 - KEEP_DTYPE, 1 - ALLOW_FP32_DOWN_PRECISION\n'
                get_cube_math_type += "auto cube_math_type = GetCubeMathType();"
                cube_math_type = ', cube_math_type'

            real_output = ', ' + op_outputs \
                if _generate_inplace_process_cpp_code(op_proto) == '' else ''

            cast_input_code, real_call_args_tensor = self._generate_tensor_cpu_cast_input_code(
                op_parser)
            cpp_func_return = _generate_cpp_func_return(op_proto)
            _, tensor_list_convert, call_args_with_tensor = op_parser.parse_need_malloc_tensors()
            call_args_after_convert, value_tuple_convert, const_number_convert = op_parser.op_args_converter()
            call_args = op_parser.parse_original_call_args(op_proto.op_args)
            call_args_with_type = op_parser.parse_call_args_with_types()
            inplace_process = _generate_inplace_process_cpp_code(op_proto)
            call_impl = self.PYBOOST_CALL_TEMPLATE.replace(aclnn_name=aclnn_name,
                                                           call_args=call_args,
                                                           call_tensors=call_args_tensor,
                                                           value_tuple_convert=value_tuple_convert,
                                                           const_number_convert=const_number_convert,
                                                           create_input_address=create_input_address,
                                                           tensor_list_convert=tensor_list_convert,
                                                           call_args_with_tensor=call_args_with_tensor,
                                                           malloc_inputs=malloc_inputs,
                                                           get_inputs_kernel_tensors=get_inputs_kernel_tensors,
                                                           get_cube_math_type=get_cube_math_type,
                                                           cube_math_type=cube_math_type,
                                                           real_call_args=call_args_after_convert,
                                                           return_values=call_func_outputs,
                                                           outputs=real_output,
                                                           inplace_process=inplace_process,
                                                           cast_input_code=cast_input_code,
                                                           real_call_args_tensor=real_call_args_tensor,
                                                           class_name=op_proto.op_class.name,
                                                           op_name_str=op_proto.op_class.name)

            merge_op_header.append(self.PYBOOST_SINGLE_OP_HEADER_TEMPLATE.replace(operator_name=op_proto.op_name,
                                                                                  device=self.device))

            merge_op_function.append(
                self.PYBOOST_SINGLE_OP_SOURCE_TEMPLATE.replace(op_name=op_proto.op_class.name,
                                                               call_args_with_type=call_args_with_type,
                                                               return_type=cpp_func_return,
                                                               call_impl=call_impl,
                                                               register_custom_kernel=''))

    def _generate_tensor_cpu_cast_input_code(self, op_parser: OpTemplateParser):
        """
        Generates the input casting code for CPU tensor operations.

        Args:
            op_parser (OpTemplateParser): The parser object for the operation prototype.

        Returns:
            tuple: A tuple containing the casting code and the updated tensor call arguments.
        """
        _, _, call_args_with_tensor = op_parser.parse_need_malloc_tensors()
        call_tensors = op_parser.get_call_args_tensor()
        cast_input = ""
        real_call_args_tensor = call_args_with_tensor.copy()
        for i, tensor in enumerate(call_args_with_tensor):
            is_tuple_tensor = real_call_args_tensor[i].endswith("_vector")
            is_tensor = real_call_args_tensor[i] in call_tensors
            if is_tensor:
                cast_input += f'const auto &real_{tensor} = PyBoostUtils::CastTensor({tensor}, ' \
                              f'select_kernel.input_type()[{i}].dtype, "CPU");\n'
                real_call_args_tensor[i] = "real_" + real_call_args_tensor[i]
            if is_tuple_tensor:
                cast_input += f'const auto &real_{tensor} = PyBoostUtils::CastTensor({tensor}, ' \
                              f'select_kernel.input_type()[{i}].dtype, "CPU");\n'
                real_call_args_tensor[i] = "PyBoostUtils::ConvertTensorVectorToTuple(real_" + real_call_args_tensor[
                    i] + ")"
        if cast_input != "":
            cast_input = "auto &select_kernel = kernel_attr_pair.second;\n" + cast_input
        return cast_input, real_call_args_tensor

    def _generate_create_input_address(self, op_parser: OpTemplateParser):
        need_malloc_tensors, _, _ = op_parser.parse_need_malloc_tensors()
        create_input_address = ''
        args_list = ''
        for item in need_malloc_tensors:
            args_list += f'{item}, '
        args_list = args_list[:-2]
        if args_list:
            create_input_address = f'PyBoostUtils::PrepareOpInputs(device_context_, op->stream_id(), {args_list});\n'
        return create_input_address

    def _generate_malloc_input(self, op_parser: OpTemplateParser):
        """
        Generates the code for creating input addresses for tensors that need to be allocated.

        Args:
            op_parser (OpTemplateParser): The parser object for the operation prototype.

        Returns:
            str: The generated code for creating input addresses.
        """
        need_malloc_tensors, _, _ = op_parser.parse_need_malloc_tensors()
        malloc_inputs = ''
        args_list = ''
        for item in need_malloc_tensors:
            args_list += f'{item}, '
        args_list = args_list[:-2]
        if args_list:
            malloc_inputs += f'PyBoostUtils::MallocOpInputs(device_context, {args_list});\n'
        return malloc_inputs

    def _generate_get_inputs_kernel_tensors(self, op_parser: OpTemplateParser):
        """
        Generates the code for retrieving input kernel tensors.

        Args:
            op_parser (OpTemplateParser): The parser object for the operation prototype.

        Returns:
            str: The generated code for retrieving input kernel tensors.
        """
        _, _, call_args_with_tensor = op_parser.parse_need_malloc_tensors()
        inputs_kernel_tensors = ''
        args_list = ''
        for item in call_args_with_tensor:
            args_list += f'{item}, '
        args_list = args_list[:-2]
        if args_list:
            inputs_kernel_tensors += f'const auto &input_address_info = PyBoostUtils::GetAddressInfo(' \
                                     f'device_context, op->stream_id(), op->input_abs(), {args_list});\n'
        return inputs_kernel_tensors


class PyboostOpFunctionGenerator(BaseGenerator):
    """
    Generates C++ source files for ACLNN operations in PyBoost.

    This class handles the generation of source files for operations that utilize the ACLNN framework,
    including customized calls and tensor management.

    Attributes:
        PYBOOST_CALL_TEMPLATE (Template): Template for generating ACLNN operation calls.
        PYBOOST_OP_SOURCE_TEMPLATE (Template): Template for generating operation source files.
        gen_path (str): Path for saving the generated C++ source files.
        device (str): The target device (ascend, cpu, or gpu).
    """

    def __init__(self):
        self.ascend_op_cpp_generator = PyboostOpCppGenerator('ascend')
        self.ascend_view_op_cpp_generator = PyboostViewOpCppGenerator('ascend')
        self.ascend_aclnn_cpp_generator = AclnnOpCppCodeGenerator('ascend')

        self.cpu_op_cpp_generator = PyboostOpCppGenerator('cpu')
        self.cpu_view_op_cpp_generator = PyboostViewOpCppGenerator('cpu')
        self.cpu_aclnn_cpp_generator = AclnnOpCppCodeGenerator('cpu')

        self.gpu_op_cpp_generator = PyboostOpCppGenerator('gpu')
        self.gpu_view_op_cpp_generator = PyboostViewOpCppGenerator('gpu')
        self.gpu_aclnn_cpp_generator = AclnnOpCppCodeGenerator('gpu')

        self.PYBOOST_ASCEND_OP_SOURCE_TEMPLATE = template.PYBOOST_ASCEND_OP_SOURCE_TEMPLATE
        self.PYBOOST_CPU_OP_SOURCE_TEMPLATE = template.PYBOOST_CPU_OP_SOURCE_TEMPLATE
        self.PYBOOST_GPU_OP_SOURCE_TEMPLATE = template.PYBOOST_GPU_OP_SOURCE_TEMPLATE
        self.ascend_gen_path = f"{K.MS_OPS_KERNEL_PATH}/ascend/pyboost/auto_generate/"
        self.cpu_gen_path = f"{K.MS_OPS_KERNEL_PATH}/cpu/pyboost/auto_generate/"
        self.gpu_gen_path = f"{K.MS_OPS_KERNEL_PATH}/gpu/pyboost/auto_generate/"

    def generate(self, work_path, op_protos):
        """
        Generate and save C++ source code for PyBoost operations across different devices.

        This method generates C++ source files for operations (`op_protos`) tailored to Ascend, CPU,
        and GPU devices. It combines headers and function implementations for each device, and then
        saves the final source files to the appropriate paths.

        Args:
            op_protos (list): A list of operation prototypes containing metadata such as
                              operation name, dispatch settings, arguments, and view attributes.
            work_path (str): The base working directory where the generated files will be saved.

        Generated Files:
            - Ascend: `pyboost_ascend_ops.cc`
            - CPU: `pyboost_cpu_ops.cc`
            - GPU: `pyboost_gpu_ops.cc`
        """
        self._generate_pyboost_ascend_ops(work_path, op_protos)
        self._generate_pyboost_cpu_ops(work_path, op_protos)
        self._generate_pyboost_gpu_ops(work_path, op_protos)

    def _generate_pyboost_ascend_ops(self, work_path, op_protos):
        """
        Generates Ascend PyBoost ops functions source files after being merged into specific chunk sizes.

        Args:
            work_path (str): The directory path where the generated C++ source files will be saved.
            op_protos (list): A list of operation prototypes that define the operations for which
                              the C++ code will be generated.
        """
        ascend_merge_op_header = []
        ascend_merge_op_function = []
        self.ascend_op_cpp_generator.generate_customize_op_cpp_code(op_protos, ascend_merge_op_header,
                                                                    ascend_merge_op_function)
        self.ascend_view_op_cpp_generator.generate_view_op_cpp_code(op_protos, ascend_merge_op_header,
                                                                    ascend_merge_op_function)
        self.ascend_aclnn_cpp_generator.generate_aclnn_op_cpp_code(op_protos, ascend_merge_op_header,
                                                                   ascend_merge_op_function)

        ascend_op_header_merge_by_chunk_size = merge_strings_by_chunk_size(ascend_merge_op_header, chunk_size=120)
        ascend_op_function_merge_by_chunk_size = merge_strings_by_chunk_size(ascend_merge_op_function, chunk_size=120)

        new_gen_num = len(ascend_op_header_merge_by_chunk_size)
        self._delete_residual_merged_ops_files(os.path.join(work_path, self.ascend_gen_path), new_gen_num)

        for i, op_header, op_function in zip(range(len(ascend_op_header_merge_by_chunk_size)),
                                             ascend_op_header_merge_by_chunk_size,
                                             ascend_op_function_merge_by_chunk_size):
            ascend_pyboost_op_source = self.PYBOOST_ASCEND_OP_SOURCE_TEMPLATE.replace(
                merge_op_header=op_header, merge_op_function=op_function)
            save_file(os.path.join(work_path, self.ascend_gen_path), f"pyboost_ascend_ops_{i}.cc",
                      ascend_pyboost_op_source)

    def _generate_pyboost_cpu_ops(self, work_path, op_protos):
        """
        Generates CPU PyBoost ops functions source files after being merged into specific chunk sizes.

        Args:
            work_path (str): The directory path where the generated C++ source files will be saved.
            op_protos (list): A list of operation prototypes that define the operations for which
                              the C++ code will be generated.
        """
        cpu_merge_op_header = []
        cpu_merge_op_function = []
        self.cpu_op_cpp_generator.generate_customize_op_cpp_code(op_protos, cpu_merge_op_header, cpu_merge_op_function)
        self.cpu_view_op_cpp_generator.generate_view_op_cpp_code(op_protos, cpu_merge_op_header, cpu_merge_op_function)
        self.cpu_aclnn_cpp_generator.generate_aclnn_op_cpp_code(op_protos, cpu_merge_op_header, cpu_merge_op_function)
        cpu_op_header_merge_by_chunk_size = merge_strings_by_chunk_size(cpu_merge_op_header, chunk_size=120)
        cpu_op_function_merge_by_chunk_size = merge_strings_by_chunk_size(cpu_merge_op_function, chunk_size=120)

        new_gen_num = len(cpu_op_header_merge_by_chunk_size)
        self._delete_residual_merged_ops_files(os.path.join(work_path, self.cpu_gen_path), new_gen_num)

        for i, op_header, op_function in zip(range(len(cpu_op_header_merge_by_chunk_size)),
                                             cpu_op_header_merge_by_chunk_size,
                                             cpu_op_function_merge_by_chunk_size):
            cpu_pyboost_op_source = self.PYBOOST_CPU_OP_SOURCE_TEMPLATE.replace(
                merge_op_header=op_header, merge_op_function=op_function)
            save_file(os.path.join(work_path, self.cpu_gen_path), f"pyboost_cpu_ops_{i}.cc",
                      cpu_pyboost_op_source)

    def _generate_pyboost_gpu_ops(self, work_path, op_protos):
        """
        Generates GPU PyBoost ops functions source files after being merged into specific chunk sizes.

        Args:
            work_path (str): The directory path where the generated C++ source files will be saved.
            op_protos (list): A list of operation prototypes that define the operations for which
                              the C++ code will be generated.
        """
        gpu_merge_op_header = []
        gpu_merge_op_function = []
        self.gpu_op_cpp_generator.generate_customize_op_cpp_code(op_protos, gpu_merge_op_header, gpu_merge_op_function)
        self.gpu_view_op_cpp_generator.generate_view_op_cpp_code(op_protos, gpu_merge_op_header, gpu_merge_op_function)
        self.gpu_aclnn_cpp_generator.generate_aclnn_op_cpp_code(op_protos, gpu_merge_op_header, gpu_merge_op_function)
        gpu_op_header_merge_by_chunk_size = merge_strings_by_chunk_size(gpu_merge_op_header, chunk_size=120)
        gpu_op_function_merge_by_chunk_size = merge_strings_by_chunk_size(gpu_merge_op_function, chunk_size=120)

        new_gen_num = len(gpu_op_header_merge_by_chunk_size)
        self._delete_residual_merged_ops_files(os.path.join(work_path, self.gpu_gen_path), new_gen_num)

        for i, op_header, op_function in zip(range(len(gpu_op_header_merge_by_chunk_size)),
                                             gpu_op_header_merge_by_chunk_size,
                                             gpu_op_function_merge_by_chunk_size):
            gpu_pyboost_op_source = self.PYBOOST_GPU_OP_SOURCE_TEMPLATE.replace(
                merge_op_header=op_header, merge_op_function=op_function)
            save_file(os.path.join(work_path, self.gpu_gen_path), f"pyboost_gpu_ops_{i}.cc",
                      gpu_pyboost_op_source)

    def _delete_residual_merged_ops_files(self, files_path, new_gen_num):
        """
        Deletes residual merged operation files in the specified directory if the number of
        newly generated files does not match the number of existing ones.

        This method first lists all files in the specified directory, then filters out the files
        that match the pattern `pyboost_.*_ops_.*.cc` (i.e., files related to pyboost ops). It compares
        the number of such files (`old_files_num`) with the `new_gen_num` argument, which represents
        the expected number of new pyboost ops files. If the counts do not match, the method will
        delete all the existing pyboost ops files in the directory before any new ones can be generated.

        Args:
            files_path (str): The path to the directory containing the files to be checked and deleted.
            new_gen_num (int): The number of newly generated pyboost ops files expected to be in the directory.

        Returns:
            None
        """
        all_files = os.listdir(files_path)
        old_pyboost_ops_files = [file for file in all_files if re.match(r'pyboost_.*_ops_.*\.cc', file)]
        old_files_num = len(old_pyboost_ops_files)
        if new_gen_num != old_files_num:
            for file in old_pyboost_ops_files:
                os.remove(os.path.join(files_path, file))


def _generate_cpp_func_return(op_proto):
    """Generates the C++ return type for the given operator prototype.

    Args:
        op_proto (OpProto): The operator prototype containing return information.

    Returns:
        str: The C++ return type for the function based on the operator prototype.

    Raises:
        Exception: If no return type is found.
    """
    returns_type = []
    type_convert_to_base = {
        'std::vector<mindspore::tensor::TensorPtr>': 'std::vector<mindspore::tensor::BaseTensorPtr>',
        'mindspore::tensor::TensorPtr': 'mindspore::tensor::BaseTensorPtr'
    }
    for return_obj in op_proto.op_returns:
        temp_return = get_return_type(return_obj.arg_dtype)
        if temp_return in type_convert_to_base:
            returns_type.append(type_convert_to_base[temp_return])
        else:
            raise Exception("Not return found")
    if len(returns_type) == 1:
        cpp_func_return = returns_type[0]
    elif len(returns_type) > 1:
        cpp_func_return = "std::tuple<"
        cpp_func_return += ','.join(s for s in returns_type)
        cpp_func_return += ">"
    else:
        raise Exception("Not return found")
    return cpp_func_return


def _generate_inplace_process_cpp_code(op_proto):
    """Generates C++ code for updating outputs by input tensors for inplace processing.

    Args:
        op_proto (OpProto): The operator prototype containing return information.

    Returns:
        str: The C++ code for inplace processing, or an empty string if no inplace processing is needed.
    """
    inplace_process = f'// RefOps update output by input tensor\n'
    has_ref = False
    for index, return_obj in enumerate(op_proto.op_returns):
        if return_obj.inplace != '':
            inplace_process += f'outputs_[{index}]->set_device_address(' \
                               f'{return_obj.inplace}_tensor->device_address()); '
            has_ref = True
            break
    if has_ref:
        return inplace_process
    return ''


def delete_residual_files(work_path, op_protos):
    """
    Deletes residual files generated for operator prototypes that are no longer needed.

    Args:
        work_path (str): The base directory path where generated files are located.
        op_protos (list): A list of operator prototypes that are currently valid.

    Returns:
        None
    """
    all_operator_name = []
    for op_proto in op_protos:
        all_operator_name.append(op_proto.op_name)
    code_generate_path_list = [f"{K.MS_OPS_KERNEL_PATH}/{device}/pyboost/auto_generate/" for device in
                               ["ascend", "gpu", "cpu"]]
    code_generate_path_list.append(f"{K.MS_COMMON_PYBOOST_KERNEL_PATH}/auto_generate/")
    for code_generate_path in code_generate_path_list:
        filter_files = []
        code_generate_path = os.path.join(work_path, code_generate_path)
        if os.path.exists(code_generate_path):
            all_files = os.listdir(code_generate_path)
            # No need to delete pyboost_.*_ops_.*.cc files and op_register.cc.
            # These residual files will be deleted before new files generate.
            filter_files = [file for file in all_files if
                            not re.match(r'pyboost_.*_ops_.*\.cc', file) and file != "op_register.cc"]
        registered_op_name = set(item.split(".")[0] for item in filter_files)
        need_clean_op = registered_op_name - set(all_operator_name)

        for file in filter_files:
            file_name = file.split(".")[0]
            if file_name in need_clean_op:
                file_path = os.path.join(code_generate_path, file)
                if os.path.exists(file_path):
                    os.remove(file_path)


class PyboostOpRegisterCppCodeGenerator:
    """
    Generates registration C++ code for PyBoost operations.

    This class is responsible for creating a registration source file that includes
    all the necessary headers and template instantiations for the registered operations.

    Attributes:
        PYBOOST_OP_REGISTER_TEMPLATE (Template): Template for generating the operation registration code.
    """

    def __init__(self):
        self.PYBOOST_OP_REGISTER_TEMPLATE = template.PYBOOST_OP_REGISTER_TEMPLATE

    def generate(self, work_path, op_protos):
        """
        Generates a C++ source file for registering all PyBoost operations.

        Args:
            work_path (str): The directory path where the registration file will be saved.
            op_protos (list): A list of operator prototypes containing information about the operations.

        Returns:
            None
        """
        all_op_names = []
        all_functional_names = []
        for op_proto in op_protos:
            if op_proto.op_dispatch is None:
                continue
            functional_name = op_proto.op_name
            op_name_str = op_proto.op_class.name
            all_op_names.append(op_name_str)
            all_functional_names.append(functional_name)

        include_str = ''
        factory_str = ''
        for op_name in all_op_names:
            factory_str += "template class OpFactory<{0}>;\n".format(op_name)
        for operator_name in all_functional_names:
            include_str += f'#include "{K.MS_COMMON_PYBOOST_KERNEL_PATH}/auto_generate/{operator_name}.h"\n'
        op_register_file_str = self.PYBOOST_OP_REGISTER_TEMPLATE.replace(op_includes=include_str,
                                                                         op_factory_templates=factory_str)
        save_path = os.path.join(work_path, f"{K.MS_COMMON_PYBOOST_KERNEL_PATH}/auto_generate/")
        file_name = "op_register.cc"
        save_file(save_path, file_name, op_register_file_str)
