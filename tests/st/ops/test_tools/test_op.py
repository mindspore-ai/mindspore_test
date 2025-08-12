from tests.st.ops.test_tools.ops_dynamic_cases import TEST_OP_DYNAMIC
from tests.st.ops.test_tools.ops_dynamic_cases import CASE_CONFIGS as DYNAMIC_CASE_CONFIGS
from tests.st.ops.test_tools.ops_generalize_cases import TEST_OP_GENERALIZE
from tests.st.ops.test_tools.ops_generalize_cases import CASE_NAMES as GENERALIZE_CASE_NAMES
from tests.st.ops.test_tools.ops_generalize_cases import CASE_CONFIGS as GENERALIZE_CASE_CONFIGS


RUNNING_MODES = [
    "PYNATIVE_MODE",
    "GRAPH_MODE_O0",
    "GRAPH_MODE_GE",
]
DEBUG_STATUS_INFO = ""


def set_debug_status_info(case_name):
    global DEBUG_STATUS_INFO
    DEBUG_STATUS_INFO = f"{case_name}"


def error_status_log():
    global DEBUG_STATUS_INFO
    print(f"\n[ERROR]: TEST_OP catch a error during testing {DEBUG_STATUS_INFO}."
          f"\nPlease use these parameters to quickly reproduce the error:")

    if 'TEST_OP_DYNAMIC' in DEBUG_STATUS_INFO:
        print("disable_generalize_test=True")

    elif 'TEST_OP_GENERALIZE' in DEBUG_STATUS_INFO:
        print("disable_dynamic_test=True")

    print("For more information, set dump_ir=True to get ir graphs or set debug_info=True to get more debug messages.")


def warning_log(*args):
    print("[WARNING]: TEST_OP ", *args)


def error_log(error, message):
    raise error(f"[ERROR]: TEST_OP " + message)


def check_args(inputs_seq, disable_dynamic_test, disable_generalize_test, disable_mode, disable_case, case_config,
               inplace_update, dump_ir, debug_info, debug_level):
    if not isinstance(inputs_seq, list):
        error_log(TypeError, f"'inputs_seq' must be type of [list], but got {type(inputs_seq)}.")

    if len(inputs_seq) != 2:
        error_log(RuntimeError, f"For complete test, you must provide 2 groups of inputs, but got {len(inputs_seq)}.")

    if len(inputs_seq[0]) != len(inputs_seq[1]):
        error_log(RuntimeError, f"For complete test, two inputs_seq you provided must have same length.")

    if not isinstance(disable_mode, list):
        error_log(ValueError, f"'disable_mode' must be a list, but got {type(disable_mode)}.")
    for mode in disable_mode:
        if not isinstance(mode, str) or mode not in RUNNING_MODES:
            error_log(ValueError, f"elements of 'disable_mode' must be in {RUNNING_MODES}, but got {mode}.")

    if not isinstance(disable_case, list):
        error_log(ValueError, f"'disable_case' must be a list, but got {type(disable_case)}.")

    all_case_names = []
    all_case_names += GENERALIZE_CASE_NAMES
    for case in disable_case:
        if not isinstance(case, str) or case not in all_case_names:
            error_log(ValueError, f"elements of 'disable_case' must be in {all_case_names}, but got {case}.")

    if not isinstance(disable_dynamic_test, bool):
        raise TypeError(f"'disable_dynamic_test' must be bool, but got {type(disable_dynamic_test)}.")
    if not isinstance(disable_generalize_test, bool):
        raise TypeError(f"'disable_generalize_test' must be bool, but got {type(disable_generalize_test)}.")
    if case_config is not None:
        if not isinstance(case_config, dict):
            error_log(ValueError, f"'case_config' must be a dict, but got {type(case_config)}.")
        valid_keys = []
        valid_keys += list(config[0] for config in DYNAMIC_CASE_CONFIGS)
        valid_keys += list(config[0] for config in GENERALIZE_CASE_CONFIGS)
        valid_types = []
        valid_types += list(config[1] for config in DYNAMIC_CASE_CONFIGS)
        valid_types += list(config[1] for config in GENERALIZE_CASE_CONFIGS)
        for key, value in case_config.items():
            if key not in valid_keys:
                error_log(ValueError, f"{key} is not a valid key for 'case_config', valid key is {valid_keys}.")
            for idx, valid_key in enumerate(valid_keys):
                if key == valid_key and not isinstance(value, valid_types[idx]):
                    error_log(ValueError, f"'{key}' must be {valid_types[idx]}, but got {type(value)}.")

    if not isinstance(inplace_update, bool):
        error_log(ValueError, f"'inplace_update' must be bool, but got {type(inplace_update)}.")
    if not isinstance(dump_ir, bool):
        error_log(ValueError, f"'dump_ir' must be bool, but got {type(dump_ir)}.")
    if not isinstance(debug_info, bool):
        error_log(ValueError, f"'debug_info' must be bool, but got {type(debug_info)}.")
    if not isinstance(debug_level, int):
        error_log(ValueError, f"'debug_level' must be int, but got {type(debug_level)}.")


def run_op_dynamic(op, inputs, disable_dynamic_test, disable_mode, case_config, inplace_update, dump_ir, debug_info,
                   debug_level):
    if disable_dynamic_test:
        warning_log(f"disable_dynamic_test is True, TEST_OP_DYNAMIC is skipped.")
        return

    set_debug_status_info('TEST_OP_DYNAMIC')
    print(f"\n*************************************************************************************************\n"
          f"**************************************TEST_OP_DYNAMIC Start**************************************\n"
          f"*************************************************************************************************\n")
    TEST_OP_DYNAMIC(op, inputs, disable_mode=disable_mode, case_config=case_config, inplace_update=inplace_update,
                    dump_ir=dump_ir, debug_info=debug_info, debug_level=debug_level)
    print(f"\n*************************************************************************************************\n"
          f"***************************************TEST_OP_DYNAMIC End***************************************\n"
          f"*************************************************************************************************\n")


def run_op_generalize(op, inputs, disable_generalize_test, disable_mode, disable_case, case_config, inplace_update,
                      dump_ir, debug_info, debug_level):
    if disable_generalize_test:
        warning_log(f"disable_generalize_test is True, TEST_OP_GENERALIZE is skipped.")
        return

    set_debug_status_info('TEST_OP_GENERALIZE')
    print(f"\n**************************************************************************************************\n"
          f"*************************************TEST_OP_GENERALIZE Start*************************************\n"
          f"**************************************************************************************************\n")
    TEST_OP_GENERALIZE(op, inputs, disable_mode=disable_mode, disable_case=disable_case, case_config=case_config,
                       inplace_update=inplace_update, dump_ir=dump_ir, debug_info=debug_info, debug_level=debug_level)
    print(f"\n**************************************************************************************************\n"
          f"**************************************TEST_OP_GENERALIZE End**************************************\n"
          f"**************************************************************************************************\n")


def parse_case_config(case_config):
    dynamic_case_config = None
    generalize_case_config = None

    if case_config:
        dynamic_case_config_keys = list(config[0] for config in DYNAMIC_CASE_CONFIGS)
        generalize_case_config_keys = list(config[0] for config in GENERALIZE_CASE_CONFIGS)
        dynamic_case_config = {k: v for k, v in case_config.items() if k in dynamic_case_config_keys}
        generalize_case_config = {k: v for k, v in case_config.items() if k in generalize_case_config_keys}

    if dynamic_case_config == {}:
        dynamic_case_config = None
    if generalize_case_config == {}:
        generalize_case_config = None

    return dynamic_case_config, generalize_case_config


def TEST_OP(op, inputs_seq, *, disable_generalize_test=False, disable_dynamic_test=False, disable_mode=[],
            disable_case=[], case_config=None, inplace_update=False, dump_ir=False, debug_info=False, debug_level=0):
    """
    This function is an operator testing tool, which includes dynamic case and generalized case.

    Args:
        op (Union[Primitive, Function]): The operator instance to be test.
        inputs_seq (list[list, list]): The inputs(attribute is not needed) of the operator.
            `inputs_seq` contains two groups of data: the first group will be used to running in the all dynamic cases,
            and the second will only be used to test `Resize`. The two groups of data should have the same type
            accordingly: if data is a Tensor, data with same dtype and different rank of shape should be given.
            if data is tuple/list/scalar, data only with different value should be given. e.g.: For ReduceSum,
            the two groups of inputs could be `[[Tensor(shape=[2, 3, 4], dtype=float32), [0]],
            [Tensor(shape=[4, 3, 2, 4], dtype=float32), [1]]`.

    Keyword Args:
        disable_generalize_test (bool): Whether run TEST_OP_GENERALIZE case.
            If ``False`` , TEST_OP_GENERALIZE will run.
            If ``True`` , `TEST_OP_GENERALIZE will not run.
            Default: ``False`` .
        disable_dynamic_test (bool): Whether run TEST_OP_DYNAMIC case.
            If ``False`` , TEST_OP_DYNAMIC will run.
            If ``True`` , `TEST_OP_DYNAMIC will not run.
            Default: ``False`` .
        disable_mode (list[str]): Disable the given running mode.
            If ``PYNATIVE_MODE`` , PYNATIVE_MODE will not set as running mode.
            If ``GRAPH_MODE_O0`` , GRAPH_MODE with jit_level=O0 will not set as running mode, kernel by kernel will be
            enabled on this running mode.
            If ``GRAPH_MODE_GE`` , GRAPH_MODE with jit_level=O2 will not set as running mode, ge backend will be enabled
            on this running mode.
            Default: ``[]`` .
        disable_case (list[str]): Disable the specified test cases.
            For more details, please refer to TEST_OP_GENERALIZE.
        case_config (dict): Disable the specified test case.
            For more details, please refer to TEST_OP_DYNAMIC and TEST_OP_GENERALIZE.
        inplace_update (bool): Whether the op updates its inputs. Default ``False`` .
        dump_ir (bool): Whether to save the ir_graphs during test.
            If ``False`` , no ir_graphs will be generated.
            If ``True`` , `save_graphs` will be set and save_graphs_path is generated by Primitive's name and dynamic
            type.
            Default: ``False`` .
        debug_info (bool): Whether to print more debug information. Default ``False`` .
        debug_level (int): Set the level for debug infos, level should be in [0, 1].
            If ``0`` , print shape and dtype of Tensor args.
            If ``1`` , print shape, dtype and actual values of Tensor args.
            Default: ``0`` .
    """

    check_args(inputs_seq, disable_dynamic_test, disable_generalize_test, disable_mode, disable_case, case_config,
               inplace_update, dump_ir, debug_info, debug_level)

    dynamic_case_config, generalize_case_config = parse_case_config(case_config)

    try:
        run_op_generalize(op, inputs_seq[0], disable_generalize_test, disable_mode, disable_case,
                          generalize_case_config, inplace_update, dump_ir, debug_info, debug_level)
        run_op_dynamic(op, inputs_seq, disable_dynamic_test, disable_mode, dynamic_case_config, inplace_update,
                       dump_ir, debug_info, debug_level)
    except Exception as error:
        error_status_log()
        raise error
