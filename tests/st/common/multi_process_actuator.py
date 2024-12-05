import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback


def case_actuators(func, *args, **kwargs):
    """
    Feature: execute test case
    Description: execute test case function.
    Expectation: expect correct result which is bool.
    """
    flag_result = True
    param = ""
    for p in args:
        param = param + str(p)
    log_file_name = "%s_%s_error.log" % (func.__name__, param.replace(" ", ""))
    os.system(f"rm -rf {log_file_name}")
    try:
        func(*args, **kwargs)
    except (AssertionError, RuntimeError):
        flag_result = False
        with open(log_file_name, "a") as f:
            f.writelines(f"{func.__name__} fail {args} {kwargs}\n")
            traceback.print_exc(file=f)
    except (IOError, ValueError, EnvironmentError, ZeroDivisionError) as e:
        flag_result = False
        with open(log_file_name, "a") as f:
            f.writelines(f"{func.__name__} fail {args} {kwargs}\n")
            f.writelines(str(e))
    except Exception as e:
        flag_result = False
        with open(log_file_name, "a") as f:
            f.writelines(f"{func.__name__} fail {args} {kwargs}\n")
            f.writelines(str(e))
    finally:
        if not flag_result:
            with open(log_file_name, "r") as f:
                print("============================================ERROR============================================")
                print(f.read())
                print("=======================================ERROR PRINT END=======================================")
    return (f"{func} {args} {kwargs}", flag_result)


def get_max_worker():
    """
    Feature: get max workers number function.
    Description: get max workers number function.
    Expectation: expect correct result int.
    """
    max_worker_cnt = os.getenv("MS_DEV_ST_PARALLEL_RUN_COUNT", "8")
    return int(max_worker_cnt)


def run_cases_multi_process(test_case_list):
    """
    Feature: multiprocess function.
    Description: multiprocess function.
    Expectation: expect correct result list which members is (str, bool).
    """
    max_worker_cnt = get_max_worker()
    with ProcessPoolExecutor(max_workers=max_worker_cnt) as executor:
        all_task = [executor.submit(case_actuators, *p) for p in test_case_list]
        return [future.result() for future in as_completed(all_task)]
