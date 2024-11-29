import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback


def case_actuators(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        print(f"{func.__name__} pass")
    except (AssertionError, RuntimeError):
        print(f"{func.__name__} fail \n")
        with open(f"{func.__name__}_error.log", "a") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        return False
    except (IOError, ValueError, EnvironmentError, ZeroDivisionError) as e:
        print(f"{func.__name__} fail \n {e}")
        with open(f"{func.__name__}_error.log", "a") as f:
            f.writelines(e)
        return False
    except Exception as e:
        print(f"{func.__name__} fail \n {e}")
        with open(f"{func.__name__}_error.log", "a") as f:
            f.writelines(e)
        return False
    return True


def run_cases_multi_process(test_case_list):
    """
    Feature: multiprocess function.
    Description: multiprocess function.
    Expectation: expect correct result list which members is bool.
    """
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        all_task = [executor.submit(case_actuators, *p) for p in test_case_list]
        return [future.result() for future in as_completed(all_task)]
