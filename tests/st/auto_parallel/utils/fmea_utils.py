import filecmp
import os


def check_key_word_in_file(key_word, file_name):
    try:
        with open(file_name, 'r') as file:
            for line in file:
                if key_word in line:
                    print(f"Found the key word: {line.strip()}")
                    return True
        print("No matching error found.")
        return False
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
        return False


def check_comm_order_same(rank_size, log_path, remove_log=True):
    log_files = []
    for rank_id in range(rank_size):
        worker_log = "{}/worker_{}.log".format(log_path, rank_id)
        comm_log = "worker_{}_comm.log".format(rank_id)
        ret = os.system(
            "grep -Po PrintGraphExecuteOrder.* {} | grep All | grep -o 'All[^ ]*-op' &> {}".format(worker_log,
                                                                                                   comm_log))
        assert ret == 0
        log_files.append(comm_log)
    reference_file = log_files[0]
    all_same = all(filecmp.cmp(reference_file, file, shallow=False) for file in log_files[1:])
    if all_same:
        print("The execution order of communication operators in all workers is the same.")
        if remove_log:
            os.system("rm -rf worker_*_comm.log")
        return True
    print("The execution order of communication operators in different workers is different, please check model.")
    if remove_log:
        os.system("rm -rf worker_*_comm.log")
    return False
