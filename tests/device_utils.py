import os
import mindspore as ms

def set_device():
    devcie_target = os.getenv("DEVICE_TARGET")
    device_id = os.getenv("DEVICE_ID")
    if device_id is None:
        ms.set_device(devcie_target)
    else:
        ms.set_device(devcie_target, int(device_id))


def get_device():
    devcie_target = os.getenv("DEVICE_TARGET")
    return devcie_target
