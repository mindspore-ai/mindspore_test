from mindspore import context
from mindspore.communication.management import init
from mindspore.communication.management import release
from mindspore.communication.management import get_rank
import os
from mindspore import log as logger


env_dist = os.environ
def find_file(filename, search_path):
    result = []
    for root, _, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result

class MetaFactory:
    def __init__(self):
        self._default_context()
        self._default_ps_context(False)
        self._set_context_from_env()
        self.device_target = context.get_context('device_target')
        self.rank_size = None
        self.device_id = os.environ.get('DEVICE_ID')
        if self.device_id is not None:
            self.device_id = int(self.device_id)
        self.global_rank_id = None
        self._init_parallel()
        self._set_parallel_env()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def __del__(self):
        self._release_parallel()

    @staticmethod
    def _default_context():
        context.set_context(mode=context.GRAPH_MODE)
        if 'CONTEXT_DEVICE_TARGET' in os.environ:
            context.set_context(device_target=os.environ['CONTEXT_DEVICE_TARGET'])
        # else:
            # raise RuntimeError

    @staticmethod
    def _default_ps_context(enable_ssl):
        security_ctx = {
            "config_file_path": os.path.join(os.path.dirname(__file__), "config.json"),
            "enable_ssl": False
        }
        context.set_ps_context(**security_ctx)

    @staticmethod
    def _set_context_from_env():
        mode_dict = {
            'GRAPH': context.GRAPH_MODE,
            'GRAPH_MODE': context.GRAPH_MODE,
            'CONTEXT.GRAPH_MODE': context.GRAPH_MODE,
            'PYNATIVE': context.PYNATIVE_MODE,
            'PYNATIVE_MODE': context.PYNATIVE_MODE,
            'CONTEXT.PYNATIVE_MODE': context.PYNATIVE_MODE
        }
        if 'CONTEXT_MODE' in os.environ:
            mode_key = os.environ['CONTEXT_MODE'].upper()
            context.set_context(mode=mode_dict[mode_key])
        if 'CONTEXT_DEVICE_TARGET' in os.environ:
            context.set_context(device_target=os.environ['CONTEXT_DEVICE_TARGET'])
        if 'CONTEXT_ENABLE_SPARSE' in os.environ:
            context.set_context(enable_sparse=True)
        if 'CONTEXT_ENABLE_GRAPH_KERNEL' in os.environ:
            context.set_context(enable_graph_kernel=True)
        # GRAPH_OP_RUN is only for jenkeins
        # export GRAPH_OP_RUN=1 --> KBK: jit_level=O0
        # export GRAPH_OP_RUN=DVM      : jit_level=O1
        # export GRAPH_OP_RUN=0 --> GE : jit_level=O2
        # unset  GRAPH_OP_RUN default  :
        if "GRAPH_OP_RUN" not in os.environ:
            # not set default GE
            logger.info("910a:GE 910b:kbk dynamic_shape:kbk")
        elif os.environ.get("GRAPH_OP_RUN") == "1":
            context.set_context(jit_config={"jit_level": "O0"})
        elif os.environ.get("GRAPH_OP_RUN") == "DVM":
            context.set_context(jit_config={"jit_level": "O1"})
        elif os.environ.get("GRAPH_OP_RUN") == "0":
            context.set_context(jit_config={"jit_level": "O2"})
        else:
            msg = "GRAPH_OP_RUN=0/1/DVM but get " + os.environ.get("GRAPH_OP_RUN")
            raise ValueError(msg)

    def _set_parallel_env(self):
        if 'RANK_SIZE' in os.environ:
            self.rank_size = int(os.environ['RANK_SIZE'])
            self.global_rank_id = get_rank()
            self.device_id = get_rank()

    def _init_parallel(self):
        self._init_parallel_flag = False
        if 'RANK_SIZE' in os.environ:
            init()
            self._init_parallel_flag = True

    def _release_parallel(self):
        if self._init_parallel_flag:
            release()

    @staticmethod
    def _save_graphs(save_graph_flag=False, save_graph_path="."):
        context.set_context(save_graphs=save_graph_flag, save_graphs_path=save_graph_path)

