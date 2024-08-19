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

from mindspore._c_expression import GraphExecutor_ as graph_exec

sub_a_1 = [
    'switch_defer_inline',
    'switch_layer_defer_inline',
    'switch_simplify',
    'exchange_switch_depend_value',
    'float_depend_g_call',
    # Safe inlining
    'inline',
    'updatestate_useless_node_eliminater',
    'updatestate_pure_node_eliminater',
    'load_eliminater',
    'stopgrad_eliminater',
    'partial_eliminate',
    'replace_applicator',
    # Miscellaneous
    'tuple_list_get_item_eliminator',
    'make_slice_get_slice_eliminator',
    'tuple_list_get_item_const_eliminator',
    'tuple_list_set_item_eliminator',
    'tuple_list_get_set_item_eliminator',
    'tuple_list_get_item_depend_reorder',
    'tuple_list_convert_item_index_to_positive',
    # environ
    'environ_get_eliminate',
    'environ_get_add_eliminate',
    'environ_get_set_eliminate',
    'environ_get_depend_swap',
    'environ_add_const_eliminate',
    #
    'cast_eliminate',
    'reshape_eliminate',
    'reduce_eliminate',
    'tile_eliminate',
    'transpose_eliminate',
    'minmaximum_grad',
    # Arithmetic simplifications
    'arithmetic_simplify',
    # 'addn_zero_filter',
    # 'adjust_all_reduce_mul_add',
    # 'accumulaten_eliminater',
    # Safe inlining
    'inline',
    'updatestate_useless_node_eliminater',
    'updatestate_pure_node_eliminater',
    'load_eliminater',
    'stopgrad_eliminater'
]

sub_a_2 = [
    'switch_simplify',
    'specialize_transform',
    # 'merge_addn',
    'compare_switch_simplify',
    'addn_check_dump',
    'float_tuple_getitem_switch',
    'float_environ_get_switch',
    'inline',
    'updatestate_useless_node_eliminater',
    'tuple_list_set_item_eliminator',
    'tuple_list_get_item_eliminator',
    'incorporate_call',
    'incorporate_call_switch',
    'environ_get_eliminate',
    'depend_value_elim',
    'reduce_all_const_elim'
]

sub_a_3 = [
    'same_eliminate',
    'check_bprop_eliminate',
    'switch_layer_defer_inline',
    'replace_applicator',
    'row_tensor_add_zeros_like',
    'mini_step_allgather_replace',
    'micro_step_allgather_replace',
    'split_environ_get_set_with_tuple_value'
]

sub_b_1 = [
    'zero_like_fill_zero',
    'tuple_list_get_item_eliminator',
    'tuple_list_get_item_const_eliminator',
    'tuple_list_set_item_eliminator',
    'tuple_list_get_set_item_eliminator',
    'tuple_list_get_item_depend_reorder',
    'tuple_list_convert_item_index_to_positive',
    'make_slice_get_slice_eliminator',
    'float_tuple_getitem_switch',
    'reset_defer_inline',
    'inline',
    'updatestate_useless_node_eliminater',
    'updatestate_pure_node_eliminater',
    'load_eliminater',
    'stopgrad_eliminater',
    'special_op_eliminate',
    'environ_get_eliminate',
    'environ_get_add_eliminate',
    'environ_get_set_eliminate',
    'environ_get_depend_swap',
    'environ_add_const_eliminate',
    'value_based_eliminate',
    'parallel_virtual_node'
]

sub_b_2 = ['row_tensor_eliminate']

sub_c_1 = [
    # Safe inlining,
    'inline',
    'updatestate_useless_node_eliminater',
    'updatestate_pure_node_eliminater',
    'load_eliminater',
    'switch_call_monad_eliminater',
    'stopgrad_eliminater',
    'partial_eliminate'
]

sub_c_1 = [
    # Safe inlining,
    'inline',
    'updatestate_useless_node_eliminater',
    'updatestate_pure_node_eliminater',
    'load_eliminater',
    'switch_call_monad_eliminater',
    'stopgrad_eliminater',
    'partial_eliminate'
]

sub_d_1 = [
    'call_graph_tuple_transform',
    'tuple_list_get_item_eliminator',
    'tuple_list_get_item_const_eliminator',
    'tuple_list_set_item_eliminator',
    'tuple_list_get_set_item_eliminator',
    'tuple_list_get_item_depend_reorder',
    'tuple_list_convert_item_index_to_positive'
]

a_1 = {
    "name": "a_1",
    "once": False,
    "renormalize": False,
    "sensitive": False,
    "list": sub_a_1
}

a_2 = {
    "name": "a_2",
    "once": False,
    "renormalize": False,
    "sensitive": True,
    "list": sub_a_2
}

a_3 = {
    "name": "a_3",
    "once": False,
    "renormalize": False,
    "sensitive": True,
    "list": sub_a_3
}

expand_dump_flag = {"name": "expand_dump_flag", "pass_func": "ExpandDumpFlag"}
switch_simplify = {"name": "switch_simplify", "list": ["switch_simplify"]}
recompute_prepare = {"name": "recompute_prepare", "list": ["set_cell_output_no_recompute"]}
updatestate_depend_eliminate = {"name": "updatestate_depend_eliminate", "pass_func": "UpdatestateDependEliminater"}
updatestate_assign_eliminate = {"name": "updatestate_assign_eliminate", "pass_func": "UpdatestateAssignEliminater"}
updatestate_loads_eliminate = {"name": "updatestate_loads_eliminate", "pass_func": "UpdatestateLoadsEliminater"}
parameter_eliminate = {"name": "parameter_eliminate", "pass_func": "ParameterEliminator"}
accelerated_algorithm = {"name": "accelerated_algorithm", "list": ["less_batch_normalization"]}
virtual_dataset = {"name": "virtual_dataset", "list": ["virtual_dataset_eliminate"]}
virtual_output = {"name": "virtual_output", "list": ["virtual_output_eliminate"]}
after_resolve = {"name": "after_resolve", "list": ["replace_old_param"]}
meta_fg_expand = {"name": "meta_fg_expand", "pass_func": "ExpandMetaFg"}
a_after_grad = {"name": "after_grad", "list": ["inline_without_move", "stack_unstack_eliminate"]}
renormalize = {"name": "renormalize", "renormalize": True}
real_op_eliminate = {"name": "real_op_eliminate", "list": ["real_op_eliminate"]}
auto_monad_grad = {"name": "auto_monad_grad", "pass_func": "ReAutoMonadWrapper"}
auto_monad_eliminator = {"name": "auto_monad_eliminator", "pass_func": "AutoMonadEliminator"}
cse = {"name": "cse", "pass_func": "CSEPass"}

pass_group_a = [
    expand_dump_flag,
    switch_simplify,
    a_1,
    recompute_prepare,
    updatestate_depend_eliminate,
    updatestate_assign_eliminate,
    updatestate_loads_eliminate,
    parameter_eliminate,
    a_2,
    accelerated_algorithm,
    # pynative_shard,
    # auto_parallel,
    # parallel,
    # allreduce_fusion,
    virtual_dataset,
    virtual_output,
    meta_fg_expand,
    after_resolve,
    a_after_grad,
    renormalize,
    real_op_eliminate,
    auto_monad_grad,
    auto_monad_eliminator,
    # cse,
    a_3,
]

opt_a = {'name': 'opt_a', 'once': False, 'renormalize': False, 'node_first': True, 'pass_group': pass_group_a}


def set_optimize_config():
    passes = graph_exec.get_instance().get_optimize_config()
    print(passes)
    cconv = {'name': 'ccovn', 'fun': 'CconvPass'}
    pass_config = [opt_a, cconv]
    graph_exec.get_instance().set_optimize_config(pass_config)

    print("-----------------------------------------------")
    passes = graph_exec.get_instance().get_optimize_config()
    print(passes)


def clear_optimize_config():
    graph_exec.get_instance().set_optimize_config([])
