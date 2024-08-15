/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "transform/graph_ir/op_declare/nn_other_ops_declare.h"
#include <vector>
#include <string>
#include "transform/graph_ir/op_declare/op_declare_macro.h"
namespace mindspore::transform {
// InitPartitionMap
INPUT_MAP(InitPartitionMap) = {{1, INPUT_DESC(ps_num)}, {2, INPUT_DESC(ps_ids)}};
ATTR_MAP(InitPartitionMap) = {{"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
                              {"partition_num", ATTR_DESC(partition_num, AnyTraits<int64_t>())},
                              {"_embedding_dim", ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
                              {"_max_key_num", ATTR_DESC(_max_key_num, AnyTraits<int64_t>())},
                              {"_ps_num", ATTR_DESC(_ps_num, AnyTraits<int64_t>())}};
OUTPUT_MAP(InitPartitionMap) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(InitPartitionMap, kNameInitPartitionMap, ADPT_DESC(InitPartitionMap))

// InitEmbeddingHashmap
INPUT_MAP(InitEmbeddingHashmap) = {{1, INPUT_DESC(table_id)}};
ATTR_MAP(InitEmbeddingHashmap) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
  {"value_total_len", ATTR_DESC(value_total_len, AnyTraits<int64_t>())},
  {"embedding_dim", ATTR_DESC(embedding_dim, AnyTraits<int64_t>())},
  {"bucket_size", ATTR_DESC(bucket_size, AnyTraits<int64_t>())},
  {"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())},
  {"initializer_mode", ATTR_DESC(initializer_mode, AnyTraits<std::string>())},
  {"constant_value", ATTR_DESC(constant_value, AnyTraits<float>())},
  {"min", ATTR_DESC(min, AnyTraits<float>())},
  {"max", ATTR_DESC(max, AnyTraits<float>())},
  {"mu", ATTR_DESC(mu, AnyTraits<float>())},
  {"sigma", ATTR_DESC(sigma, AnyTraits<float>())},
  {"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
  {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())},
  {"filter_mode", ATTR_DESC(filter_mode, AnyTraits<std::string>())},
  {"optimizer_mode", ATTR_DESC(optimizer_mode, AnyTraits<std::string>())},
  {"optimizer_params", ATTR_DESC(optimizer_params, AnyTraits<std::vector<float>>())},
  {"_table_id", ATTR_DESC(_table_id, AnyTraits<int64_t>())}};
OUTPUT_MAP(InitEmbeddingHashmap) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(InitEmbeddingHashmap, kNameInitEmbeddingHashmap, ADPT_DESC(InitEmbeddingHashmap))

// EmbeddingTableFind
INPUT_MAP(EmbeddingTableFind) = {{1, INPUT_DESC(table_id)}, {2, INPUT_DESC(keys)}};
ATTR_MAP(EmbeddingTableFind) = {
  {"embedding_dim", ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
  {"default_value", ATTR_DESC(default_value, AnyTraits<std::vector<float>>())},
  {"_embedding_dim", ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
  {"_max_key_num", ATTR_DESC(_max_key_num, AnyTraits<int64_t>())},
  {"_use_counter_filter", ATTR_DESC(_use_counter_filter, AnyTraits<int64_t>())},
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())}};
OUTPUT_MAP(EmbeddingTableFind) = {{0, OUTPUT_DESC(values)}};
REG_ADPT_DESC(EmbeddingTableFind, kNameEmbeddingTableFind, ADPT_DESC(EmbeddingTableFind))

// EmbeddingTableFindAndInit
INPUT_MAP(EmbeddingTableFindAndInit) = {{1, INPUT_DESC(table_id)}, {2, INPUT_DESC(keys)}};
ATTR_MAP(EmbeddingTableFindAndInit) = {
  {"embedding_dim", ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
  {"value_total_len", ATTR_DESC(value_total_len, AnyTraits<std::vector<int64_t>>())},
  {"initializer_mode", ATTR_DESC(initializer_mode, AnyTraits<std::vector<std::string>>())},
  {"constant_value", ATTR_DESC(constant_value, AnyTraits<std::vector<float>>())},
  {"min", ATTR_DESC(min, AnyTraits<std::vector<float>>())},
  {"max", ATTR_DESC(max, AnyTraits<std::vector<float>>())},
  {"mu", ATTR_DESC(mu, AnyTraits<std::vector<float>>())},
  {"sigma", ATTR_DESC(sigma, AnyTraits<std::vector<float>>())},
  {"seed", ATTR_DESC(seed, AnyTraits<std::vector<int64_t>>())},
  {"seed2", ATTR_DESC(seed2, AnyTraits<std::vector<int64_t>>())},
  {"filter_mode", ATTR_DESC(filter_mode, AnyTraits<std::vector<std::string>>())},
  {"filter_freq", ATTR_DESC(filter_freq, AnyTraits<std::vector<int64_t>>())},
  {"default_key_or_value", ATTR_DESC(default_key_or_value, AnyTraits<std::vector<int64_t>>())},
  {"default_key", ATTR_DESC(default_key, AnyTraits<std::vector<int64_t>>())},
  {"default_value", ATTR_DESC(default_value, AnyTraits<std::vector<float>>())},
  {"completion_key", ATTR_DESC(completion_key, AnyTraits<std::vector<int64_t>>())},
  {"completion_key_mask", ATTR_DESC(completion_key_mask, AnyTraits<std::vector<int64_t>>())},
  {"optimizer_mode", ATTR_DESC(optimizer_mode, AnyTraits<std::vector<std::string>>())},
  {"optimizer_params", ATTR_DESC(optimizer_params, AnyTraits<std::vector<float>>())},
  {"_embedding_dim", ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
  {"_max_key_num", ATTR_DESC(_max_key_num, AnyTraits<int64_t>())},
  {"_use_counter_filter", ATTR_DESC(_use_counter_filter, AnyTraits<int64_t>())},
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
  {"_execute_times", ATTR_DESC(_execute_times, AnyTraits<int64_t>())}};
OUTPUT_MAP(EmbeddingTableFindAndInit) = {{0, OUTPUT_DESC(values)}};
REG_ADPT_DESC(EmbeddingTableFindAndInit, kNameEmbeddingTableFindAndInit, ADPT_DESC(EmbeddingTableFindAndInit))

// EmbeddingApplyFtrl
INPUT_MAP(EmbeddingApplyFtrl) = {{1, INPUT_DESC(var_handle)}, {2, INPUT_DESC(lr)},         {3, INPUT_DESC(lr_power)},
                                 {4, INPUT_DESC(lambda1)},    {5, INPUT_DESC(lambda2)},    {6, INPUT_DESC(grad)},
                                 {7, INPUT_DESC(keys)},       {8, INPUT_DESC(global_step)}};
INPUT_ATTR_MAP(EmbeddingApplyFtrl) = {{9, ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
                                      {10, ATTR_DESC(mask_zero, AnyTraits<std::vector<int64_t>>())},
                                      {11, ATTR_DESC(padding_key, AnyTraits<std::vector<int64_t>>())},
                                      {12, ATTR_DESC(padding_key_mask, AnyTraits<std::vector<int64_t>>())},
                                      {13, ATTR_DESC(completion_key, AnyTraits<std::vector<int64_t>>())},
                                      {14, ATTR_DESC(completion_key_mask, AnyTraits<std::vector<int64_t>>())},
                                      {15, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
                                      {16, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingApplyFtrl) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
};
OUTPUT_MAP(EmbeddingApplyFtrl) = {{0, OUTPUT_DESC(var_handle)}};
REG_ADPT_DESC(EmbeddingApplyFtrl, ops::kNameEmbeddingApplyFtrl, ADPT_DESC(EmbeddingApplyFtrl))

// EmbeddingApplyAdam
INPUT_MAP(EmbeddingApplyAdam) = {
  {1, INPUT_DESC(var_handle)}, {2, INPUT_DESC(beta1_power)}, {3, INPUT_DESC(beta2_power)}, {4, INPUT_DESC(lr)},
  {5, INPUT_DESC(beta1)},      {6, INPUT_DESC(beta2)},       {7, INPUT_DESC(epsilon)},     {8, INPUT_DESC(grad)},
  {9, INPUT_DESC(keys)},       {10, INPUT_DESC(global_step)}};
INPUT_ATTR_MAP(EmbeddingApplyAdam) = {{11, ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
                                      {12, ATTR_DESC(mask_zero, AnyTraits<std::vector<int64_t>>())},
                                      {13, ATTR_DESC(padding_key, AnyTraits<std::vector<int64_t>>())},
                                      {14, ATTR_DESC(padding_key_mask, AnyTraits<std::vector<int64_t>>())},
                                      {15, ATTR_DESC(completion_key, AnyTraits<std::vector<int64_t>>())},
                                      {16, ATTR_DESC(completion_key_mask, AnyTraits<std::vector<int64_t>>())},
                                      {17, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
                                      {18, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingApplyAdam) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())}};
OUTPUT_MAP(EmbeddingApplyAdam) = {{0, OUTPUT_DESC(var_handle)}};
REG_ADPT_DESC(EmbeddingApplyAdam, ops::kNameEmbeddingApplyAdam, ADPT_DESC(EmbeddingApplyAdam))

// EmbeddingApplyAdamW
INPUT_MAP(EmbeddingApplyAdamW) = {
  {1, INPUT_DESC(var_handle)}, {2, INPUT_DESC(beta1_power)},    {3, INPUT_DESC(beta2_power)},
  {4, INPUT_DESC(lr)},         {5, INPUT_DESC(weight_decay)},   {6, INPUT_DESC(beta1)},
  {7, INPUT_DESC(beta2)},      {8, INPUT_DESC(epsilon)},        {9, INPUT_DESC(grad)},
  {10, INPUT_DESC(keys)},      {11, INPUT_DESC(max_grad_norm)}, {12, INPUT_DESC(global_step)}};
INPUT_ATTR_MAP(EmbeddingApplyAdamW) = {{13, ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
                                       {14, ATTR_DESC(amsgrad, AnyTraits<std::vector<int64_t>>())},
                                       {15, ATTR_DESC(maximize, AnyTraits<std::vector<int64_t>>())},
                                       {16, ATTR_DESC(mask_zero, AnyTraits<std::vector<int64_t>>())},
                                       {17, ATTR_DESC(padding_key, AnyTraits<std::vector<int64_t>>())},
                                       {18, ATTR_DESC(padding_key_mask, AnyTraits<std::vector<int64_t>>())},
                                       {19, ATTR_DESC(completion_key, AnyTraits<std::vector<int64_t>>())},
                                       {20, ATTR_DESC(completion_key_mask, AnyTraits<std::vector<int64_t>>())},
                                       {21, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
                                       {22, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingApplyAdamW) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())}};
OUTPUT_MAP(EmbeddingApplyAdamW) = {{0, OUTPUT_DESC(var_handle)}};
REG_ADPT_DESC(EmbeddingApplyAdamW, ops::kNameEmbeddingApplyAdamW, ADPT_DESC(EmbeddingApplyAdamW))

// EmbeddingApplyAdaGrad
INPUT_MAP(EmbeddingApplyAdaGrad) = {
  {1, INPUT_DESC(var_handle)}, {2, INPUT_DESC(lr)},          {3, INPUT_DESC(grad)},
  {4, INPUT_DESC(keys)},       {5, INPUT_DESC(global_step)},
};
INPUT_ATTR_MAP(EmbeddingApplyAdaGrad) = {{6, ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
                                         {7, ATTR_DESC(mask_zero, AnyTraits<std::vector<int64_t>>())},
                                         {8, ATTR_DESC(padding_key, AnyTraits<std::vector<int64_t>>())},
                                         {9, ATTR_DESC(padding_key_mask, AnyTraits<std::vector<int64_t>>())},
                                         {10, ATTR_DESC(completion_key, AnyTraits<std::vector<int64_t>>())},
                                         {11, ATTR_DESC(completion_key_mask, AnyTraits<std::vector<int64_t>>())},
                                         {12, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
                                         {13, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingApplyAdaGrad) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())}};
OUTPUT_MAP(EmbeddingApplyAdaGrad) = {{0, OUTPUT_DESC(var_handle)}};
REG_ADPT_DESC(EmbeddingApplyAdaGrad, ops::kNameEmbeddingApplyAdaGrad, ADPT_DESC(EmbeddingApplyAdaGrad))

// EmbeddingApplySgd
INPUT_MAP(EmbeddingApplySgd) = {{1, INPUT_DESC(var_handle)},
                                {2, INPUT_DESC(lr)},
                                {3, INPUT_DESC(grad)},
                                {4, INPUT_DESC(keys)},
                                {5, INPUT_DESC(global_step)}};
INPUT_ATTR_MAP(EmbeddingApplySgd) = {{6, ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
                                     {7, ATTR_DESC(mask_zero, AnyTraits<std::vector<int64_t>>())},
                                     {8, ATTR_DESC(padding_key, AnyTraits<std::vector<int64_t>>())},
                                     {9, ATTR_DESC(padding_key_mask, AnyTraits<std::vector<int64_t>>())},
                                     {10, ATTR_DESC(completion_key, AnyTraits<std::vector<int64_t>>())},
                                     {11, ATTR_DESC(completion_key_mask, AnyTraits<std::vector<int64_t>>())},
                                     {12, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
                                     {13, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingApplySgd) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())}};
OUTPUT_MAP(EmbeddingApplySgd) = {{0, OUTPUT_DESC(var_handle)}};
REG_ADPT_DESC(EmbeddingApplySgd, ops::kNameEmbeddingApplySgd, ADPT_DESC(EmbeddingApplySgd))

// EmbeddingApplyRmsprop
INPUT_MAP(EmbeddingApplyRmsprop) = {{1, INPUT_DESC(var_handle)}, {2, INPUT_DESC(lr)},         {3, INPUT_DESC(rho)},
                                    {4, INPUT_DESC(momentum)},   {5, INPUT_DESC(epsilon)},    {6, INPUT_DESC(grad)},
                                    {7, INPUT_DESC(keys)},       {8, INPUT_DESC(global_step)}};
INPUT_ATTR_MAP(EmbeddingApplyRmsprop) = {{9, ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
                                         {10, ATTR_DESC(mask_zero, AnyTraits<std::vector<int64_t>>())},
                                         {11, ATTR_DESC(padding_key, AnyTraits<std::vector<int64_t>>())},
                                         {12, ATTR_DESC(padding_key_mask, AnyTraits<std::vector<int64_t>>())},
                                         {13, ATTR_DESC(completion_key, AnyTraits<std::vector<int64_t>>())},
                                         {14, ATTR_DESC(completion_key_mask, AnyTraits<std::vector<int64_t>>())},
                                         {15, ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
                                         {16, ATTR_DESC(_max_key_num, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingApplyRmsprop) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())}};
OUTPUT_MAP(EmbeddingApplyRmsprop) = {{0, OUTPUT_DESC(var_handle)}};
REG_ADPT_DESC(EmbeddingApplyRmsprop, ops::kNameEmbeddingApplyRmsprop, ADPT_DESC(EmbeddingApplyRmsprop))

// EmbeddingTableImport
INPUT_MAP(EmbeddingTableImport) = {{1, INPUT_DESC(file_path)}, {2, INPUT_DESC(ps_id)}, {3, INPUT_DESC(table_id)}};
ATTR_MAP(EmbeddingTableImport) = {
  {"embedding_dim", ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
  {"value_total_len", ATTR_DESC(value_total_len, AnyTraits<std::vector<int64_t>>())},
  {"only_var_flag", ATTR_DESC(only_var_flag, AnyTraits<bool>())},
  {"file_type", ATTR_DESC(file_type, AnyTraits<std::string>())},
  {"table_name", ATTR_DESC(table_name, AnyTraits<std::vector<std::string>>())},
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
};
OUTPUT_MAP(EmbeddingTableImport) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(EmbeddingTableImport, kNameEmbeddingTableImport, ADPT_DESC(EmbeddingTableImport))

// EmbeddingTableExport
INPUT_MAP(EmbeddingTableExport) = {
  {1, INPUT_DESC(file_path)}, {2, INPUT_DESC(ps_id)}, {3, INPUT_DESC(table_id)}, {4, INPUT_DESC(global_step)}};
ATTR_MAP(EmbeddingTableExport) = {
  {"embedding_dim", ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
  {"value_total_len", ATTR_DESC(value_total_len, AnyTraits<std::vector<int64_t>>())},
  {"export_mode", ATTR_DESC(export_mode, AnyTraits<std::string>())},
  {"only_var_flag", ATTR_DESC(only_var_flag, AnyTraits<bool>())},
  {"file_type", ATTR_DESC(file_type, AnyTraits<std::string>())},
  {"table_name", ATTR_DESC(table_name, AnyTraits<std::vector<std::string>>())},
  {"filter_export_flag", ATTR_DESC(filter_export_flag, AnyTraits<bool>())},
  {"steps_to_live_list", ATTR_DESC(steps_to_live_list, AnyTraits<std::vector<int64_t>>())},
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
};
OUTPUT_MAP(EmbeddingTableExport) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(EmbeddingTableExport, kNameEmbeddingTableExport, ADPT_DESC(EmbeddingTableExport))

// EmbeddingComputeVarExport
INPUT_MAP(EmbeddingComputeVarExport) = {
  {1, INPUT_DESC(file_path)},
  {2, INPUT_DESC(ps_id)},
  {3, INPUT_DESC(table_id)},
};
ATTR_MAP(EmbeddingComputeVarExport) = {
  {"table_name", ATTR_DESC(table_name, AnyTraits<std::vector<std::string>>())},
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
};
OUTPUT_MAP(EmbeddingComputeVarExport) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(EmbeddingComputeVarExport, kNameEmbeddingComputeVarExport, ADPT_DESC(EmbeddingComputeVarExport))

// EmbeddingComputeVarImport
INPUT_MAP(EmbeddingComputeVarImport) = {
  {1, INPUT_DESC(file_path)},
  {2, INPUT_DESC(ps_id)},
  {3, INPUT_DESC(table_id)},
};
ATTR_MAP(EmbeddingComputeVarImport) = {
  {"table_name", ATTR_DESC(table_name, AnyTraits<std::vector<std::string>>())},
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
};
OUTPUT_MAP(EmbeddingComputeVarImport) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(EmbeddingComputeVarImport, kNameEmbeddingComputeVarImport, ADPT_DESC(EmbeddingComputeVarImport))

// FakeRemoteLookupUniqued
INPUT_MAP(FakeRemoteLookupUniqued) = {{1, INPUT_DESC(table_id)},
                                      {2, INPUT_DESC(keys)},
                                      {3, INPUT_DESC(actual_keys_num)},
                                      {4, INPUT_DESC(unique_indices)},
                                      {5, INPUT_DESC(key_count)}};
ATTR_MAP(FakeRemoteLookupUniqued) = {
  {"embedding_dim", ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
  {"value_total_len", ATTR_DESC(value_total_len, AnyTraits<std::vector<int64_t>>())},
  {"initializer_mode", ATTR_DESC(initializer_mode, AnyTraits<std::vector<std::string>>())},
  {"constant_value", ATTR_DESC(constant_value, AnyTraits<std::vector<float>>())},
  {"min", ATTR_DESC(min, AnyTraits<std::vector<float>>())},
  {"max", ATTR_DESC(max, AnyTraits<std::vector<float>>())},
  {"mu", ATTR_DESC(mu, AnyTraits<std::vector<float>>())},
  {"sigma", ATTR_DESC(sigma, AnyTraits<std::vector<float>>())},
  {"seed", ATTR_DESC(seed, AnyTraits<std::vector<int64_t>>())},
  {"seed2", ATTR_DESC(seed2, AnyTraits<std::vector<int64_t>>())},
  {"filter_mode", ATTR_DESC(filter_mode, AnyTraits<std::vector<std::string>>())},
  {"filter_freq", ATTR_DESC(filter_freq, AnyTraits<std::vector<int64_t>>())},
  {"default_key_or_value", ATTR_DESC(default_key_or_value, AnyTraits<std::vector<int64_t>>())},
  {"default_key", ATTR_DESC(default_key, AnyTraits<std::vector<int64_t>>())},
  {"default_value", ATTR_DESC(default_value, AnyTraits<std::vector<float>>())},
  {"completion_key", ATTR_DESC(completion_key, AnyTraits<std::vector<int64_t>>())},
  {"completion_key_mask", ATTR_DESC(completion_key_mask, AnyTraits<std::vector<int64_t>>())},
  {"optimizer_mode", ATTR_DESC(optimizer_mode, AnyTraits<std::vector<std::string>>())},
  {"optimizer_params", ATTR_DESC(optimizer_params, AnyTraits<std::vector<float>>())},
  {"_embedding_dim", ATTR_DESC(_embedding_dim, AnyTraits<int64_t>())},
  {"_max_key_num", ATTR_DESC(_max_key_num, AnyTraits<int64_t>())},
  {"_use_counter_filter", ATTR_DESC(_use_counter_filter, AnyTraits<int64_t>())},
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())},
  {"_execute_times", ATTR_DESC(_execute_times, AnyTraits<int64_t>())}};
OUTPUT_MAP(FakeRemoteLookupUniqued) = {{0, OUTPUT_DESC(values)}};
REG_ADPT_DESC(FakeRemoteLookupUniqued, kNameFakeRemoteLookupUniqued, ADPT_DESC(FakeRemoteLookupUniqued))

// EmbeddingTableEvict
INPUT_MAP(EmbeddingTableEvict) = {{1, INPUT_DESC(var_handle)}, {2, INPUT_DESC(global_step)}};
INPUT_ATTR_MAP(EmbeddingTableEvict) = {{3, ATTR_DESC(steps_to_live, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingTableEvict) = {
  {"_process_node_engine_id", ATTR_DESC(_process_node_engine_id, AnyTraits<std::string>())}};
OUTPUT_MAP(EmbeddingTableEvict) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(EmbeddingTableEvict, ops::kNameEmbeddingTableEvict, ADPT_DESC(EmbeddingTableEvict))

// EmbeddingFeatureMappingV2
INPUT_MAP(EmbeddingFeatureMappingV2) = {{1, INPUT_DESC(table_name)}, {2, INPUT_DESC(feature_id)}};
INPUT_ATTR_MAP(EmbeddingFeatureMappingV2) = {{3, ATTR_DESC(table_total_size, AnyTraits<std::vector<int64_t>>())},
                                             {4, ATTR_DESC(table_actual_size, AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(EmbeddingFeatureMappingV2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(EmbeddingFeatureMappingV2) = {{0, OUTPUT_DESC(offset_id)}};
REG_ADPT_DESC(EmbeddingFeatureMappingV2, ops::kNameEmbeddingFeatureMappingV2, ADPT_DESC(EmbeddingFeatureMappingV2))

// EmbeddingFeatureMappingTableSize
INPUT_MAP(EmbeddingFeatureMappingTableSize) = {{1, INPUT_DESC(table_name)}};
ATTR_MAP(EmbeddingFeatureMappingTableSize) = EMPTY_ATTR_MAP;
OUTPUT_MAP(EmbeddingFeatureMappingTableSize) = {{0, OUTPUT_DESC(feature_size)}};
REG_ADPT_DESC(EmbeddingFeatureMappingTableSize, ops::kNameEmbeddingFeatureMappingTableSize,
              ADPT_DESC(EmbeddingFeatureMappingTableSize))

// EmbeddingFeatureMappingFind
INPUT_MAP(EmbeddingFeatureMappingFind) = {{1, INPUT_DESC(table_name)}, {2, INPUT_DESC(feature_size)}};
INPUT_ATTR_MAP(EmbeddingFeatureMappingFind) = {{3, ATTR_DESC(num, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingFeatureMappingFind) = EMPTY_ATTR_MAP;
DYN_OUTPUT_MAP(EmbeddingFeatureMappingFind) = {{0, DYN_OUTPUT_DESC(feature_id)}, {1, DYN_OUTPUT_DESC(offset_id)}};
REG_ADPT_DESC(EmbeddingFeatureMappingFind, ops::kNameEmbeddingFeatureMappingFind,
              ADPT_DESC(EmbeddingFeatureMappingFind))

// EmbeddingFeatureMappingExport
INPUT_MAP(EmbeddingFeatureMappingExport) = {
  {1, INPUT_DESC(file_path)}, {2, INPUT_DESC(table_name)}, {3, INPUT_DESC(values)}};
DYN_INPUT_MAP(EmbeddingFeatureMappingExport) = {{5, DYN_INPUT_DESC(feature_id)}, {6, DYN_INPUT_DESC(offset_id)}};
INPUT_ATTR_MAP(EmbeddingFeatureMappingExport) = {{4, ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(EmbeddingFeatureMappingExport) = EMPTY_ATTR_MAP;
OUTPUT_MAP(EmbeddingFeatureMappingExport) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(EmbeddingFeatureMappingExport, ops::kNameEmbeddingFeatureMappingExport,
              ADPT_DESC(EmbeddingFeatureMappingExport))

// EmbeddingFeatureMappingFileSize
INPUT_MAP(EmbeddingFeatureMappingFileSize) = {{1, INPUT_DESC(file_path)}, {2, INPUT_DESC(table_name)}};
INPUT_ATTR_MAP(EmbeddingFeatureMappingFileSize) = {{3, ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
                                                   {4, ATTR_DESC(only_offset_flag, AnyTraits<bool>())}};
ATTR_MAP(EmbeddingFeatureMappingFileSize) = EMPTY_ATTR_MAP;
OUTPUT_MAP(EmbeddingFeatureMappingFileSize) = {{0, OUTPUT_DESC(feature_size)}};
REG_ADPT_DESC(EmbeddingFeatureMappingFileSize, ops::kNameEmbeddingFeatureMappingFileSize,
              ADPT_DESC(EmbeddingFeatureMappingFileSize))

// EmbeddingFeatureMappingImport
INPUT_MAP(EmbeddingFeatureMappingImport) = {
  {1, INPUT_DESC(file_path)}, {2, INPUT_DESC(table_name)}, {3, INPUT_DESC(feature_size)}};
INPUT_ATTR_MAP(EmbeddingFeatureMappingImport) = {{4, ATTR_DESC(embedding_dim, AnyTraits<std::vector<int64_t>>())},
                                                 {5, ATTR_DESC(only_offset_flag, AnyTraits<bool>())},
                                                 {6, ATTR_DESC(num, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingFeatureMappingImport) = EMPTY_ATTR_MAP;
DYN_OUTPUT_MAP(EmbeddingFeatureMappingImport) = {
  {0, DYN_OUTPUT_DESC(feature_id)}, {1, DYN_OUTPUT_DESC(offset_id)}, {2, DYN_OUTPUT_DESC(values)}};
REG_ADPT_DESC(EmbeddingFeatureMappingImport, ops::kNameEmbeddingFeatureMappingImport,
              ADPT_DESC(EmbeddingFeatureMappingImport))

// EmbeddingFeatureMappingInsert
INPUT_MAP(EmbeddingFeatureMappingInsert) = {{1, INPUT_DESC(table_name)}};
INPUT_ATTR_MAP(EmbeddingFeatureMappingInsert) = {{2, ATTR_DESC(num, AnyTraits<int64_t>())}};
ATTR_MAP(EmbeddingFeatureMappingInsert) = EMPTY_ATTR_MAP;
DYN_INPUT_MAP(EmbeddingFeatureMappingInsert) = {{3, DYN_INPUT_DESC(feature_id)}, {4, DYN_INPUT_DESC(offset_id)}};
OUTPUT_MAP(EmbeddingFeatureMappingInsert) = EMPTY_OUTPUT_MAP;
REG_ADPT_DESC(EmbeddingFeatureMappingInsert, ops::kNameEmbeddingFeatureMappingInsert,
              ADPT_DESC(EmbeddingFeatureMappingInsert))

}  // namespace mindspore::transform
