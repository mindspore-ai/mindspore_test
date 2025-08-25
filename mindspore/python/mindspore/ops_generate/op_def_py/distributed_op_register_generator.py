# Copyright 2025 Huawei Technologies Co., Ltd
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
Generate distributed operator register python files.
"""
from collections import defaultdict
from common.template import Template
import common.gen_utils as gen_utils


class DistributedOpGenerator:
    """Generates distributed operator registration code from YAML files"""

    def __init__(self):
        """Initialize the generator with appropriate templates"""
        self.file_template = Template("""
# Copyright 2025 Huawei Technologies Co., Ltd
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

${import_statements}

${operator_definitions}
""")

        self.import_template = Template("from mindspore.parallel.spmd.ops.${module} import ${classes}")
        self.operator_template = Template("${dist_op_name} = ${distributed_op_class}('${operator_name}')")

    def generate_import_statements(self, ops_data):
        """Generate import statements based on distributed_op_file from YAML data"""
        # Group classes by their module
        module_classes = defaultdict(list)
        for op_info in ops_data.values():
            module = op_info['distributed_op_file']
            class_name = op_info['distributed_op_class']
            module_classes[module].append(class_name)

        # Generate import statements for each module
        import_statements = []
        for module, classes in module_classes.items():
            # Remove duplicates while preserving order
            unique_classes = []
            seen = set()
            for cls in classes:
                if cls not in seen:
                    seen.add(cls)
                    unique_classes.append(cls)

            import_stmt = self.import_template.replace(
                module=module,
                classes=", ".join(unique_classes)
            )
            import_statements.append(import_stmt)

        return import_statements

    def generate_operator_definitions(self, ops_data):
        """Generate operator definition lines from operator data"""
        definitions = []
        for op_name, op_info in ops_data.items():
            op_definition = self.operator_template.replace(
                dist_op_name=op_info['dist_op_name'],
                distributed_op_class=op_info['distributed_op_class'],
                operator_name=op_name
            )
            definitions.append(op_definition)
        return definitions

    def generate(self, yaml_dir, work_dir, file_name="distributed_op_init.py"):
        """
        Generate the distributed operator registration code

        Args:
            yaml_dir: Directory containing YAML files with operator definitions
            work_dir: Directory to save the generated code
            file_name: Name of the generated Python file
        """
        ops_data = gen_utils.safe_load_yaml_from_dir(yaml_dir)

        import_statements = self.generate_import_statements(ops_data)
        operator_definitions = self.generate_operator_definitions(ops_data)

        full_code = self.file_template.replace(
            import_statements="\n".join(import_statements),
            operator_definitions="\n".join(operator_definitions)
        )

        gen_utils.save_file(work_dir, file_name, full_code)
