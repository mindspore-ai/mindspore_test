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
"""LlmBoostRegister"""
import inspect


class LlmBoostType:
    """Class module type for vision pretrain"""

    def __init__(self):
        pass

    BUILDIN = 'BuildIn'


class LlmBoostRegister:
    """
    Module class factory.
    """

    def __init__(self):
        pass

    registry = {}

    @classmethod
    def register(cls, boost_type=LlmBoostType.BUILDIN, alias=None):
        """Register class into registry
        Args:
            boost_type:
                boost type name, default LlmBoostType.BUILDIN
            alias (str) : model_name

        Returns:
            wrapper
        """

        def wrapper(register_class):
            """Register-Class with wrapper function.

            Args:
                register_class : class need to register

            Returns:
                wrapper of register_class
            """
            model_name = alias if alias is not None else register_class.__name__
            if boost_type not in cls.registry:
                cls.registry[boost_type] = {model_name: register_class}
            else:
                cls.registry[boost_type][model_name] = register_class
            return register_class

        return wrapper

    @classmethod
    def is_exist(cls, boost_type, model_name=None):
        """Determine whether class name is in the current type group.

        Args:
            boost_type : Module type
            model_name : model name

        Returns:
            True/False
        """
        if not model_name:
            return boost_type in cls.registry
        registered = boost_type in cls.registry and model_name in cls.registry.get(
            boost_type)
        return registered

    @classmethod
    def get_cls(cls, boost_type, model_name=None):
        """Get class

        Args:
            boost_type : Module type
            model_name : model name

        Returns:
            register_class
        """
        if not cls.is_exist(boost_type, model_name):
            raise ValueError("Can't find class type {} class name {} \
            in class registry".format(boost_type, model_name))

        if not model_name:
            raise ValueError(
                "Can't find model. model name = {}".format(model_name))
        register_class = cls.registry.get(boost_type).get(model_name)
        return register_class

    @classmethod
    def get_instance(cls, boost_type=LlmBoostType.BUILDIN, model_name=None, **kwargs):
        """Get instance.
        Args:
            boost_type : module type
            model_name : model type
        Returns:
            object : The constructed object
        """
        if model_name is None:
            raise ValueError("Class name cannot be None.")

        if isinstance(model_name, str):
            obj_cls = cls.get_cls(boost_type, model_name)
        elif inspect.isclass(model_name):
            obj_cls = model_name
        else:
            raise ValueError("Can't find boost type {} model name {} \
            in class registry.".format(boost_type, model_name))

        try:
            return obj_cls(**kwargs)
        except Exception as e:
            raise type(e)('{}: {}'.format(obj_cls.__name__, e))
