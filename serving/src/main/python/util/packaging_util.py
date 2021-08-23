import importlib
import inspect
import sys


def list_processor_classes(module, parent_class=None):
    """
    Parse user defined module to get all model service classes in it.

    :param module:
    :param parent_class:
    :return: List of model service class definitions
    """

    # Parsing the module to get all defined classes
    classes = [cls[1] for cls in inspect.getmembers(module, lambda member: inspect.isclass(member) and
                                                                           member.__module__ == module.__name__)]
    # filter classes that is subclass of parent_class
    if parent_class is not None:
        return [c for c in classes if issubclass(c, parent_class)]

    return classes


def get_class_name(file_name, function_name):
    if not file_name.endswith(".py"):
        raise ValueError("It has to be a python file")

    module = file_name[:-3]
    module = module.split("/")[-1]
    sys.path.append(file_name[:-(len(module) + 3)])
    module = importlib.import_module(module)

    if module is None:
        raise ValueError("Unable to load module {}, make sure it is added to python path".format(module))

    if hasattr(module, function_name):
        return module
    else:
        processor_class_definitions = list_processor_classes(module)
        if len(processor_class_definitions) != 1:
            raise ValueError("Expected only one class in python processor or a function entry point {}".format(
                processor_class_definitions))

        processor_class = processor_class_definitions[0]
        return processor_class
