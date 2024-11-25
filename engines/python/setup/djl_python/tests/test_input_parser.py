import unittest
from djl_python import Input
from djl_python.input_parser import parse_adapters
from djl_python.properties_manager.hf_properties import HuggingFaceProperties
from djl_python.request_io import TextInput

adapter_properties = {
    "name": "adapter1",
    "src": "src",
}


class TestInputParser(unittest.TestCase):

    def test_parse_adapters_from_input_map(self):
        request_input = TextInput()
        input_item = Input()
        input_map = {
            "inputs": "inputs1",
            "adapters": adapter_properties["name"]
        }

        adapter = Input()
        adapter.properties = adapter_properties
        kwargs = {
            "configs": HuggingFaceProperties(model_id="model",
                                             enable_lora=True),
            "adapter_registry": {
                adapter_properties["name"]: adapter
            }
        }
        parse_adapters(request_input, input_item, input_map, **kwargs)
        self.assertIsNotNone(request_input.adapters)
        self.assertEqual(adapter_properties["name"],
                         request_input.adapters.get_property("name"))
        self.assertEqual(adapter_properties["src"],
                         request_input.adapters.get_property("src"))

    def test_parse_adapters_from_content(self):
        request_input = TextInput()
        input_item = Input()
        input_item.content.add(
            key="adapter", value=adapter_properties["name"].encode("utf-8"))

        adapter = Input()
        adapter.properties = adapter_properties
        kwargs = {
            "configs": HuggingFaceProperties(model_id="model",
                                             enable_lora=True),
            "adapter_registry": {
                adapter_properties["name"]: adapter
            }
        }
        parse_adapters(request_input, input_item, {}, **kwargs)
        self.assertIsNotNone(request_input.adapters)
        self.assertEqual(adapter_properties["name"],
                         request_input.adapters.get_property("name"))
        self.assertEqual(adapter_properties["src"],
                         request_input.adapters.get_property("src"))

    def test_parse_adapters_from_properties(self):
        request_input = TextInput()
        input_item = Input()
        input_item.properties[
            "X-Amzn-SageMaker-Adapter-Identifier"] = adapter_properties["name"]
        input_item.properties["X-Amzn-SageMaker-Adapter-Alias"] = "a1"

        adapter = Input()
        adapter.properties = adapter_properties
        kwargs = {
            "configs": HuggingFaceProperties(model_id="model",
                                             enable_lora=True),
            "adapter_registry": {
                adapter_properties["name"]: adapter
            }
        }
        parse_adapters(request_input, input_item, {}, **kwargs)
        self.assertIsNotNone(request_input.adapters)
        self.assertEqual(adapter_properties["name"],
                         request_input.adapters.get_property("name"))
        self.assertEqual(adapter_properties["src"],
                         request_input.adapters.get_property("src"))

    def test_parse_adapters_override(self):
        request_input = TextInput()
        input_item = Input()
        input_item.properties[
            "X-Amzn-SageMaker-Adapter-Identifier"] = adapter_properties["name"]
        input_item.properties["X-Amzn-SageMaker-Adapter-Alias"] = "a1"
        input_map = {"inputs": "inputs1", "adapters": "adapter2"}

        adapter = Input()
        adapter.properties = adapter_properties
        kwargs = {
            "configs": HuggingFaceProperties(model_id="model",
                                             enable_lora=True),
            "adapter_registry": {
                adapter_properties["name"]: adapter
            }
        }
        parse_adapters(request_input, input_item, input_map, **kwargs)
        self.assertIsNotNone(request_input.adapters)
        self.assertEqual(adapter_properties["name"],
                         request_input.adapters.get_property("name"))
        self.assertEqual(adapter_properties["src"],
                         request_input.adapters.get_property("src"))

    def test_parse_adapters_not_exist(self):
        request_input = TextInput()
        input_item = Input()
        input_map = {"inputs": "inputs1", "adapters": "non_exist"}

        adapter = Input()
        adapter.properties = adapter_properties
        kwargs = {
            "configs": HuggingFaceProperties(model_id="model",
                                             enable_lora=True),
            "adapter_registry": {}
        }
        with self.assertRaises(ValueError):
            parse_adapters(request_input, input_item, input_map, **kwargs)


if __name__ == '__main__':
    unittest.main()
