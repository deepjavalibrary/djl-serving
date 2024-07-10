import unittest

from djl_python.chat_completions.chat_properties import ChatProperties, Message, TextInput, ImageInput
from djl_python.tests.utils import parameterized, parameters

chat_messages_properties = {
    "messages": [{
        "role": "system",
        "content": "You are a friendly chatbot"
    }, {
        "role": "user",
        "content": "Which is bigger, the moon or the sun?"
    }, {
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": "What is this image of?"
        }, {
            "type": "image_url",
            "image_url": {
                "url": "https://resources.djl.ai/images/dog_bike_car.jpg"
            }
        }]
    }]
}

expected_chat_conversion = [
    Message(role="system", content="You are a friendly chatbot"),
    Message(role="user", content="Which is bigger, the moon or the sun?"),
    Message(role="user",
            content=[{
                "type": "text",
                "text": "What is this image of?"
            }, {
                "type": "image_url",
                "image_url": {
                    "url": "https://resources.djl.ai/images/dog_bike_car.jpg"
                }
            }])
]


@parameterized
class TestChatCompletions(unittest.TestCase):

    def test_chat_configs(self):

        def test_chat_min_configs():
            chat_configs = ChatProperties(**chat_messages_properties)
            for message, expected_message in zip(chat_configs.messages,
                                                 expected_chat_conversion):
                self.assertEqual(message.get_tokenizer_inputs(),
                                 expected_message.get_tokenizer_inputs())
                self.assertEqual(message.get_images(),
                                 expected_message.get_images())
            self.assertIsNone(chat_configs.model)
            self.assertEqual(chat_configs.frequency_penalty, 0.0)
            self.assertIsNone(chat_configs.logit_bias)
            self.assertFalse(chat_configs.logprobs)
            self.assertIsNone(chat_configs.top_logprobs)
            self.assertIsNone(chat_configs.max_tokens)
            self.assertEqual(chat_configs.n, 1)
            self.assertEqual(chat_configs.presence_penalty, 0.0)
            self.assertIsNone(chat_configs.seed)
            self.assertIsNone(chat_configs.stop)
            self.assertFalse(chat_configs.stream)
            self.assertEqual(chat_configs.temperature, 1.0)
            self.assertEqual(chat_configs.top_p, 1.0)
            self.assertIsNone(chat_configs.user)

        def test_chat_all_configs():
            properties = dict(chat_messages_properties)
            properties["model"] = "model"
            properties["frequency_penalty"] = "1.0"
            properties["logit_bias"] = {"2435": -100.0, "640": -100.0}
            properties["logprobs"] = "false"
            properties["top_logprobs"] = "3"
            properties["max_tokens"] = "256"
            properties["n"] = "1"
            properties["presence_penalty"] = "1.0"
            properties["seed"] = "123"
            properties["stop"] = ["stop"]
            properties["stream"] = "true"
            properties["temperature"] = "1.0"
            properties["top_p"] = "3.0"
            properties["user"] = "user"

            chat_configs = ChatProperties(**properties)
            for message, expected_message in zip(chat_configs.messages,
                                                 expected_chat_conversion):
                self.assertEqual(message.get_tokenizer_inputs(),
                                 expected_message.get_tokenizer_inputs())
                self.assertEqual(message.get_images(),
                                 expected_message.get_images())
            self.assertEqual(chat_configs.model, properties['model'])
            self.assertEqual(chat_configs.frequency_penalty,
                             float(properties['frequency_penalty']))
            self.assertEqual(chat_configs.logit_bias, properties['logit_bias'])
            self.assertFalse(chat_configs.logprobs)
            self.assertIsNone(chat_configs.top_logprobs)
            self.assertEqual(chat_configs.max_tokens,
                             int(properties['max_tokens']))
            self.assertEqual(chat_configs.n, int(properties['n']))
            self.assertEqual(chat_configs.presence_penalty,
                             float(properties['presence_penalty']))
            self.assertEqual(chat_configs.seed, int(properties['seed']))
            self.assertEqual(chat_configs.stop, properties['stop'])
            self.assertTrue(chat_configs.stream)
            self.assertEqual(chat_configs.temperature,
                             float(properties['temperature']))
            self.assertEqual(chat_configs.top_p, float(properties['top_p']))
            self.assertEqual(chat_configs.user, properties['user'])

        test_chat_min_configs()
        test_chat_all_configs()

    @parameters([{
        "messages": [{
            "role1": "system",
            "content": "You are a friendly chatbot"
        }]
    }, {
        "frequency_penalty": "-3.0"
    }, {
        "frequency_penalty": "3.0"
    }, {
        "logit_bias": {
            "2435": -100.0,
            "640": 200.0
        }
    }, {
        "logit_bias": {
            "2435": -200.0,
            "640": 100.0
        }
    }, {
        "logprobs": "true",
        "top_logprobs": "-1"
    }, {
        "logprobs": "true",
        "top_logprobs": "30"
    }, {
        "presence_penalty": "-3.0"
    }, {
        "presence_penalty": "3.0"
    }, {
        "temperature": "-1.0"
    }, {
        "temperature": "3.0"
    }])
    def test_chat_invalid_configs(self, params):
        test_properties = {**chat_messages_properties, **params}
        with self.assertRaises(ValueError):
            ChatProperties(**test_properties)
