import unittest
from scheduler.lm_block import HuggingfaceGTP2Block
import torch

class TestScheduler(unittest.TestCase):

    def test_lm_block(self):
        model_name = "gpt2"
        lm_block = HuggingfaceGTP2Block([model_name], {})

        input0 = [torch.tensor([[40, 2883, 6155, 351, 616, 13779, 3290]]),
                  torch.arange(7)[None, :],
                  torch.ones(7, dtype=torch.int64)[None, :]]

        output0 = lm_block.forward(input0, None)

        # input with kv_cache
        past_key_values = output0[1]
        input_ids = torch.tensor([[404]])
        past_seq = past_key_values[0][0].shape[-2]
        position_ids = torch.tensor([[past_seq]])
        attention_mask = torch.ones(past_seq + 1, dtype=torch.int64)
        output1 = lm_block.forward([input_ids, position_ids, attention_mask], past_key_values)


if __name__ == '__main__':
    unittest.main()
