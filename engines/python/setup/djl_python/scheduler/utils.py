import torch


@staticmethod
def merge_tensors(tensor1: torch.Tensor,
                  tensor2: torch.Tensor,
                  seq_delta,
                  seq_order,
                  is_pad_token=False) -> torch.Tensor:
    if seq_delta == 0:
        return torch.cat([tensor1, tensor2], dim=0)

    shape1 = tensor1.shape()
    shape2 = tensor2.shape()

    delta_shape = [0, 0]
    delta_shape[0] = shape2[0]
    delta_shape[1] = shape1[1]

    if is_pad_token:
        delta_tensor = torch.full(size=delta_shape,
                                  fill_value=PAD_TOKEN_ID,
                                  dtype=tensor1.dtype)
    else:
        delta_tensor = torch.zeros(delta_shape[0],
                                   delta_shape[1],
                                   dtype=tensor1.dtype)

    # augment the batch 1
    tensor1 = torch.cat([tensor1, delta_tensor], dim=0)

    if seq_order == 1:
        tensor1[shape1[0]:, seq_order:, ...] = tensor2
    elif seq_order == 2:
        tensor1[shape1[0]:, seq_order:, ...] = tensor2
    return tensor1
