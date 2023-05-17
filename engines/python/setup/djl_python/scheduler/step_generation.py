import torch
from torch.nn.functional import normalize, softmax


def contrast_step_generate(top_k_ids: torch.Tensor,
                           logits: torch.Tensor,
                           context_hidden_states: torch.Tensor,
                           top_k_hidden_states: torch.Tensor,
                           offsets: torch.Tensor,
                           alpha: float):
    # topKIds: [batch, topK]
    # attentionMask: [batch, past_seq]
    # logits:  [batch, vocabSize]
    # contextHiddenStates: [batch, past_seq, dim]
    # topkHiddenStates: [batch*topK, seq=1, dim]
    # attentionMaskSlice: [batch, 2]: (startPosition, endPosition)

    batch_len = top_k_ids.shape[0]
    top_k_len = top_k_ids.shape[1]
    hidden_dim = top_k_hidden_states.shape[-1]

    # [batch*topK, seq=1, dim] -> [batch, topK, dim]
    top_k_hidden_states = top_k_hidden_states.reshape(batch_len, top_k_len, hidden_dim)

    # [batch, topK, dim] * [batch, past_seq, dim] -> [batch, topK, past_seq]
    top_k_hidden_states = normalize(top_k_hidden_states, p=2, dim=2)
    context_hidden_states = normalize(context_hidden_states, p=2, dim=2)
    cos_similarity = torch.bmm(top_k_hidden_states, context_hidden_states.permute(0, 2, 1))

    offsets_list = offsets.tolist()
    for i in range(len(offsets_list)):
        cos_similarity[i][:][0:offsets_list[i]] = -1

    # [batch, topK, past_seq] -> [batch, topK]
    top_k_score_part1, _ = torch.max(cos_similarity, dim=2)
    assert len(top_k_score_part1.shape) == 2
    # [batch, logitDim].gather([batch, topK) -> [batch, topK]
    top_k_score_part2 = torch.gather(softmax(logits, dim=1), dim=1, index=top_k_ids)

    top_k_score = torch.subtract(torch.mul(top_k_score_part2, 1 - alpha), torch.mul(top_k_score_part1, alpha))

    # [batch, topK] => [batch, 1]
    select = torch.argmax(top_k_score, dim=1)
    output_ids = top_k_ids[0:top_k_ids.shape[0], select.tolist(), ...].reshape(-1, 1)
    return [output_ids, select]


def greedy_step_generate(logits: torch.Tensor):
    assert logits.shape == 3
    logits = logits[:, -1, :]
    return torch.unsqueeze(torch.argmax(logits, dim=-1), dim=1)


def beam_step_generate(last_probs: torch.Tensor,
                       logits: torch.Tensor,
                       batch_len: int,
                       beam_len: int):
    all_probs = torch.softmax(logits[:, -1, :], dim=1).reshape(batch_len, beam_len, -1)
    top_k = torch.topk(all_probs, k=beam_len, dim=-1, largest=True, sorted=False)
    output_ids = top_k[1]
    step_probs = top_k[0]

    # Chain the probability
    # [batch, beamSource] -> [batch, beamSource, 1]
    last_probs = last_probs.reshape(batch_len, beam_len, 1)
    # [batch, beamSource, beamChild]
    new_probs = torch.mul(step_probs, last_probs)

    topK = torch.topk(new_probs.reshape(batch_len, beam_len * beam_len),
                      k=beam_len,
                      dim=-1,
                      largest=True,
                      sorted=False)

    # The select indices act on (beamSource, beamChild) dimension. Decides how the new
    # generated tokenIds correspond to the past tokenIds.
    # [batch, beamNew].
    select = topK[1]
