#  Seq-Scheduler and Max-Sparsity Thresholding

Seq-scheduler is a language model serving system built on native Huggingface inference backend, and works with any 
huggingface model out-of-box. It features dynamic batching, dynamic batch splitting and prompt caching, with the 
support of various autoregressive search methods, including greedy, sampling and [contrastive search](https://huggingface.co/blog/introducing-csearch). In the 
following part of the document, the way to run these features will be introduced.

## Start a model server with a serving.properties file
The model server is configured by a stand-along serving.properties file. Here `/workspace/llama-2-70b-hf/` model is 
used as an example. 
```
# serving.properties

engine=MPI
option.model_id=/workspace/llama-2-70b-hf/
option.tensor_parallel_degree=1
option.dtype=fp16
option.trust_remote_code=true

# rolling-batch parameters
option.max_rolling_batch_size=64
option.rolling_batch=scheduler

# seq-scheduler parameters
option.max_sparsity=0.33
option.max_splits=3
option.decoding_strategy=contrastive  # other options: greedy, sample
```

Djl-serving typically is started with the following docker command. For comprehensive ways starting djl-serving, 
please take look at the djl-serving introduction tutorial.
```bash
docker run -it --name zen_dirac --runtime=nvidia --gpus all --shm-size 3g \
-v /home/ubuntu/model:/opt/ml/model -v /tmp:/tmp -v /home/ubuntu/mount_folder:/opt/mount_folder \
-v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
-p 8080:8080 deepjavalibrary/djl-serving:deepspeed-nightly
```

## Send request to the server with a curl command
Next, by running a curl command, a request can be sent to the model server for text generation. A generic request 
looks like the following.
```bash
export CONCURRENCY=64
export REPETITION=3
TOKENIZER=/workspace/llama-2-70b-hf/ ./awscurl -c ${CONCURRENCY} -N ${REPETITION} -X POST http://127.0.0.
1:8080/invocations \
  --connect-timeout 60 \
  -H "Content-type: application/json" \
  -d '{"inputs":"The new movie that got Oscar this year","parameters":{"max_new_tokens":50, "do_sample":true},
  "cached_prompt":"This is a prompt that will be cached and appended to the front of the inputs strings."}' \
  -t \
  -o output-test.txt
```

## The dynamic batch splitting
In real application of large language model (LLM), input prompt text’s lengths are not only different, but also can 
vary a lot, ranging over several orders of magnitude. To deal with these inhomogeneous-length sequences, the most 
general way is padding. 

Based on the padding solution, in this section, we introduce the dynamic batch splitting feature, which optimally 
split the batch kv_cache into sub-batches, to avoid the usage of too much padding, and run inference on each part 
separately. The optimal partition is computed efficiently by a dynamic optimization process.
This allows serving largely-variant-length prompts.

### MaxSparse parameters
* Feature: in the padding solution of various-length input, for a batch input of length [300, 50], the padding for the sequence will be [0, 250], which causes sparsity of 0.7, a huge waste of memory. The max-sparsity-thresholding effectively solves this problem.
* Configuration:
    * `max_sparsity`: 0 - 1, default 0.33
    * `max_splits`: >= 1, default 3. 


* The max_splits limit the max number of sequential inference call in one step of generation, which is used to guarantee the latency is not too much
* The max_sparsityis used to control the sparsity caused by padding and number of splits. 
    * When max_sparsity_threshold is small, then the high sparsity is not tolerated, then the number of splits will increase. The space is saved, but the latency may increase due to the increase in the number of inference call.
    * When max_sparsity_threshold is large, then the high sparsity is tolerable, then the number of splits will decrease. Then more padding is needed, the latency will decrease.
* They are set by passing the values through serving.properties as shown the in section above.

### Demonstration

This demonstration shows that, for the same sequence length array [38, 33, 24, 22], how it is dynamically partitioned according to different max_sparsity threshold. It can be seen that the partition is the optimal one in the sense that after the partition, the number of padding is maximally reduced, below certain max_sparsity threshold, which saves as much memory as possible. 

```bash
seq_lengths: [38, 33, 24, 22]
max_sparsity: 0.01
num_splits: 4
partition: [[1], [2], [3], [4]]


seq_lengths: [38, 33, 24, 22]
max_sparsity: 0.025
num_splits: 3
partition: [[1], [2], [3, 4]]

seq_lengths: [38, 33, 24, 22]
max_sparsity: 0.1
num_splits: 2
partition: [[1, 2], [3, 4]]

seq_lengths: [38, 33, 24, 22]
max_sparsity: 0.9
num_splits: 1
partition: [[1, 2, 3, 4]]
```
