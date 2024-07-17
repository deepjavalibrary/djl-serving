# Dataset for DJLServing/LMI

## Add benchmark dataset

You can modify the [dataset_prep.py](dataset_prep.py) script to add more dataset for serving benchmarks.

## MMLU standard dataset

Step 1: run mmlu_generator.py to generate the mmlu standard dataset with sampling.

Step 2: run dataset_prep.py

The answer is part of request payload in the `answer` field.

The way to evaluate correctness is to

```python
def evaluate_correctness(result_str: str, answer: str) -> bool:
    return result_str.strip().startswith(answer)
```

Simple choice matching. Then do a percentage of correct answer as correctness.
