There are 3 steps to use this benchmark tool:

* Install llmperf

Follow installation instructions at: https://github.com/ray-project/llmperf?tab=readme-ov-file#installation

* Configure YAML file in foler `config`
Configure the following parameters:
- Metrics settings
- S3 log location
- Benchmark parameters including:
1. Docker image
2. Docker settings
3. Server settings
4. LLMperf parameters

* Run the benchmark script `script/run_benchmark.py`
Execute the following command:
```
python run_benchmark.py -j config_file_path -p llmperf_folder_path
```


