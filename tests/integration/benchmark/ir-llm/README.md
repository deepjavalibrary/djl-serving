# IR-LLM Benchmark on SageMaker Endpoint
This folder contains scripts and configurations to run benchmarks on SageMaker Endpoints using the IR-LLM.


## Usage
Run the benchmark script:
```
cd tests/integration/benchmark/ir-llm
python cw_metrics.py -j config.yml -c ./configs -m ./metrics
```
This command runs the benchmark script cw_metrics.py with the following arguments:

-j config.yml: Specifies the main configuration file config.yml.
-c ./configs: Specifies the directory containing the IR-LLM configuration files for each model's test case.
-m ./metrics: Specifies the directory where the benchmark reports will be saved.

## Configuration
### config.yml
The config.yml file defines the overall benchmark configuration, including:

* cloudwatch_metrics_namespace: The CloudWatch namespace for the metrics.
* metrics_definitions: A list of metric definitions to be collected during the benchmark.
* benchmark_report_s3_location: The S3 location where the benchmark reports will be stored.
* model_test_cases: A list of model test cases to be benchmarked.


### benchmark_config_xxx.json
The xxx.json files in the configs directory define the IR-LLM configuration for each model's test case. 


## Benchmark Reports
After running the benchmark, the reports will be saved in the specified S3 location. The reports will contain detailed metrics and performance data for each benchmarked model test case.