import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import boto3
from sagemaker import Session
from sagemaker import image_uris
from sagemaker.benchmarking.benchmarker import Benchmarker
from sagemaker.benchmarking.config_grid import ConfigGrid
from sagemaker.benchmarking.preset.traffic_pattern import STREAMING_INVOCATION_TYPE
from sagemaker.benchmarking.preset.traffic_pattern import PresetTrafficPattern
from sagemaker.benchmarking.preset.traffic_pattern import TrafficPatternConfig
from sagemaker.inference_recommender.inference_recommender_mixin import Concurrency
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.model import Model
from sagemaker.utils import name_from_base

#import logging
#logging.basicConfig(level=logging.DEBUG)

@dataclass
class BenchmarkConfiguration:
    """Data class to store an inference recommender benchmark configuration."""

    tokenizer_model_id: str
    model_args: Dict[str, Any]
    benchmark_configurations: List[Dict[str, Any]]
    image_uri_args: Dict[str, str] | None = None
    image_uri: str | None = None
    use_jumpstart_prod_artifact: bool = False
    jumpstart_model_id: str | None = None

    def get_image_uri(self, session: Session) -> str:
        """Return either the provided image URI or retrieve an image URI from the SageMaker Python SDK."""
        if self.image_uri:
            return self.image_uri
        elif self.image_uri_args:
            return image_uris.retrieve(
                region=session._region_name,
                sagemaker_session=session,
                **self.image_uri_args,
            )
        else:
            raise ValueError("Either 'image_uri' or 'image_uri_args' key must be provided in configuration file.")


def get_benchmarking_session(region_name: str) -> Session:
    """Create a Session for inference recommender benchmarking job."""
    boto_session = boto3.Session(region_name=region_name)
    sagemaker = boto3.client(service_name="sagemaker", region_name=region_name)
    sagemaker_runtime = boto3.client(service_name="sagemaker-runtime", region_name=region_name)
    return Session(boto_session=boto_session, sagemaker_client=sagemaker, sagemaker_runtime_client=sagemaker_runtime)


def run_benchmark_job(configuration_file, payload_url, metrics_dir, region="us-west-2", hf_token=None) -> None:    
    """Serially execute a benchmark job for each provided configuration file."""

    session = get_benchmarking_session(region)
    benchmarker = Benchmarker(role_arn=session.get_caller_identity_arn(), sagemaker_session=session)

        
    with open(configuration_file, "r") as f:
        configuration_dict = json.load(f)

    if hf_token is not None:
        configuration_dict = json.loads(json.dumps(configuration_dict).replace("{{hub_token}}", hf_token))
    configuration = BenchmarkConfiguration(**configuration_dict)

    model = Model(
        image_uri=configuration.get_image_uri(session),
        role=session.get_caller_identity_arn(),
        sagemaker_session=session,
        **configuration.model_args,
    )
    if configuration.use_jumpstart_prod_artifact:
        js_model = JumpStartModel(model_id=configuration.jumpstart_model_id, sagemaker_session=session)
        model.model_data = js_model.model_data
    model.env["OPTION_ENABLE_STREAMING"] = "true"
    model.env["OPTION_TGI_COMPAT"] = "true"  # required workaround for Rubikon V

    benchmark_job = benchmarker.create_benchmark_job(
        job_name=name_from_base(f"ir-js-{configuration_file.stem.replace('_', '-')}"),
        model=model,
        sample_payload_url=payload_url,
        content_type="application/json",
        benchmark_configurations=[ConfigGrid(**x) for x in configuration.benchmark_configurations],
        traffic_pattern=TrafficPatternConfig(
            #traffic_pattern=PresetTrafficPattern.LOGISTIC_GROWTH,
            traffic_pattern=[
                Concurrency(duration_in_seconds=600, concurrent_users=1),
                Concurrency(duration_in_seconds=600, concurrent_users=2),
                Concurrency(duration_in_seconds=300, concurrent_users=4),       
                Concurrency(duration_in_seconds=300, concurrent_users=8),       
                Concurrency(duration_in_seconds=300, concurrent_users=16), 
                Concurrency(duration_in_seconds=300, concurrent_users=32),
                Concurrency(duration_in_seconds=300, concurrent_users=64), 
            ],
            inference_invocation_types=STREAMING_INVOCATION_TYPE,
        ),
        stopping_conditions={
            "FlatInvocations": "Continue",
            "MaxInvocations": 600000,
            "MaxModelLatencyInMs": 50000,
        },
        tokenizer_config={"ModelId": configuration.tokenizer_model_id, "AcceptEula": True},
        job_duration=10800,
    )
    time.sleep(5)
    benchmark_job.wait()

    local_file = Path(benchmark_job.get_benchmark_results_csv_path(download=True, local_path=metrics_dir))
    local_file.rename(local_file.with_stem(configuration_file.stem))


