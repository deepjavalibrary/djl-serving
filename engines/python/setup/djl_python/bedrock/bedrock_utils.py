from typing import Optional


def is_bedrock_request(bedrock_invoke_type: Optional[str]) -> bool:
    # TODO, not sure if this is reliable
    # We might want to just use an env var since a container running in bedrock will not run any other way
    return bedrock_invoke_type == "InvokeEndpoint" or bedrock_invoke_type == "InvokeEndpointWithResponseStream"


def parse_bedrock_request(input_map: dict, is_rolling_batch: bool, tokenizer, invoke_type: str):
    _inputs = input_map.pop("inputs")
    _params = input_map.pop("parameters", {})
    if invoke_type == "InvokeEndpointWithResponseStream":
        _params["stream"] = True
    return _inputs, _params