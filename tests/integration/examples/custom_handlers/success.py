from djl_python import Input, Output
from djl_python.async_utils import create_non_stream_output
from djl_python.encode_decode import decode
from vllm import LLM, SamplingParams
import json

llm = None

async def handle(inputs: Input):
    """Custom async handler with simple vLLM generate"""
    global llm
    
    print("[CUSTOM_HANDLER] Starting handle function")
    
    # Initialize vLLM LLM if not already done
    if llm is None:
        print("[CUSTOM_HANDLER] Initializing vLLM engine")
        properties = inputs.get_properties()
        model_id = properties.get("model_id", "gpt2")
        llm = LLM(model=model_id, tensor_parallel_size=1)
        print(f"[CUSTOM_HANDLER] vLLM engine initialized with model: {model_id}")
    
    # Parse input
    batch = inputs.get_batches()
    raw_request = batch[0]
    content_type = raw_request.get_property("Content-Type")
    decoded_payload = decode(raw_request, content_type)
    
    prompt = decoded_payload.get("inputs", "Hello")
    if not prompt or prompt.strip() == "":
        prompt = "Hello"
    
    print(f"[CUSTOM_HANDLER] Using prompt: {prompt}")
    
    # Create sampling parameters
    sampling_params = SamplingParams(max_tokens=50, temperature=0.8)
    
    # Generate using simple vLLM generate
    print("[CUSTOM_HANDLER] Starting generation")
    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text if outputs else "No output"
    
    print(f"[CUSTOM_HANDLER] Generated text: {generated_text}")
    
    # Create response with custom marker
    response = {
        "custom_handler_used": True,
        "generated_text": generated_text
    }
    
    print(f"[CUSTOM_HANDLER] Response created: {response}")
    
    output = create_non_stream_output(response)
    
    print("[CUSTOM_HANDLER] Output object created, returning")
    return output