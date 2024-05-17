from huggingface_hub import InferenceClient
from huggingface_hub.inference._generated.types import TextGenerationOutputDetails, TextGenerationStreamOutputStreamDetails

client = InferenceClient(model="http://127.0.0.1:8080/invocations")

result = client.text_generation(
    prompt="tell me a story of the little red riding hood",
    max_new_tokens=100,
    details=True)
print(result)

assert isinstance(result.generated_text, str)
assert isinstance(result.details, TextGenerationOutputDetails)

result_list = []
for result in client.text_generation("How do you make cheese?",
                                     max_new_tokens=100,
                                     stream=True,
                                     details=True):
    print(result)
    result_list.append(result)
    assert isinstance(result.token.id, int)
    assert isinstance(result.token.logprob, float)

last_result = result_list[-1]
assert isinstance(last_result.generated_text, str)
assert isinstance(last_result.details, TextGenerationStreamOutputStreamDetails)
