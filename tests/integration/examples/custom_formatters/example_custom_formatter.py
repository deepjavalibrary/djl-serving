from djl_python.output_formatter import output_formatter

@output_formatter
def custom_output_formatter(output):
    if hasattr(output, 'choices') and len(output.choices) > 0:
        return {
            "custom_formatter_applied": True,
            "generated_text": output.choices[0].text if hasattr(output.choices[0], 'text') else output.choices[0].message.content,
            "model": output.model,
        }
    return {"custom_formatter_applied": True}