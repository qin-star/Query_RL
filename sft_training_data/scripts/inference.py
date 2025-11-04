from modelscope import snapshot_download

model_dir = "/home/jovyan2/query_rl/model/Qwen3-8B"
adapter_dir = "/home/jovyan2/query_rl/output/v19-20251028-165612/checkpoint-120"


def infer_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype='auto', device_map='auto', trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    messages = [{
        'role': 'user',
        'content': 'who are you?'
    }]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors='pt', add_special_tokens=False).to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f'response: {response}')
    return response


def infer_swift():
    from swift.llm import InferRequest, RequestConfig, PtEngine

    engine = PtEngine(model_dir, adapters=[adapter_dir])

    messages = [{
        'role': 'user',
        'content': 'who are you?'
    }]
    request_config = RequestConfig(max_tokens=512, temperature=0)
    resp_list = engine.infer([InferRequest(messages=messages)], request_config=request_config)
    response = resp_list[0].choices[0].message.content
    print(f'response: {response}')
    return response

response2 = infer_swift()
print(response2)
