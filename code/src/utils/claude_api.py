import json
import boto3
import time

with open('./aws_access.key', 'r') as f:
    AWS_ACCESS_KEY_ID = f.read().strip()

with open('./claude_api_aws.key', 'r') as f:
    AWS_SECRET_ACCESS_KEY = f.read().strip()

AWS_DEFAULT_REGION = "us-west-2"


boto3_client = boto3.client("bedrock-runtime",
    region_name=AWS_DEFAULT_REGION, 
    aws_access_key_id=AWS_ACCESS_KEY_ID, 
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


# model="claude-3-haiku-20240307"
BEDROCK_MODEL_NAME_MAP = {
    "claude": "anthropic.claude-v2:1",  # default to sonnet
    "haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "opus": "anthropic.claude-3-opus-20240229-v1:0"
}


def get_claude_response(llm, prompt, max_tokens=2048, temperature=0, tools=None, tool_choice=None):
    model = BEDROCK_MODEL_NAME_MAP.get(llm)
    if model is None:
        raise ValueError(f"Unknown LLM: {llm}")
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",    
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": temperature,
    }
    if tools:
        request_body["tools"] = tools
    if tool_choice:
        request_body["tool_choice"] = tool_choice

    request_body = json.dumps(request_body)

    response = boto3_client.invoke_model(
        body=request_body,
        modelId=model
    )
    response_body = json.loads(response.get('body').read())
    return response_body['content'][0]['text']