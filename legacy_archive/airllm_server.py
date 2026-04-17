import time
import uuid

import torch
from airllm import AirLLMQWen2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Model ID for Qwen2.5-Coder-7B-Instruct
model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"

print(f"Loading model: {model_id} via AirLLM (layer-wise)...")
model = AirLLMQWen2(model_id)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])

    # Simple prompt extraction from messages
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"{role}: {content}\n"
    prompt += "assistant: "

    print(f"Generating response for prompt: {prompt[:50]}...")

    # Tokenize
    input_tokens = model.tokenizer(
        prompt, return_tensors="pt", return_attention_mask=False
    ).input_ids.cuda()

    # Generate
    generation_output = model.generate(input_tokens, max_new_tokens=500, use_cache=True)

    # Decode
    output_text = model.tokenizer.decode(generation_output[0], skip_special_tokens=True)

    # Extract only the assistant's new response
    response_content = (
        output_text[len(prompt) :].strip()
        if output_text.startswith(prompt)
        else output_text
    )

    # OpenAI compatible response format
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_content},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
