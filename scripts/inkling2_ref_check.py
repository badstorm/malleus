"""Throwaway: load the Inkling-2-Mini reference and print its argmax for a short prompt."""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "/models/thinkingmachines/Inkling-2-Mini"

print("loading processor", flush=True)
proc = AutoProcessor.from_pretrained(MODEL)
print("loading model", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
)
model.eval()
print("loaded", flush=True)

msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
enc = proc.apply_chat_template(
    msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True
)
ids = enc["input_ids"]
print("ids", ids.shape, flush=True)
with torch.no_grad():
    out = model(input_ids=ids.to("cuda")).logits[0].float()
print("logits", out.shape, flush=True)
print("last argmax", out[-1].argmax().item(), flush=True)
print("last token", proc.tokenizer.decode([out[-1].argmax().item()]), flush=True)
print("next 8 argmax", out[-1].topk(8).indices.tolist(), flush=True)
