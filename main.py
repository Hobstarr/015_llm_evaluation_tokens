import json
import time
from typing import List

from transformers import pipeline, set_seed, GPT2Tokenizer
from pydantic import BaseModel

MODEL_NAME = 'gpt2'


class EvaluationResult(BaseModel):
    """Evaluation Result pydantic model"""
    model_input: str
    model_output: List[str]
    input_tokens: int
    output_tokens: int
    time_taken: float
    token_per_sec: float


with open('databricks-dolly-15k.jsonl', 'r', encoding="UTF-8") as f:
    data = [json.loads(line) for line in f]

generator = pipeline('text-generation', model=MODEL_NAME, max_new_tokens=100)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
set_seed(42)

for entry in data[:1000]:
    # Structure the input prompt for the model.
    model_input = (f"Instruction:{entry['instruction']}\n\n" +
                   f"Context:{entry['context']}\n\n" +
                   "Response:")

    # Ensure inputs don't exceed max input length for model.
    if len(model_input) > 1024:
        model_input = model_input[:1024]

    # Record the length of the input
    input_tokens = len(tokenizer.encode(model_input))

    # Time the Generation: (here we return 5 generations for each prompt)
    start = time.time()
    responses = generator(model_input, num_return_sequences=1)
    time_taken = time.time() - start

    # For each response, record the number of tokens and append the output
    total_tokens = 0
    output = []
    for response in responses:
        total_tokens += len(tokenizer.encode(response['generated_text']))
        output.append(response['generated_text'])

    # Store as EvaluationResult
    result = EvaluationResult(model_input=model_input,
                              model_output=output,
                              input_tokens=input_tokens,
                              output_tokens=total_tokens,
                              time_taken=round(time_taken, 4),
                              token_per_sec=round(total_tokens/time_taken, 4))

    # Dump result to jsonl file
    with open("results_single.jsonl", "a", encoding="UTF-8") as f:
        f.write(json.dumps(result.model_dump()) + "\n")
