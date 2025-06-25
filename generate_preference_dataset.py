import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None


def load_responses(path: str) -> Dict[str, List[str]]:
    """Return mapping from prompt to list of responses."""
    data = json.loads(Path(path).read_text())
    result: Dict[str, List[str]] = defaultdict(list)
    for record in data:
        result[record["input"]].append(record["output"])
    return result


def rate_response(generator, prompt: str, response: str) -> float:
    """Ask the model to rate the empathy of a candidate response."""
    query = (
        "Rate how empathetic the following reply is to the given prompt on a scale "
        "from 1 to 5. Respond with only the number.\n\n"
        f"Prompt: {prompt}\nReply: {response}\nRating:"
    )
    generated = generator(query, max_new_tokens=5, do_sample=False)[0]["generated_text"]
    match = re.search(r"(-?\d+(?:\.\d+)?)", generated)
    return float(match.group(1)) if match else 0.0


def pick_best_and_worst(generator, prompt: str, responses: List[str]):
    scores = [rate_response(generator, prompt, r) for r in responses]
    best = responses[int(max(range(len(scores)), key=lambda i: scores[i]))]
    worst = responses[int(min(range(len(scores)), key=lambda i: scores[i]))]
    return best, worst


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create preference dataset using DeepSeek")
    parser.add_argument("train", help="Path to train.json")
    parser.add_argument("candidates", help="Path to candidate dataset")
    parser.add_argument(
        "output", nargs="?", default="LLMprefer.json", help="Output dataset file"
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model checkpoint for scoring",
    )
    args = parser.parse_args()

    train_map = load_responses(args.train)
    cand_map = load_responses(args.candidates)

    prompts = sorted(set(train_map) | set(cand_map))

    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    iterator = prompts
    if tqdm:
        iterator = tqdm(iterator, desc="scoring")

    records = []
    for prompt in iterator:
        responses = train_map.get(prompt, []) + cand_map.get(prompt, [])
        if len(responses) < 2:
            continue
        best, worst = pick_best_and_worst(generator, prompt, responses)
        records.append({"prompt": prompt, "chosen": best, "reject": worst})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
