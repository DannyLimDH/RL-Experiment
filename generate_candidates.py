import json
from pathlib import Path
from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


def load_inputs(path: str) -> List[str]:
    """Load a list of input strings from the dataset JSON file."""
    data = json.loads(Path(path).read_text())
    return [ex["input"] for ex in data]


def generate_responses(
    inputs: List[str],
    model_name: str = "google/gemma-3-1b-it",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[str]:
    """Generate a single response for each input using the specified model."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    responses = []
    for prompt in inputs:
        result = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )[0]["generated_text"]

        # Remove the prompt from the generated text if present
        if result.startswith(prompt):
            result = result[len(prompt) :]
        responses.append(result.strip())
    return responses


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate candidate responses for Empathetic Dialogue prompts using"
            " the Gemma model"
        )
    )
    parser.add_argument("data", help="Path to train.json file")
    parser.add_argument(
        "output",
        nargs="?",
        default="candidate.json",
        help="Where to write the candidate dataset",
    )
    parser.add_argument("--model", default="google/gemma-3-1b-it", help="Model to use for generation")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()

    inputs = load_inputs(args.data)
    responses = generate_responses(
        inputs,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            [{"input": i, "output": r} for i, r in zip(inputs, responses)],
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
