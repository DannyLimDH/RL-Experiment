import json
from pathlib import Path
from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


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
    use_fp16: bool = True,
) -> List[str]:
    """Generate a single response for each input using the specified model.

    Parameters
    ----------
    inputs : List[str]
        Prompts to feed into the model.
    model_name : str, optional
        Model checkpoint to load.
    max_new_tokens : int, optional
        Number of tokens to generate.
    temperature : float, optional
        Sampling temperature.
    top_p : float, optional
        Top-p sampling parameter.
    use_fp16 : bool, optional
        Load model weights in ``float16`` when running on CUDA.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs = {}
    if use_fp16 and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    total = len(inputs)
    responses = []
    if tqdm:
        iterator = tqdm(inputs, desc="generating", unit="req")
        for prompt in iterator:
            result = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )[0]["generated_text"]

            if result.startswith(prompt):
                result = result[len(prompt) :]
            responses.append(result.strip())
    else:
        for idx, prompt in enumerate(inputs, 1):
            result = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )[0]["generated_text"]

            if result.startswith(prompt):
                result = result[len(prompt) :]
            responses.append(result.strip())
            print(f"Generated {idx}/{total} responses", end="\r")
        print()
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
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 model weights even if CUDA is available",
    )
    args = parser.parse_args()

    inputs = load_inputs(args.data)
    responses = generate_responses(
        inputs,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_fp16=not args.no_fp16,
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

