import json
from pathlib import Path
from typing import List, Optional
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

try:
    from datasets import Dataset, KeyDataset
except Exception:  # pragma: no cover - datasets is optional
    Dataset = None  # type: ignore
    KeyDataset = None  # type: ignore

import torch._dynamo
torch._dynamo.config.cache_size_limit = 256

DEFAULT_TEMPLATE = (
    "You are an empathetic conversation partner. "
    "Consider the user's intent — for example questioning, acknowledging, "
    "consoling, agreeing, encouraging, sympathizing, suggesting, or wishing. "
    "Reply to the latest user message in one or two concise sentences without "
    "describing your own feelings. Here is the conversation so far:\n{input}\n"
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


def load_inputs(path: str) -> List[str]:
    """Load a list of input strings from the dataset JSON file."""
    data = json.loads(Path(path).read_text())
    return [ex["input"] for ex in data]


def dump_records(path: str, prompts: List[str], responses: List[List[str]]) -> None:
    """Write generated prompts/responses to ``path`` as JSON."""

    records = []
    for inp, outs in zip(prompts, responses):
        for r in outs:
            records.append({"input": inp, "output": r})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def sanitize_output(text: str) -> str:
    """Clean model output and filter obvious junk.

    The Gemma model sometimes produces stray markdown or truncated fragments.
    This helper strips common artifacts and returns an empty string if the
    result does not look like a usable sentence.
    """

    text = text.strip()
    if not text:
        return ""

    # Remove code blocks and divider lines
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"^[\-*`#_=~]{2,}$", "", text, flags=re.MULTILINE)

    # Drop common prefixes or markup
    text = re.sub(r"(?i)^your response:\s*", "", text)
    text = re.sub(r"^#+\s*", "", text)

    # Remove bullet characters and repeated punctuation
    text = re.sub(r"^[\-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.replace("\n", " ")

    # Strip sign-offs or placeholders like [Your Name]
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"(?i)(warmly|sincerely|best regards|regards),?", "", text)

    text = text.strip()
    if not text or re.fullmatch(r"[\-*`#_=~\s]+", text):
        return ""

    # Trim to at most two sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 2:
        text = " ".join(sentences[:2]).strip()
    else:
        text = " ".join(sentences).strip()
    if not re.search(r"[.!?]$", text):
        text += "."

    # Very short fragments are rarely useful
    if len(text.split()) < 3:
        return ""

    return text


def generate_responses(
    inputs: List[str],
    *,
    original_inputs: Optional[List[str]] = None,
    model_name: str = "google/gemma-3-1b-it",
    max_new_tokens: int = 40,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_bf16: bool = True,
    batch_size: int = 1,
    num_candidates: int = 3,
    save_midway: bool = True,
) -> List[List[str]]:
    """Generate one or more responses for each input using the specified model.

    Parameters
    ----------
    inputs : List[str]
        Prompts to feed into the model.
    original_inputs : Optional[List[str]], optional
        If provided, the unformatted prompts used when saving progress mid-way.
    model_name : str, optional
        Model checkpoint to load.
    max_new_tokens : int, optional
        Number of tokens to generate.
    temperature : float, optional
        Sampling temperature.
    top_p : float, optional
        Top-p sampling parameter.
    use_bf16 : bool, optional
        Load model weights in ``bfloat16`` when running on CUDA.
    batch_size : int, optional
        How many prompts to process at once.
    num_candidates : int, optional
        How many responses to generate for each prompt.
    
    If ``save_midway`` is True, partial results are saved to ``save1.json``,
    ``save2.json``, and ``save3.json`` when 1/4, 1/2, and 3/4 of the prompts
    have been processed.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model_kwargs = {}
    if use_bf16 and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_full_text=False,
    )

    responses: List[List[str]] = []
    quarter_points = [len(inputs) * i // 4 for i in range(1, 4)]
    next_save = 0

    iterator = tqdm(total=len(inputs), desc="generating") if tqdm else None
    idx = 0

    if KeyDataset is not None and Dataset is not None:
        data = Dataset.from_dict({"text": inputs})
        output_iter = generator(
            KeyDataset(data, "text"),
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_candidates,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:  # fall back to sequential batches
        def _iter():
            for start in range(0, len(inputs), batch_size):
                batch = inputs[start : start + batch_size]
                outs = generator(
                    batch,
                    batch_size=len(batch),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_candidates,
                    pad_token_id=tokenizer.eos_token_id,
                )
                for o in outs:
                    yield o

        output_iter = _iter()

    for prompt, out in zip(inputs, output_iter):
        idx += 1
        if iterator:
            iterator.update(1)
        if not isinstance(out, list):
            out = [out]
            if not isinstance(out, list):
                out = [out]
            cands = []
            for o in out:
                result = o["generated_text"]
                if result.startswith(prompt):
                    result = result[len(prompt) :]
                cleaned = sanitize_output(result)
                if cleaned:
                    cands.append(cleaned)
            responses.append(list(dict.fromkeys(cands)))

            if save_midway and next_save < len(quarter_points) and idx == quarter_points[next_save]:
                dump_records(
                    f"save{next_save + 1}.json",
                    (original_inputs or inputs)[:idx],
                    responses,
                )
                next_save += 1

    if iterator:
        iterator.close()
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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Maximum tokens to generate for each reply",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bfloat16 model weights even if CUDA is available",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--num-candidates", type=int, default=3, help="Number of responses to generate per input")
    parser.add_argument(
        "--template",
        "--prompt-prefix",
        dest="template",
        default=DEFAULT_TEMPLATE,
        help="Format string used to create the prompt; must contain '{input}'",
    )
    args = parser.parse_args()

    raw_inputs = load_inputs(args.data)
    prompts = [args.template.format(input=inp) for inp in raw_inputs]
    responses = generate_responses(
        prompts,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_bf16=not args.no_bf16,
        batch_size=args.batch_size,
        num_candidates=args.num_candidates,
        save_midway=True,
        original_inputs=raw_inputs,
    )

    dump_records(args.output, raw_inputs, responses)


if __name__ == "__main__":
    main()
