import json
from pathlib import Path
from typing import List, Optional
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# ``torch._dynamo`` may not be available on older PyTorch versions. The
# cache size tweak improves performance when present but should not crash the
# script when running with a different installation.
try:  # pragma: no cover - optional optimisation
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 256
except Exception:
    torch._dynamo = None  # type: ignore

DEFAULT_TEMPLATE = (
    "You are in a conversation with someone. They just said something to you. "
    "Respond naturally - you might ask a question, acknowledge their feelings, "
    "console them, agree with them, encourage them, sympathize, suggest something, "
    "or wish them well.\n\n"
    "CRITICAL: Write ONLY your conversational response. "
    "Do NOT include any labels, formatting, or analysis.\n\n"
    "Example of what NOT to do:\n"
    "- 'Your Response:** That sounds hard.'\n"
    "- '**Analysis:** The user seems sad.'\n\n"
    "Example of what TO do:\n"
    "- 'That sounds really hard. How are you coping with it?'\n"
    "- 'I can understand why you'd feel that way.'\n\n"
    "Keep it natural and brief (1-2 sentences).\n\n"
    "{input}\n"
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
    """Clean model output and filter obvious junk."""

    text = text.strip().strip("* `_")
    if not text:
        return ""

    # Remove any response labels or formatting
    label_patterns = [
        r"^.*?Response\*\*:?:\s*",
        r"^.*?Analysis\*\*:?:\s*",
        r"^.*?Output\*\*:?:\s*",
        r"^.*?Answer\*\*:?:\s*",
        r"^\*\*.*?\*\*:?:\s*",
        r"^Your\s+Response\*\*.*?:?\s*",
        r"^Response\*\*.*?:?\s*",
        r"^\*\*Your\s+Response\*\*.*?:?\s*",
    ]

    for pattern in label_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove any remaining bold formatting
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)

    # Remove analysis-style language
    analytical_phrases = [
        "emotional state",
        "primary intent",
        "context analysis",
        "let's analyze",
        "based on this conversation",
        "the user appears",
        "relationship dynamics",
        "conversation history",
        "assessment",
        "emotional tone",
        "primary goal",
        "intent recognition",
    ]

    text_lower = text.lower()
    if any(phrase in text_lower for phrase in analytical_phrases):
        return ""

    # Remove structured formatting
    text = re.sub(r"^\s*[-*•]\s*", "", text)
    text = re.sub(r"^#+\s*", "", text)

    # Remove incomplete responses
    if text.endswith(":**") or text.endswith("**") or text.endswith(":"):
        return ""

    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Ensure natural sentence ending
    if text and not re.search(r"[.!?]$", text):
        text = text + "."

    # Filter out very short or very long responses
    if len(text.split()) < 3 or len(text.split()) > 30:
        return ""

    # Check if it looks like a natural conversational response
    if not text or re.fullmatch(r"[\-*`#_=~\s.!?]+", text):
        return ""

    return text


def is_natural_conversation(text: str) -> bool:
    """Check if the response sounds like natural conversation."""

    # Reject responses with labels
    if re.search(r"(response|analysis|output|answer)\s*\*\*", text, re.IGNORECASE):
        return False

    # Reject responses with analytical language
    analytical_terms = [
        "emotional state",
        "primary intent",
        "user appears",
        "based on this",
        "let's analyze",
        "assessment",
        "emotional tone",
        "relationship dynamics",
    ]

    if any(term in text.lower() for term in analytical_terms):
        return False

    # Prefer responses that sound conversational
    conversational_indicators = [
        "?",
        "that sounds",
        "how",
        "what",
        "why",
        "tell me",
        "i can",
        "you seem",
        "it feels",
        "i understand",
        "wow",
        "oh",
    ]

    if any(indicator in text.lower() for indicator in conversational_indicators):
        return True

    # Check length (natural conversations are usually brief)
    word_count = len(text.split())
    if 3 <= word_count <= 25:
        return True

    return False


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
    seed: Optional[int] = None,
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
    seed : Optional[int], optional
        If provided, sets the torch random seed for reproducible generation.
    
    If ``save_midway`` is True, partial results are saved to ``save1.json``,
    ``save2.json``, and ``save3.json`` when 1/4, 1/2, and 3/4 of the prompts
    have been processed.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if seed is not None:
        torch.manual_seed(seed)

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

    def batched_outputs():
        """Yield model outputs in ``batch_size`` chunks.

        ``text-generation`` pipelines sometimes ignore the provided ``batch_size``
        when iterating over a Dataset object. To ensure consistent batching we
        split the prompts manually regardless of whether ``datasets`` is
        installed. This avoids extremely slow one-by-one generation when the
        pipeline disregards ``batch_size``.
        """

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
            # ``transformers`` returns a flat list when batch size is 1
            if len(batch) == 1 and not isinstance(outs[0], list):
                outs = [outs]
            for o in outs:
                yield o

    output_iter = batched_outputs()
    for prompt, out in zip(inputs, output_iter):
        idx += 1
        if iterator:
            iterator.update(1)
        if not isinstance(out, list):
            out = [out]

        cands: List[str] = []
        for o in out:
            result = o["generated_text"]
            if result.startswith(prompt):
                result = result[len(prompt) :]
            cleaned = sanitize_output(result)
            if cleaned and is_natural_conversation(cleaned):
                cands.append(cleaned)

        uniq = list(dict.fromkeys(cands))[:num_candidates]
        responses.append(uniq)

        if save_midway and next_save < len(quarter_points) and idx == quarter_points[next_save]:
            dump_records(
                f"save{next_save + 1}.json",
                (original_inputs or inputs)[:idx],
                responses,
            )
            next_save += 1

    if iterator:
        iterator.close()

    for resp in responses:
        if len(resp) > num_candidates:
            del resp[num_candidates:]

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
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic generation",
    )
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
        seed=args.seed,
        original_inputs=raw_inputs,
    )

    dump_records(args.output, raw_inputs, responses)


if __name__ == "__main__":
    main()
