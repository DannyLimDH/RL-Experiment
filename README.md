# RL-Experiment

This repository contains utilities for working with the Empathetic Dialogue dataset.

- `generate_candidates.py` generates multiple candidate responses for each input prompt using a specified language model. Results are saved to `candidate.json` by default.
- `generate_preference_dataset.py` scores the original and candidate responses with `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` to create a preference dataset named `LLMprefer.json` containing `prompt`, `chosen`, and `reject` fields.
