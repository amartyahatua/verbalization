"""
Small compatibility helper: different HF model architectures expose
their transformer blocks under different attribute paths.

- GPT-2 family:                 model.transformer.h
- Llama, Gemma-2, Qwen2, ...:   model.model.layers
- GPT-NeoX / Pythia:            model.gpt_neox.layers

Everything in this codebase that needs to register a forward hook on
"layer N" should go through get_transformer_layers() instead of
hardcoding one architecture's path — that's what broke when the
default model switched from Llama-family to GPT-2.
"""


def get_transformer_layers(model):
    candidates = (
        ("transformer", "h"),       # GPT-2 family
        ("model", "layers"),        # Llama, Gemma-2, Qwen2, Mistral, ...
        ("gpt_neox", "layers"),     # GPT-NeoX / Pythia
    )
    for path in candidates:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue

    raise AttributeError(
        f"Don't know how to locate transformer layers for {type(model).__name__}. "
        f"Add its attribute path to get_transformer_layers() in model_utils.py "
        f"(try `print(model)` to find the right one)."
    )
