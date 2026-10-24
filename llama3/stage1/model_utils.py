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


def format_prompt_for_chat(tokenizer, text: str, use_chat_template: bool = True) -> str:
    """
    Wraps `text` as a single user turn via the tokenizer's chat
    template when available and requested, so an instruction-tuned
    model (e.g. Qwen2-0.5B-Instruct) sees a properly formatted
    conversation instead of a raw text completion it was never tuned
    to expect. Falls back to returning `text` unchanged when the
    tokenizer has no chat template (e.g. GPT-2) or
    use_chat_template=False, so this is safe to call unconditionally
    regardless of which model is configured.

    Deliberately NOT used for Stage 1's context-processing (J-lens
    extraction, CounterFact probes, background-context steering) --
    those operate on generic text, not a conversational turn, and
    should stay as plain completions regardless of model family.
    Only the elicitation-style "what concept is active" prompts in
    Stages 2-7 are conversational and need this.
    """
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def get_transformer_layers(model):
    # If this is a PEFT-wrapped model, unwrap to the actual base model first.
    # PEFT's own wrapper defines a `.model` attribute pointing at the base
    # model -- this collides with a base model's OWN `.model.layers` path
    # (Llama, Qwen2, Gemma-2, Mistral, ...), since `.model` then resolves to
    # PEFT's wrapper attribute directly (Python finds it without needing
    # __getattr__ delegation) rather than delegating further into the real
    # model, landing one level too shallow. GPT-2's `.transformer.h` path
    # doesn't collide with anything PEFT defines, which is why this bug
    # didn't surface until a Llama-family model was tried.
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()

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
