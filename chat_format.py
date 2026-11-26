"""One shared conversation format for training data, the chat endpoint and evaluation.

A conversation is written as plain text the GPT-2 tokenizer already understands:

    <|system|>
    You are DAVID-AI ...
    <|user|>
    Hi! What is a noun?
    <|assistant|>
    A noun is a word for a person, place or thing.<|endoftext|>
    <|user|>
    ...

Only the assistant's reply tokens (and the <|endoftext|> that ends each reply) are
trained on. That mask is what teaches the model to answer instead of copying the user.
"""
SYSTEM, USER, ASSISTANT = "<|system|>\n", "<|user|>\n", "<|assistant|>\n"
STOP_TEXT = ("<|user|>", "<|system|>", "<|assistant|>")
DEFAULT_SYSTEM = "You are DAVID-AI, a friendly and patient assistant. Answer in clear, simple English."


def eot_id(tokenizer):
    eot = tokenizer.token_to_id("<|endoftext|>")
    return 50256 if eot is None else eot


def _clean(text):
    # A literal <|endoftext|> inside a message would become the real end token.
    return (text or "").replace("<|endoftext|>", "").strip()


def encode_conversation(tokenizer, messages, eot=None):
    """Return (token_ids, train_mask) for a list of {"role", "content"} messages."""
    eot = eot_id(tokenizer) if eot is None else eot
    ids, mask = [], []
    def add(text, train):
        seg = tokenizer.encode(text).ids
        ids.extend(seg); mask.extend([train] * len(seg))
    for m in messages:
        role, text = m.get("role"), _clean(m.get("content"))
        if role == "system":
            add(SYSTEM + text + "\n", 0)
        elif role == "user":
            add(USER + text + "\n", 0)
        elif role == "assistant":
            add(ASSISTANT, 0)
            add(text, 1)
            ids.append(eot); mask.append(1)  # Learn to stop at the end of the reply.
            add("\n", 0)
    return ids, mask


def build_prompt(tokenizer, messages, max_tokens, system=DEFAULT_SYSTEM):
    """Token ids for the system line + as many recent turns as fit, ending ready for a reply."""
    eot = eot_id(tokenizer)
    head = encode_conversation(tokenizer, [{"role": "system", "content": system}], eot)[0] if system else []
    tail = tokenizer.encode(ASSISTANT).ids
    turns = [m for m in messages if m.get("role") in ("user", "assistant") and _clean(m.get("content"))]
    while turns:
        body = encode_conversation(tokenizer, turns, eot)[0]
        if len(head) + len(body) + len(tail) <= max_tokens:
            return head + body + tail
        turns = turns[1:]  # Drop the oldest turn first.
    return (head + tail)[-max_tokens:]
