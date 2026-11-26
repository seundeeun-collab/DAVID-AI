"""Chat with the model after Stage 2 (python train.py --stage chat).

    python chat.py               # talk in the terminal; type 'exit' to quit
    from chat import chat; chat([{"role": "user", "content": "Hi!"}])
"""
from chat_format import DEFAULT_SYSTEM, STOP_TEXT, build_prompt, eot_id
from generate import _load, sample


def chat(messages, max_new_tokens=200, temperature=.7, top_p=.9, top_k=50, system=DEFAULT_SYSTEM, checkpoint_path=None, config_path=None):
    """Reply to a conversation: a list of {"role": "user"|"assistant", "content": str}."""
    tok, cfg, _, _ = _load(checkpoint_path, config_path)
    max_new_tokens = max(1, min(max_new_tokens, cfg.max_seq_len // 2))
    ids = build_prompt(tok, messages, cfg.max_seq_len - max_new_tokens, system)
    new = sample(ids, max_new_tokens, temperature, top_p, top_k, stop_ids={eot_id(tok)}, stop_text=STOP_TEXT,
                 checkpoint_path=checkpoint_path, config_path=config_path)
    reply = tok.decode(new, skip_special_tokens=True)
    for marker in STOP_TEXT: reply = reply.split(marker)[0]  # Never show the model starting the user's turn.
    return reply.strip()


if __name__ == "__main__":
    history = []
    print("DAVID-AI chat. Type 'exit' to quit.")
    while True:
        try: text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt): break
        if text.lower() in ("exit", "quit"): break
        if not text: continue
        history.append({"role": "user", "content": text})
        reply = chat(history); print(f"DAVID-AI: {reply}\n")
        history.append({"role": "assistant", "content": reply}); history = history[-20:]
