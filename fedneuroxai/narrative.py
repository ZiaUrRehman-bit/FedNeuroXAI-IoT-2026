"""
Client-side LLM narrative generation (Section 3.5) and the automatic
threshold-faithfulness check (Section 5.5).

The top-kappa rules per class (ranked by confidence) are sent to a locally
hosted LLM (Llama-3.2-3B-Instruct in the paper, fp32 on CPU, greedy decoding).
Requires the optional dependencies in requirements-llm.txt.
"""
import re

import pandas as pd

from . import config

NUM_RE = re.compile(r"-?\d+\.\d+")
ASSISTANT_MARKER = "<|start_header_id|>assistant<|end_header_id|>"


def build_prompt(df_rules: pd.DataFrame, top_k: int = config.NARRATIVE_TOP_RULES,
                 class_labels: dict = None):
    """Returns (system_msg, user_msg, rules_text). rules_text is the exact
    rule text shown to the LLM, used for the faithfulness check."""
    class_labels = class_labels or {}
    rules_by_class = {}
    for cls in sorted(df_rules["consequent"].str.split("=").str[1].astype(int).unique()):
        sub = df_rules[df_rules["consequent"] == f"class={cls}"]
        rules_by_class[cls] = sub.sort_values("confidence", ascending=False).head(top_k)

    def tag(c):
        return f"class={c}" + (f": {class_labels[c]}" if c in class_labels else "")

    class_info = "\n".join(f"- {tag(c)}" for c in rules_by_class)
    rules_text = ""
    for cls, top in rules_by_class.items():
        lbl = f" ({class_labels[cls]})" if cls in class_labels else ""
        block = "\n".join(
            f"{i+1}. IF {r['antecedents']} THEN class={cls} "
            f"(support={r['support']:.3f}, confidence={r['confidence']:.3f}, zhang={r['zhang']:.3f})"
            for i, (_, r) in enumerate(top.iterrows())
        )
        rules_text += f"Rules for class={cls}{lbl}:\n{block}\n\n"

    system_msg = (
        "You are an explainable AI assistant helping a clinician understand a machine "
        "learning classifier trained via federated learning across multiple hospital sites.\n"
        f"The model predicts:\n{class_info}\n\n"
        "You are given IF-THEN association rules mined ONLY from this one hospital site's "
        "own local patient data. Do NOT invent patterns or alter numeric thresholds."
    )
    user_msg = rules_text + "Write a global explanation covering all classes above."
    return system_msg, user_msg, rules_text


def load_llm(llm_path: str = None):
    """Load the LLM as a Hugging Face text-generation pipeline (fp32, CPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipe

    llm_path = llm_path or config.LLM_PATH
    tok = AutoTokenizer.from_pretrained(llm_path)
    mdl = AutoModelForCausalLM.from_pretrained(llm_path, dtype=torch.float32)
    pipe = hf_pipe("text-generation", model=mdl, tokenizer=tok, dtype=torch.float32)
    return tok, pipe


def generate_narrative(tok, pipe, system_msg: str, user_msg: str, max_new_tokens: int = 400) -> str:
    """Greedy decoding, so the narrative is deterministic for a given rule set."""
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    full = out[0]["generated_text"]
    return full.split(ASSISTANT_MARKER)[-1].strip() if ASSISTANT_MARKER in full else full


def threshold_faithfulness(narrative: str, rules_text: str) -> dict:
    """Every decimal number in the narrative must appear verbatim among the
    numbers in the rules the LLM was given."""
    narrative_nums = NUM_RE.findall(narrative)
    source_nums = set(NUM_RE.findall(rules_text))
    unmatched = [n for n in narrative_nums if n not in source_nums]
    n_total = len(narrative_nums)
    n_matched = n_total - len(unmatched)
    return {
        "n_thresholds_in_narrative": n_total,
        "n_matched_verbatim": n_matched,
        "n_unmatched": len(unmatched),
        "unmatched_values": unmatched,
        "faithfulness_rate": round(n_matched / n_total, 3) if n_total else None,
    }
