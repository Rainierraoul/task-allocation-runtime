"""RouteLLM BERT checkpoint and decision rule, pinned and run locally.

Rule follows lm-sys/RouteLLM/routellm/routers/routers.py, BERTRouter,
Apache-2.0. Loading directly avoids importing unrelated OpenAI embedding
clients. Model.eval() disables dropout; no remote model code is executed.
"""
import json
import time


class BertRouter:
    def __init__(self, checkpoint, revision):
        if not revision or revision == "main":
            raise ValueError("Pin the router checkpoint to a commit revision")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, revision=revision, trust_remote_code=False)
        self.model.eval()
        self.seconds = 0.0

    def score(self, context):
        start = time.perf_counter()
        inputs = self.tokenizer(json.dumps(context, sort_keys=True), return_tensors="pt", padding=True, truncation=True)
        with self.torch.inference_mode():
            probs = self.torch.softmax(self.model(**inputs).logits[0], dim=-1)
        self.seconds += time.perf_counter() - start
        return float(1 - probs[-2:].sum())


class MockRouter:
    """Smoke tests only: not a learned router and never empirical evidence."""
    seconds = 0.0

    def score(self, context):
        return 0.75 if context["task_id"] == "CALL" else 0.25
