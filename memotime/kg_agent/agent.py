


# =============================
# file: kg_agent/agent.py
# =============================
import re
import json
from typing import Optional, Tuple, Dict, Any
from .registry import TemplateRegistry, TemplateCard
from .prompts import LLM_SYSTEM_PROMPT, LLM_MATCH_PROMPT, LLM_GENERATE_TEMPLATE_PROMPT
from .llm import LLM

RULES = [
    (re.compile(r"\bafter\b.*\bfirst\b", re.I), "after_first"),
    (re.compile(r"\bbefore\b.*\blast\b", re.I), "before_last"),
    (re.compile(r"\bsame\s*(day|month|year)\b", re.I), "same_bucket"),
    (re.compile(r"\b(during|between)\b", re.I), "during_between"),
]

class Agent:
    def __init__(self, registry: TemplateRegistry):
        self.registry = registry

    def _rule_match(self, question: str) -> Optional[str]:
        for rx, wid in RULES:
            if rx.search(question):
                return wid
        return None

    def select_template(self, question: str) -> Tuple[str, Optional[TemplateCard]]:
        wid = self._rule_match(question)
        if wid and wid in self.registry.list_templates():
            return wid, self.registry.load_card(wid)
        wf_list = self.registry.list_templates()
        workflows_str = "\n".join(f"- {w}" for w in wf_list)
        choice = LLM.call(LLM_SYSTEM_PROMPT, LLM_MATCH_PROMPT.format(question=question, workflows=workflows_str)).strip()
        if choice in wf_list:
            return choice, self.registry.load_card(choice)
        return "NONE", None

    def follow_or_create(self, question: str) -> Tuple[TemplateCard, bool]:
        wid, card = self.select_template(question)
        if card is not None:
            return card, False
        gen = LLM.call(LLM_SYSTEM_PROMPT, LLM_GENERATE_TEMPLATE_PROMPT.format(question=question))
        try:
            obj = json.loads(gen)
            workflow_id = obj["workflow_id"]
            example_q = obj.get("example_question", question)
            yaml_text = obj["yaml"]
            self.registry.add_card(workflow_id, yaml_text, example_q)
            card = self.registry.load_card(workflow_id)
            return card, True
        except Exception:
            fallback = self.registry.load_card("same_bucket")
            return fallback, False

    def solve_via_template(self, question: str, topic_entity_hint: Optional[str] = None) -> Dict[str, Any]:
        card, created = self.follow_or_create(question)
        try:
            self.registry.add_example(card.workflow_id, question)
        except Exception:
            pass
        from .kg_ops import KG
        linked = KG.entity_link(question, topic_hint=topic_entity_hint)
        answer = KG.run_workflow(card, question, linked)
        answer["_meta"] = {"workflow_id": card.workflow_id, "created_new_template": created}
        return answer

