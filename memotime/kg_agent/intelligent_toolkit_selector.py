#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Toolkit Selector (Simplified)
----------------------------------------
aim:  
1) route to specialized toolkits based on "question type"
2) simplify toolkits collection and parameter mapping, stable interface, easy to combine
3) prioritize rule routing; only fallback to LLM selection when not clear (compatible with original LLM interface)

dependencies:
- .llm.LLM, .prompts.LLM_SYSTEM_PROMPT (optional)
- .kg_ops.KG.get_entity_name only used to map seed id to entity name
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
import json

# ========== optional: LLM ==========
try:
    from .llm import LLM
    from .prompts import LLM_SYSTEM_PROMPT
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

# ========== optional: KG name parsing ==========
try:
    from .kg_ops import KG
    KG_AVAILABLE = True
except Exception:
    KG_AVAILABLE = False


# ============================
# data structure (compatible with original interface)
# ============================

@dataclass
class ToolkitExample:
    """Toolkit example (only used to show to LLM)"""
    name: str
    english_name: str
    description: str
    example_call: str
    use_cases: List[str]
    time_requirements: List[str]


class IntelligentToolkitSelector:
    """
    simplified toolkit selector:
    - small and beautiful toolkit collection
    - rule priority + LLM fallback
    """

    # ---- simplified "question type" enum (internal agreement) ----
    QTYPE_WHEN = "WHEN"                 # when did ...
    QTYPE_AFTER = "AFTER"               # after ...
    QTYPE_BEFORE = "BEFORE"             # before ...
    QTYPE_BETWEEN = "BETWEEN"           # between ... and ...
    QTYPE_SAME_DAY = "SAME_DAY"         # same day
    QTYPE_SAME_MONTH = "SAME_MONTH"     # same month
    QTYPE_DIRECT = "DIRECT"             # direct connection between two entities
    QTYPE_FIRST_AFTER = "FIRST_AFTER"   # first after
    QTYPE_LAST_BEFORE = "LAST_BEFORE"   # last before
    QTYPE_TIMELINE = "TIMELINE"         # timeline
    QTYPE_GENERAL = "GENERAL"           # general retrieval (default)

    # ---- simplified toolkit collection (uniform mapping to underlying method name + parameters) ----
    # only retain 8 underlying methods, cover 80% needs
    _METHOD_MAPPING: Dict[str, Dict[str, Any]] = {
        # 1) one hop retrieval + basic time filtering (direction可选)
        "OneHop": {
                "method": "retrieve_one_hop",
            "param_mapping": {"entity": "query", "direction": "direction", "limit": "limit",
                              "after": "after", "before": "before", "between": "between",
                              "same_day": "same_day", "same_month": "same_month",
                              "sort_by_time": "sort_by_time"}
        },
        # 2) after first
        "AfterFirst": {
                "method": "find_after_first",
            "param_mapping": {"entity": "entity", "after": "reference_time", "limit": "limit"}
            },
        # 3) before last
        "BeforeLast": {
                "method": "find_before_last",
            "param_mapping": {"entity": "entity", "before": "reference_time", "limit": "limit"}
            },
        # 4) time range
        "BetweenRange": {
                "method": "find_between_times",
                "param_mapping": {"entity": "entity", "between": "time_range", "limit": "limit"}
            },
        # 5) all events on day (global)
        "DayEvents": {
                "method": "events_on_day",
                "param_mapping": {"date": "day", "limit": "limit"}
            },
        # 6) all events on month (global)
        "MonthEvents": {
                "method": "events_in_month",
                "param_mapping": {"month": "month", "limit": "limit"}
            },
        # 6) all events on year (global)
        "YearEvents": {
                "method": "events_in_year",
                "param_mapping": {"year": "year", "limit": "limit"}
            },
        # 7) direct connection between two entities
        "DirectConnection": {
                "method": "find_direct_connection",
            "param_mapping": {"entity1": "entity1", "entity2": "entity2",
                              "relation_types": "relation_types", "direction": "direction",
                              "after": "after", "before": "before", "between": "between",
                              "same_day": "same_day", "same_month": "same_month",
                              "sort_by_time": "sort_by_time",
                              "limit": "limit"}
            },
        # 8) timeline (本质：OneHop + sort_by_time)
        "Timeline": {
                "method": "retrieve_one_hop",
            "param_mapping": {"entity": "query", "direction": "direction",
                              "limit": "limit", "sort_by_time": "sort_by_time",
                              "after": "after", "before": "before"}
        },
    }

    # ---- if using LLM, show simplified examples for "optional toolkits" ----
    def _initialize_toolkit_examples(self) -> List[ToolkitExample]:
        return [
            ToolkitExample(
                name="one hop retrieval (optional time sequence filtering)",
                english_name="OneHop",
                description="get one hop relation list of entity, can append before/after/between/same_day/same_month filtering and time sorting",
                example_call='OneHop(entity="Iraq", direction="both", after="2005-01-01", limit=50, sort_by_time=True)',
                use_cases=["general retrieval", "time line initial building", "candidate path discovery"],
                time_requirements=["optional: after/before/between/same_day/same_month"]
            ),
            ToolkitExample(
                name="first event after",
                english_name="AfterFirst",
                description="get first event after specific time point",
                example_call='AfterFirst(entity="Iraq", after="2006-01-05", limit=1)',
                use_cases=["first after", "who... first after"],
                time_requirements=["start time point after"]
            ),
            ToolkitExample(
                name="last event before",
                english_name="BeforeLast",
                description="get last event before specific time point",
                example_call='BeforeLast(entity="Iraq", before="2006-01-05", limit=1)',
                use_cases=["last before", "who... last before"],
                time_requirements=["end time point before"]
            ),
            ToolkitExample(
                name="time range event",
                english_name="BetweenRange",
                description="get event within given time range",
                example_call='BetweenRange(entity="Iraq", between=("2005-01-01", "2005-12-31"), limit=20)',
                use_cases=["during/between/期间"],
                time_requirements=["start and end time"]
            ),
            ToolkitExample(
                name="all events on day (global)",
                english_name="DayEvents",
                description="get all events on given day",
                example_call='DayEvents(date="2007-12-31", limit=100)',
                use_cases=["same day co-occurrence/global on day"],
                time_requirements=["specific date"]
            ),
            ToolkitExample(
                name="all events on month (global)",
                english_name="MonthEvents",
                description="get all events on given month",
                example_call='MonthEvents(month="2007-12", limit=200)',
                use_cases=["same month co-occurrence/global on month"],
                time_requirements=["specific month"]
            ),
            ToolkitExample(
                name="all events on year (global)",
                english_name="YearEvents",
                description="get all events on given year",
                example_call='YearEvents(year="2007", limit=200)',
                use_cases=["same year co-occurrence/global on year"],
                time_requirements=["specific year"]
            ),
            ToolkitExample(
                name="direct connection between two entities",
                english_name="DirectConnection",
                description="whether two entities are directly connected and its edge",
                example_call='DirectConnection(entity1="China", entity2="Iraq", direction="both", limit=200, after="2005-01-01", before="2005-12-31", between="2005-01-01", same_day="2005-01-01", same_month="2005-01-01", sort_by_time=True)',
                use_cases=["co-occurrence/direct relation judgment", "validate candidate"],
                time_requirements=["no special requirements"]
            ),
            ToolkitExample(
                name="timeline",
                english_name="Timeline",
                description="entity timeline view (one hop relation sorted by time)",
                example_call='Timeline(entity="Iraq", direction="both", after="2000-01-01", limit=100, sort_by_time=True)',
                use_cases=["time sequence and evolution"],
                time_requirements=["optional: after/before"]
            ),
        ]

    def __init__(self):
        self.toolkit_examples = self._initialize_toolkit_examples()
        self.method_mapping = self._METHOD_MAPPING  # compatible with old attribute name

        # pre-compile common regex (mixed Chinese and English)
        self._re_date = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
        self._re_month = re.compile(r"\b(\d{4}-\d{2})\b")
        self._re_between = re.compile(
            r"(?:between|between)\s*(\d{4}-\d{2}-\d{2})\s*(?:and|to|-|~)\s*(\d{4}-\d{2}-\d{2})",
            flags=re.IGNORECASE
        )

    # ============================
    # public entry
    # ============================
    def select_toolkits(self,
                        subquestion: str,
                        seeds: List[int],
                        context: Dict[str, Any],
                        question_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        compatible with old version: output dictionary contains toolkit_name (=underlying method name), parameters, reasoning, priority, original_name
        """
        try:
            # 1) seed entity names
            entity_names = self._seed_ids_to_names(seeds)

            # 2) question type classification (only used for prompt routing, not used for rules)
            qtype = question_type

            # 3) extract time hints from question
            time_hints = self._extract_time_hints(subquestion)

            # 4) completely use LLM selection
            if LLM_AVAILABLE:
                return self._llm_selection(subquestion, seeds, context, question_type, time_hints, entity_names)
            else:
                print("[Selector] LLM not available, using default toolkit")
                return [self._default_toolkit(entity_names)]

        except Exception as e:
            print(f"[Selector] Failed: {e}")
            return [self._default_toolkit(self._seed_ids_to_names(seeds))]

    # ============================
    # removed routing rules - now completely handled by LLM
    # ============================
    # NOTE: all routing decisions now completely handled by LLM in _llm_selection method
    # this method has been replaced by _build_specialized_prompt and related toolkit prompts

    # ============================
    # type recognition (rules)
    # ============================
    def _classify_question_type(self, text: str, entity_names: List[str]) -> str:
        t = (text or "").lower()

        # clear words
        if re.search(r"\bwhen\b|when", t):
            return self.QTYPE_WHEN
        if re.search(r"\bfirst\b.*\bafter\b|first after", t):
            return self.QTYPE_FIRST_AFTER
        if re.search(r"\blast\b.*\bbefore\b|last before", t):
            return self.QTYPE_LAST_BEFORE
        if re.search(r"\bafter\b|after", t):
            return self.QTYPE_AFTER
        if re.search(r"\bbefore\b|before", t):
            return self.QTYPE_BEFORE
        if re.search(r"\bbetween\b|between", t):
            return self.QTYPE_BETWEEN
        if re.search(r"same day", t):
            return self.QTYPE_SAME_DAY
        if re.search(r"same month", t):
            return self.QTYPE_SAME_MONTH
        if (len(entity_names) >= 2) or re.search(r"direct connection|between .* and .*", t):
            return self.QTYPE_DIRECT
        if re.search(r"timeline|timeline", t):
            return self.QTYPE_TIMELINE

        return self.QTYPE_GENERAL

    # ============================
    # time hints extraction (loose)
    # ============================
    def _extract_time_hints(self, text: str) -> Dict[str, Any]:
        """try to extract day / month / after / before / between from text"""
        t = text or ""
        hints: Dict[str, Any] = {}

        # between interval (priority)
        m_between = self._re_between.search(t)
        if m_between:
            hints["between"] = (m_between.group(1), m_between.group(2))

        # specific date (as reference for day or after/before)
        m_day = self._re_date.search(t)
        if m_day:
            hints["day"] = m_day.group(1)

        # month
        m_month = self._re_month.search(t)
        if m_month:
            hints["month"] = m_month.group(1)

        # after/before (simple heuristic: if keyword + date appears, it is considered as reference)
        if re.search(r"\bafter\b|after", t) and m_day:
            hints["after"] = m_day.group(1)
        if re.search(r"\bbefore\b|before", t) and m_day:
            hints["before"] = m_day.group(1)

        return hints

    # ============================
    # LLM selector (main method)
    # ============================
    def _llm_selection(self, subquestion: str, seeds: List[int], context: Dict[str, Any], 
                       qtype: str, time_hints: Dict[str, Any], entity_names: List[str]) -> List[Dict[str, Any]]:
        """completely use LLM for toolkit selection, select specialized prompt based on question type"""
        
        # build basic information
        seed_info = self._build_seed_info(seeds)
        context_info = self._build_context_info(context)
        
        # select specialized prompt based on question type
        prompt = self._build_specialized_prompt(
            subquestion=subquestion,
            seed_info=seed_info,
            context_info=context_info,
            qtype=qtype,
            time_hints=time_hints,
            entity_names=entity_names
        )
        
        try:
            print(f"LLM toolkit selection prompt: {prompt}")
            # exit()
            # use toolkit selection model
            from config import TPKGConfig
            toolkit_selection_model = TPKGConfig.TOOLKIT_SELECTION_LLM_MODEL
            resp = LLM.call(LLM_SYSTEM_PROMPT, prompt, model=toolkit_selection_model)
            print(prompt)
            print(f"LLM toolkit selection response: {resp}")
            # exit()
            selected = self._parse_llm_response(resp)
            result = self._map_to_methods(selected)
            
            if not result:
                # if no valid selection, use default
                return [self._default_toolkit(entity_names)]
            return result
            
        except Exception as e:
            print(f"[Selector] LLM selection failed: {e}")
            return [self._default_toolkit(entity_names)]

    # ============================
    # mapping and default
    # ============================
    def _map_to_methods(self, plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """map original_name + friendly parameters in plans to underlying method + final parameters"""
        out = []
        for p in plans:
            orig = p.get("original_name")
            params = dict(p.get("parameters", {}))
            mapping = self.method_mapping.get(orig)
            if not mapping:
                # if no mapping found, default to OneHop
                mapped = {
                    "toolkit_name": "retrieve_one_hop",
                    "parameters": {"query": params.get("entity"), "direction": "both", "limit": params.get("limit", 100)},
                    "reasoning": p.get("reasoning", f"default mapping for {orig}"),
                    "priority": p.get("priority", 9),
                    "original_name": orig or "OneHop"
                }
                out.append(mapped)
                continue

            # param mapping
            final_params = {}
            for human_k, real_k in mapping.get("param_mapping", {}).items():
                if human_k in params:
                    final_params[real_k] = params[human_k]
            # other parameters pass through
            for k, v in params.items():
                if k not in mapping.get("param_mapping", {}):
                    final_params[k] = v

            out.append({
                "toolkit_name": mapping["method"],  # already underlying method name
                "parameters": final_params,
                "reasoning": p.get("reasoning", ""),
                "priority": p.get("priority", 5),
                "original_name": orig
            })
        # priority from small to large
        out.sort(key=lambda x: x.get("priority", 5))
        return out

    def _default_toolkit(self, entity_names: List[str]) -> Dict[str, Any]:
        ent = entity_names[0] if entity_names else None
        return {
            "toolkit_name": "retrieve_one_hop", 
            "parameters": {"query": ent, "direction": "both", "limit": 100, "sort_by_time": True},
            "reasoning": "default fallback: one hop retrieval + time sorting",
            "priority": 9,
            "original_name": "OneHop"
        }

    # ============================
    # helper methods (display / parse)
    # ============================
    def _build_toolkit_info(self) -> str:
        lines = []
        for i, t in enumerate(self.toolkit_examples, 1):
            lines.append(f"{i}. {t.name}")
            lines.append(f"   English Name: {t.english_name}")
            lines.append(f"   Description: {t.description}")
            lines.append(f"   Example Call: {t.example_call}")
            lines.append(f"   Use Cases: {', '.join(t.use_cases)}")
            lines.append(f"   Time Requirements: {', '.join(t.time_requirements)}")
            lines.append("")
        return "\n".join(lines)
    
    def _build_seed_info(self, seeds: List[int]) -> str:
        if not seeds:
            return "No seed entities"
        names = self._seed_ids_to_names(seeds)
        out = []
        for sid, nm in zip(seeds, names):
            out.append(f"ID: {sid}, Name: {nm}")
        return "\n".join(out)
    
    def _build_context_info(self, context: Dict[str, Any]) -> str:
        parts = []
        if context.get("times"):
            parts.append(f"Known Times: {context['times']}")
        # if context.get("answers"):
        #     parts.append(f"Existing Answers: {context['answers']}")
        # if context.get("entities"):
        #     parts.append(f"Related Entities: {context['entities']}")
        return "\n".join(parts) if parts else "No context information"

    def _get_indicator_info(self, context_info: str) -> str:
        """extract indicator information from context information"""
        # find possible indicator related content
        lines = context_info.split('\n') if context_info else []
        indicator_parts = []
        
        for line in lines:
            # find possible indicator related information
            if 'indicator' in line.lower() or 'related entities' in line.lower():
                indicator_parts.append(line)
        
        if indicator_parts:
            return "\n".join(indicator_parts)
        else:
            return "No specific indicator information detected"

    def _build_specialized_prompt(self, subquestion: str, seed_info: str, 
                                 context_info: str, qtype: str, 
                                 time_hints: Dict[str, Any], entity_names: List[str]) -> str:
        """build specialized prompt based on question type"""
        # import specialized prompt function
        try:
            from .toolkit_prompts_v2 import get_prompt_for_question_type as get_prompt_v2
            
            # USE VERSION 2 implemented new toolkit views that LOADS examples
            specialized_prompt = get_prompt_v2(
                qtype=qtype,
                subquestion=subquestion,
                seed_info=seed_info,
                context_info=context_info,
                time_hints=time_hints,
                entity_names=entity_names
            )
            return specialized_prompt
            
        except ImportError:
            print("[Selector] Failed to import toolkit_prompts_v2, using fallback prompt")
            
        except Exception as e:
            print(f"[Selector] Error building specialized prompt: {e}")
        
        # Fallback: use general prompt
        return self._build_general_fallback_prompt(
            subquestion, seed_info, context_info, qtype, time_hints, entity_names
        )

    def _build_general_fallback_prompt(self, subquestion: str, seed_info: str, 
                                     context_info: str, qtype: str, 
                                     time_hints: Dict[str, Any], entity_names: List[str]) -> str:
        """general fallback prompt (when specialized prompt is not available)"""
        time_info = "\n".join([
            f"{k.capitalize()}: {v}" for k, v in time_hints.items() if v
        ]) if time_hints else "No time constraints"
        
        return f"""You are an expert toolkit selector. Select the most appropriate toolkit based on question type and context.

## Subquestion: {subquestion}
## Question Type: {qtype}
## Seed Entities: {seed_info}
## Context: {context_info}
## Time Constraints: {time_info}

## Select appropriate toolkit:
Choose from: OneHop, AfterFirst, BeforeLast, BetweenRange, DayEvents, MonthEvents, YearEvents, DirectConnection

JSON Output:
{{
    "selected_toolkits": [
        {{
      "original_name": "ToolkitName",
      "parameters": {{}},
      "reasoning": "Explanation",
            "priority": 1
    }}
  ]
}}"""
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        try:
            js = json.loads(response)
            return js.get("selected_toolkits", [])
        except Exception:
            # extract JSON fragment
            m = re.search(r'\{.*\}', response, flags=re.S)
            if m:
                try:
                    js = json.loads(m.group(0))
                    return js.get("selected_toolkits", [])
                except Exception:
                    pass
            print(f"[Selector] LLM response parse failed (truncated): {response[:200]}...")
            return []

    # ============================
    # tool functions
    # ============================
    def _seed_ids_to_names(self, seeds: List[int]) -> List[str]:
        names: List[str] = []
        if not seeds:
            return names
        if KG_AVAILABLE:
            for sid in seeds:
                try:
                    nm = KG.get_entity_name(sid)
                except Exception:
                    nm = None
                if nm:
                    names.append(nm)
                else:
                    # if cannot get name, use ID as placeholder, keep list length consistent
                    names.append(f"Entity_{sid}")
                    print(f"⚠️ cannot get entity name, ID: {sid}, use placeholder")
        else:
            # if KG is not available, use ID as placeholder for all seeds
            names = [f"Entity_{sid}" for sid in seeds]
        return names
