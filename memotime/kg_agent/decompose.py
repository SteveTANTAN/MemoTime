
# =============================
# file: kg_agent/decompose.py
# =============================
import re
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, NamedTuple, Optional
from .prompts import (
    LLM_SYSTEM_PROMPT, 
    LLM_FURTHER_QUESTION_PROMPT,
    LLM_TEMPORAL_CLARIFICATION_PROMPT,
    LLM_MULTI_HOP_CLARIFICATION_PROMPT,
    LLM_SEED_SELECT_PROMPT
)
from .fixed_prompts import (
    LLM_DECOMP_AFTER_FIRST_PROMPT,
    LLM_DECOMP_BEFORE_LAST_PROMPT,
    LLM_DECOMP_FIRST_LAST_PROMPT,
    LLM_DECOMP_BEFORE_AFTER_PROMPT,
    LLM_DECOMP_EQUAL_PROMPT,
    LLM_DECOMP_EQUAL_MULTI_PROMPT,
    LLM_DECOMP_DEFAULT_PROMPT
)
from .llm import LLM

@dataclass
class IndicatorEdge:
    subj: str
    rel: str
    obj: str
    time_var: str

@dataclass
class Indicator:
    edges: List[IndicatorEdge]
    constraints: List[str]

@dataclass
class SubQuestion:
    sid: str
    text: str
    indicator: Indicator
    depends_on: List[str]

class DecompositionResult(NamedTuple):
    subquestions: List[SubQuestion]
    time_vars: List[str]
    question_type: str
    further_question: Optional[str] = None
    needs_clarification: bool = False

# ---- utils ----

def _norm_name(s: str) -> str:
    return s.replace("_", " ").strip()
def extract_question_type_from_LLM_respones(res: str) -> str:
    # print(res)
    # Try to extract the question_type from the LLM response string
    import json
    try:
        if isinstance(res, dict):
            return res.get("question_type", "None")
        # Try to parse as JSON
        result = json.loads(res)
        if isinstance(result, dict):
            return result.get("question_type", "None")
        # If result is a list, try first element
        if isinstance(result, list) and result:
            return result[0].get("question_type", "None")
    except Exception:
        pass
    # Fallback: try to extract with a regex
    import re
    match = re.search(r'"question_type"\s*:\s*"([^"]+)"', res)
    if match:
        return match.group(1)
    match = re.search(r"'question_type'\s*:\s*'([^']+)'", res)
    if match:
        return match.group(1)
    # Fallback: return None
    return "None"
    # exit()
    # return res.get("question_type", "None")
# ---- question type detection ----
def detect_question_type_with_template(question: str) -> str:
    all_question_types = ["equal", "before", "after", "during", "between", "first", "last", "count", "comparison"]
    try:
        classification_prompt = """
Classify the following question into one of the supported question types.

Supported question types:
{all_question_types}

Question:
{question}

Return JSON:
{{
    "question_type": "question_type"
}}
        """
        classification_prompt = classification_prompt.format(all_question_types=all_question_types, question=question)
        enhanced_prompt = classification_prompt
        print("✅ Classification prompt enhanced successfully")
        response = LLM.call(LLM_SYSTEM_PROMPT, enhanced_prompt)
        return extract_question_type_from_LLM_respones(response)
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return "None"



    return response

def detect_question_type(question: str) -> str:
    question_lower = question.lower()

    print("rule-based classification")
    if "after" in question_lower and "first" in question_lower.lower():
            return "after_first"
    
    if "before" in question_lower and "last" in question_lower:
        return "before_last"

    if "first" in question_lower or "last" in question_lower:
        return "first_last"

    if "after" in question_lower or "before" in question_lower:
        return "before_after"

    if re.search(r"(?:same\s+(?:month|day|year)|in\s+the\s+same)", question_lower):
        return "equal_multi"
    
    if re.search(r"(?:who\s+.+\s+in\s+|which\s+.+\s+in\s+|what\s+happened\s+in\s+)", question_lower):
        return "equal"

    LLM_result = detect_question_type_with_template(question)

    valid_types = ["equal", "before", "after", "during", "between", "first", "last", "count", "comparison"]
    if LLM_result in valid_types:
        print(f"LLM-based classification: {LLM_result}")
        return LLM_result
    else:
        return "None"

 


def get_decomposition_prompt(question_type: str) -> str:
    prompt_map = {
        "after_first": LLM_DECOMP_AFTER_FIRST_PROMPT,
        "before_last": LLM_DECOMP_BEFORE_LAST_PROMPT,
        "first_last": LLM_DECOMP_FIRST_LAST_PROMPT,
        "before_after": LLM_DECOMP_BEFORE_AFTER_PROMPT,
        "equal": LLM_DECOMP_EQUAL_PROMPT,
        "equal_multi": LLM_DECOMP_EQUAL_MULTI_PROMPT,
        "default": LLM_DECOMP_DEFAULT_PROMPT
    }
    return prompt_map.get(question_type, LLM_DECOMP_DEFAULT_PROMPT)

# ---- analysis and clarification ----

def analyze_decomposition_quality(result: DecompositionResult, original_question: str, question_type: str) -> Dict[str, Any]:
    analysis = {
        "has_temporal_info": len(result.time_vars) > 0,
        "has_dependencies": any(subq.depends_on for subq in result.subquestions),
        "has_constraints": any(subq.indicator.constraints for subq in result.subquestions),
        "subquestion_count": len(result.subquestions),
        "expected_subquestion_count": get_expected_subquestion_count(question_type),
        "missing_temporal_info": [],
        "missing_reasoning_steps": [],
        "needs_clarification": False
    }
    
    if not analysis["has_temporal_info"]:
        analysis["missing_temporal_info"].append("No time variables found")
        analysis["needs_clarification"] = True
    
    if question_type in ["after_first", "before_last", "equal_multi"] and analysis["subquestion_count"] < 2:
        analysis["missing_reasoning_steps"].append("Missing intermediate reasoning steps")
        analysis["needs_clarification"] = True
    
    if not analysis["has_constraints"] and question_type in ["first_last", "equal"]:
        analysis["missing_temporal_info"].append("Missing temporal constraints")
        analysis["needs_clarification"] = True
    
    return analysis

def get_expected_subquestion_count(question_type: str) -> int:
    expected_counts = {
        "after_first": 2,
        "before_last": 2,
        "first_last": 1,
        "equal": 1,
        "equal_multi": 2,
        "default": 1
    }
    return expected_counts.get(question_type, 1)

def generate_further_question(result: DecompositionResult, original_question: str, question_type: str) -> Optional[str]:
    analysis = analyze_decomposition_quality(result, original_question, question_type)
    
    if not analysis["needs_clarification"]:
        return None
    
    decomposition_desc = []
    for i, subq in enumerate(result.subquestions, 1):
        decomposition_desc.append(f"Subquestion {i}: {subq.text}")
        if subq.indicator.edges:
            edges_desc = [f"{e.subj} --[{e.rel}]--> {e.obj}" for e in subq.indicator.edges]
            decomposition_desc.append(f"  Edges: {', '.join(edges_desc)}")
        if subq.indicator.constraints:
            decomposition_desc.append(f"  Constraints: {', '.join(subq.indicator.constraints)}")
    
    current_decomposition = "\n".join(decomposition_desc)
    
    missing_info = []
    if analysis["missing_temporal_info"]:
        missing_info.extend(analysis["missing_temporal_info"])
    if analysis["missing_reasoning_steps"]:
        missing_info.extend(analysis["missing_reasoning_steps"])
    
    analysis_text = f"Missing: {', '.join(missing_info)}"
    
    try:
        prompt = LLM_FURTHER_QUESTION_PROMPT.format(
            original_question=original_question,
            current_decomposition=current_decomposition,
            analysis=analysis_text
        )
        
        response = LLM.call(LLM_SYSTEM_PROMPT, prompt)
        result_obj = json.loads(response)
        
        return result_obj.get("further_question", None)
        
    except Exception as e:
        return None

def generate_temporal_clarification(original_question: str, result: DecompositionResult) -> Optional[str]:
    subquestions_desc = []
    for i, subq in enumerate(result.subquestions, 1):
        subquestions_desc.append(f"{i}. {subq.text}")
    
    subquestions_text = "\n".join(subquestions_desc)
    
    missing_temporal = []
    if not result.time_vars:
        missing_temporal.append("No time variables identified")
    if not any(subq.indicator.constraints for subq in result.subquestions):
        missing_temporal.append("No temporal constraints specified")
    
    if not missing_temporal:
        return None
    
    missing_temporal_text = "; ".join(missing_temporal)
    
    try:
        prompt = LLM_TEMPORAL_CLARIFICATION_PROMPT.format(
            original_question=original_question,
            subquestions=subquestions_text,
            missing_temporal_info=missing_temporal_text
        )
        
        response = LLM.call(LLM_SYSTEM_PROMPT, prompt)
        result_obj = json.loads(response)
        
        return result_obj.get("clarification_question", None)
        
    except Exception as e:
        return None

def generate_multi_hop_clarification(original_question: str, result: DecompositionResult, question_type: str) -> Optional[str]:
    expected_count = get_expected_subquestion_count(question_type)
    
    if len(result.subquestions) >= expected_count:
        return None
    
    subquestions_desc = []
    for i, subq in enumerate(result.subquestions, 1):
        subquestions_desc.append(f"{i}. {subq.text}")
    
    subquestions_text = "\n".join(subquestions_desc)
    
    missing_steps = []
    if question_type in ["after_first", "before_last"] and len(result.subquestions) < 2:
        missing_steps.append("Missing reference time identification step")
    if question_type == "equal_multi" and len(result.subquestions) < 2:
        missing_steps.append("Missing reference entity time lookup step")
    
    if not missing_steps:
        return None
    
    missing_steps_text = "; ".join(missing_steps)
    
    try:
        prompt = LLM_MULTI_HOP_CLARIFICATION_PROMPT.format(
            original_question=original_question,
            subquestions=subquestions_text,
            missing_reasoning_steps=missing_steps_text
        )
        
        response = LLM.call(LLM_SYSTEM_PROMPT, prompt)
        result_obj = json.loads(response)
        
        return result_obj.get("clarification_question", None)
        
    except Exception as e:
        return None

# ---- text parsing ----

def _parse_list_content(content: str) -> List[str]:
    if not content:
        return []
    
    if content.startswith("[") and content.endswith("]"):
        content = content[1:-1]
    
    items = []
    current_item = ""
    in_quotes = False
    quote_char = None
    
    for char in content:
        if char in ['"', "'"] and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
        elif char == ',' and not in_quotes:
            if current_item.strip():
                items.append(current_item.strip().strip('"\''))
            current_item = ""
            continue
        current_item += char
    
    if current_item.strip():
        items.append(current_item.strip().strip('"\''))
    
    return items

def _fallback_parse_questions(response: str) -> Optional[Dict[str, Any]]:
    try:
        questions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(('Subquestions:', 'Indicators:', 'Constraints:', 'Time_vars:')):
                continue
            
            clean_line = re.sub(r'^\d+\.\s*', '', line)
            clean_line = clean_line.strip('"\'[]').strip()
            
            if '?' in clean_line and len(clean_line) > 10:
                questions.append(clean_line)
        
        if questions:
            return {
                "subquestions": questions,
                "indicators": [f"indicator_{i+1}" for i in range(len(questions))],
                "constraints": [],
                "time_vars": []
            }
        
        return None
        
    except Exception as e:
        print(f"Fallback解析失败: {e}")
        return None

def _convert_json_to_simple_format(json_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        result = {
            "subquestions": [],
            "indicators": [],
            "constraints": [],
            "time_vars": []
        }
        
        subquestions_key = None
        for key in ["subquestions", "Subquestions", "questions", "Questions"]:
            if key in json_data:
                subquestions_key = key
                break
        
        if subquestions_key:
            subquestions = json_data[subquestions_key]
            if isinstance(subquestions, list):
                for subq in subquestions:
                    if isinstance(subq, str):
                        result["subquestions"].append(subq)
                    elif isinstance(subq, dict) and "text" in subq:
                        result["subquestions"].append(subq["text"])
        
        indicators_key = None
        for key in ["indicators", "Indicators", "edges", "Edges"]:
            if key in json_data:
                indicators_key = key
                break
        
        if indicators_key:
            indicators = json_data[indicators_key]
            if isinstance(indicators, list):
                result["indicators"] = [str(ind) for ind in indicators]
        
        constraints_key = None
        for key in ["constraints", "Constraints", "conditions", "Conditions"]:
            if key in json_data:
                constraints_key = key
                break
        
        if constraints_key:
            constraints = json_data[constraints_key]
            if isinstance(constraints, list):
                result["constraints"] = [str(con) for con in constraints]
        
        time_vars_key = None
        for key in ["time_vars", "Time_vars", "time_variables", "Time_variables", "times", "Times"]:
            if key in json_data:
                time_vars_key = key
                break
        
        if time_vars_key:
            time_vars = json_data[time_vars_key]
            if isinstance(time_vars, list):
                result["time_vars"] = [str(tv) for tv in time_vars]
        
        if result["subquestions"]:
            return result
        
        return None
        
    except Exception as e:
        return None

def parse_simple_text_response(response: str) -> Optional[Dict[str, Any]]:
    try:
        lines = response.strip().split('\n')
        result = {
            "subquestions": [],
            "indicators": [],
            "constraints": [],
            "time_vars": []
        }
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Subquestions:"):
                content = line.replace("Subquestions:", "").strip()
                questions = _parse_list_content(content)
                if questions:
                    result["subquestions"] = questions
                    
            elif line == "Subquestions:":
                questions = []
                for next_line in lines[lines.index(line) + 1:]:
                    next_line = next_line.strip()
                    if next_line and not next_line.startswith(("Indicators:", "Constraints:", "Time_vars:")):
                        clean_question = next_line.strip('"\'[]').strip()
                        if clean_question:
                            questions.append(clean_question)
                    elif next_line.startswith(("Indicators:", "Constraints:", "Time_vars:")):
                        break
                if questions:
                    result["subquestions"] = questions
                    
            elif re.match(r'^\d+\.', line):
                questions = []
                for l in lines:
                    l = l.strip()
                    if re.match(r'^\d+\.', l):
                        question = re.sub(r'^\d+\.\s*', '', l).strip().strip('"\'')
                        if question:
                            questions.append(question)
                if questions:
                    result["subquestions"] = questions
            elif line.startswith("Indicators:"):
                content = line.replace("Indicators:", "").strip()
                indicators = _parse_list_content(content)
                if indicators:
                    result["indicators"] = indicators
            elif line.startswith("Constraints:"):
                content = line.replace("Constraints:", "").strip()
                constraints = _parse_list_content(content)
                if constraints:
                    result["constraints"] = constraints
            elif line.startswith("Time_vars:"):
                content = line.replace("Time_vars:", "").strip()
                time_vars = _parse_list_content(content)
                if time_vars:
                    result["time_vars"] = time_vars
        
        if result["subquestions"]:
            if not result["indicators"]:
                result["indicators"] = [f"indicator_{i+1}" for i in range(len(result["subquestions"]))]
            return result
        else:
            return _fallback_parse_questions(response)
            
    except Exception as e:
        return None

def convert_simple_to_structured(simple_result: Dict[str, Any], question_type: str = "default") -> DecompositionResult:
    subquestions = []
    
    for i, subq_text in enumerate(simple_result["subquestions"], 1):
        edges = []
        if i <= len(simple_result["indicators"]):
            indicator_text = simple_result["indicators"][i-1]
            # format: "A --[rel]--> B (t1)"
            edge_match = re.match(r"(.+?)\s*--\[(.+?)\]\s*-->\s*(.+?)\s*\((.+?)\)", indicator_text)
            if edge_match:
                subj, rel, obj, time_var = edge_match.groups()
                edges.append(IndicatorEdge(
                    subj=subj.strip(),
                    rel=rel.strip(),
                    obj=obj.strip(),
                    time_var=time_var.strip()
                ))
        
        # create constraints
        constraints = []
        if i <= len(simple_result["constraints"]):
            constraints = [c.strip() for c in simple_result["constraints"] if c.strip()]
        
        # create subquestion
        indicator = Indicator(edges=edges, constraints=constraints)
        depends_on = []
        
        # simple dependency detection
        if "t1" in subq_text and i > 1:
            depends_on = ["t1"]
        
        subquestions.append(SubQuestion(
            sid=f"s{i}",
            text=subq_text,
            indicator=indicator,
            depends_on=depends_on
        ))
    
    return DecompositionResult(
        subquestions=subquestions,
        time_vars=simple_result["time_vars"],
        question_type=question_type
    )

# ---- decomposition ----

def fallback_decompose_after_first_visit(question: str) -> Optional[DecompositionResult]:
    rx = re.compile(r"after\s+the\s+(?P<anchor>.+?),\s*who\s+was\s+the\s+first\s+to\s+visit\s+(?P<place>[^?]+)", re.I)
    m = rx.search(question)
    if not m:
        return None
    anchor = m.group("anchor").strip()
    place = m.group("place").strip()

    s1 = SubQuestion(
        sid="s1",
        text=f"When did {anchor} visit {place}?",
        indicator=Indicator(edges=[IndicatorEdge(subj=anchor, rel="visit", obj=place, time_var="t1")], constraints=[]),
        depends_on=[],
    )
    s2 = SubQuestion(
        sid="s2",
        text=f"After t1, who was the first to visit {place}?",
        indicator=Indicator(edges=[
            IndicatorEdge(subj=anchor, rel="visit", obj=place, time_var="t1"),
            IndicatorEdge(subj="?x", rel="visit", obj=place, time_var="t2"),
        ], constraints=["t2 >= t1", "first_after(t2, t1)"]),
        depends_on=["t1"],
    )
    return DecompositionResult(subquestions=[s1, s2], time_vars=["t1", "t2"], question_type="after_first")


def fallback_decompose_after_first_negotiate(question: str) -> Optional[DecompositionResult]:
    # format: "after X, who was the first to negotiate with Y"
    rx = re.compile(r"after\s+the\s+(?P<anchor>.+?),\s*who\s+was\s+the\s+first\s+to\s+(?:express\s+the\s+intention\s+to\s+)?negotiate\s+with\s+(?P<target>[^?]+)", re.I)
    m = rx.search(question)
    if not m:
        return None
    anchor = m.group("anchor").strip()
    target = m.group("target").strip()

    s1 = SubQuestion(
        sid="s1",
        text=f"When did {anchor} express the intention to negotiate with {target}?",
        indicator=Indicator(edges=[IndicatorEdge(subj=anchor, rel="express_intention_to_negotiate", obj=target, time_var="t1")], constraints=[]),
        depends_on=[],
    )
    s2 = SubQuestion(
        sid="s2",
        text=f"After t1, who was the first to express the intention to negotiate with {target}?",
        indicator=Indicator(edges=[
            IndicatorEdge(subj=anchor, rel="express_intention_to_negotiate", obj=target, time_var="t1"),
            IndicatorEdge(subj="?x", rel="express_intention_to_negotiate", obj=target, time_var="t2"),
        ], constraints=["t2 >= t1", "first_after(t2, t1)"]),
        depends_on=["t1"],
    )
    return DecompositionResult(subquestions=[s1, s2], time_vars=["t1", "t2"], question_type="after_first")


def fallback_decompose_temporal_after(question: str) -> Optional[DecompositionResult]:
    # format: "after X, who/what/when"
    # 匹配 "after X, who was the first to Y" 模式
    rx1 = re.compile(r"after\s+(?:the\s+)?(?P<anchor>.+?),\s*who\s+was\s+the\s+first\s+to\s+(?P<action>[^?]+)", re.I)
    m1 = rx1.search(question)
    if m1:
        anchor = m1.group("anchor").strip()
        action = m1.group("action").strip()
        
        # extract target object
        target_match = re.search(r"with\s+([^?]+)", action)
        target = target_match.group(1).strip() if target_match else "?target"
        

        relation = action.replace(" ", "_")
        
        s1 = SubQuestion(
            sid="s1",
            text=f"When did {anchor} {action}?",
            indicator=Indicator(edges=[IndicatorEdge(subj=anchor, rel=relation, obj=target, time_var="t1")], constraints=[]),
            depends_on=[],
        )
        s2 = SubQuestion(
            sid="s2",
            text=f"After t1, who was the first to {action}?",
            indicator=Indicator(edges=[
                IndicatorEdge(subj=anchor, rel=relation, obj=target, time_var="t1"),
                IndicatorEdge(subj="?x", rel=relation, obj=target, time_var="t2"),
            ], constraints=["t2 >= t1", "first_after(t2, t1)"]),
            depends_on=["t1"],
        )
        return DecompositionResult(subquestions=[s1, s2], time_vars=["t1", "t2"], question_type="after_first")
    
    # 匹配 "after X, what happened" 模式
    rx2 = re.compile(r"after\s+(?:the\s+)?(?P<anchor>.+?),\s*what\s+happened", re.I)
    m2 = rx2.search(question)
    if m2:
        anchor = m2.group("anchor").strip()
        
        s1 = SubQuestion(
            sid="s1",
            text=f"When did {anchor} occur?",
            indicator=Indicator(edges=[IndicatorEdge(subj=anchor, rel="occur", obj="?time", time_var="t1")], constraints=[]),
            depends_on=[],
        )
        s2 = SubQuestion(
            sid="s2",
            text=f"After t1, what happened?",
            indicator=Indicator(edges=[
                IndicatorEdge(subj=anchor, rel="occur", obj="?time", time_var="t1"),
                IndicatorEdge(subj="?x", rel="happen", obj="?y", time_var="t2"),
            ], constraints=["t2 >= t1"]),
            depends_on=["t1"],
        )
        return DecompositionResult(subquestions=[s1, s2], time_vars=["t1", "t2"], question_type="after_first")
    
    return None


def fallback_decompose_temporal_before(question: str) -> Optional[DecompositionResult]:
    # format: "before X, who/what/when"
    # match "before X, who was the last to Y"
    rx1 = re.compile(r"before\s+(?:the\s+)?(?P<anchor>.+?),\s*who\s+was\s+the\s+last\s+to\s+(?P<action>[^?]+)", re.I)
    m1 = rx1.search(question)
    if m1:
        anchor = m1.group("anchor").strip()
        action = m1.group("action").strip()
        
        # extract target object
        target_match = re.search(r"with\s+([^?]+)", action)
        target = target_match.group(1).strip() if target_match else "?target"
        
        # standardize relation
        relation = action.replace(" ", "_")
        
        s1 = SubQuestion(
            sid="s1",
            text=f"When did {anchor} {action}?",
            indicator=Indicator(edges=[IndicatorEdge(subj=anchor, rel=relation, obj=target, time_var="t1")], constraints=[]),
            depends_on=[],
        )
        s2 = SubQuestion(
            sid="s2",
            text=f"Before t1, who was the last to {action}?",
            indicator=Indicator(edges=[
                IndicatorEdge(subj=anchor, rel=relation, obj=target, time_var="t1"),
                IndicatorEdge(subj="?x", rel=relation, obj=target, time_var="t2"),
            ], constraints=["t2 <= t1", "last_before(t2, t1)"]),
            depends_on=["t1"],
        )
        return DecompositionResult(subquestions=[s1, s2], time_vars=["t1", "t2"], question_type="before_last")
    
    return None

   
def TL_enhandced_decompose_question(question: str, question_type: str) -> str:
    # use template enhanced decomposition question
    try:
        # first try to get examples from enhanced unified knowledge storage
        try:
            from .enhanced_unified_integration import get_question_decomposition_enhanced
            # try to get global experiment_setting
            try:
                from .stepwise import CURRENT_EXPERIMENT_SETTING
                experiment_setting = CURRENT_EXPERIMENT_SETTING
            except:
                experiment_setting = None
            
            examples = get_question_decomposition_enhanced(
                given_question=question,
                topk=10,
                question_type=question_type,
                experiment_setting=experiment_setting,
                similarity_threshold=0.1
            )
            
            if examples:
                print(f"✅ get {len(examples)} examples from unified knowledge storage")
                # build enhanced prompt
                enhanced_prompt = "## Successful Examples for Question Decomposition:\n\n"
                for i, example in enumerate(examples, 1):
                    print(f"Example {i}: {example}")
                    enhanced_prompt += f"Example {i}:\n"
                    enhanced_prompt += f"Question: {example['Q']}\n"
                    enhanced_prompt += f"Subquestions: {example['Subquestions']}\n"
                    enhanced_prompt += f"Indicators: {example['Indicators']}\n"
                    enhanced_prompt += f"Constraints: {example['Constraints']}\n"
                    enhanced_prompt += f"Time_vars: {example['Time_vars']}\n\n"
                
                return enhanced_prompt
            else:
                print("📋 no similar decomposition examples found, use template learning")
        except Exception as e:
            print(f"⚠️ unified knowledge storage query failed: {e}, fallback to template learning")
        
        # fallback to original template learning
        classification_prompt = ""
        # template learning removed, use basic prompt
        enhanced_prompt = classification_prompt
        print("✅ Decomposition prompt enhanced with template learning")
        return enhanced_prompt
        
    except Exception as e:
        print(f"❌ decomposition enhanced failed: {e}")
        return ""
        # # if "## Successful Examples" in enhanced_prompt:
        #     print("✅ Examples found in enhanced prompt")
        # else:
        #     print("⚠️ No examples found in enhanced prompt")
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return ""

def decompose_question(question: str, max_retries: int = 3) -> DecompositionResult:
    # detect question type
    question_type = detect_question_type(question)
    print(f"detected question type: {question_type}")
    
    # get corresponding decomposition prompt
    # TL_enhandced_decompose_question(question, question_type)
    # exit()

    prompt_TL = TL_enhandced_decompose_question(question, question_type)
    # print(prompt_TL)
    prompt_TL = str(prompt_TL)
    if not prompt_TL:
        prompt_TL = ""
    prompt = get_decomposition_prompt(question_type)
    prompt = prompt.format(enhanced_prompt=prompt_TL, question=question) 



    print(f"use {question_type} type decomposition prompt")
    
    # retry mechanism
    for attempt in range(max_retries):
        print(f"attempt {attempt + 1}/{max_retries}")
        
        try:
            # use type specific prompt for decomposition
            # print(prompt)
            from config import TPKGConfig
            decompostion_model = TPKGConfig.DECOMPOSITION_LLM_MODEL
            response = LLM.call(LLM_SYSTEM_PROMPT, prompt, model=decompostion_model)
            print(f"LLM response: {response[:200]}...")
            # exit()
            # first try JSON parsing
            json_result = None
            try:
                json_data = json.loads(response)
                if isinstance(json_data, dict) and ("subquestions" in json_data or "Subquestions" in json_data):
                    print("successfully parsed JSON format")
                    json_result = json_data
                else:
                    print("JSON format does not contain subquestions field, try pure text parsing")
            except json.JSONDecodeError:
                print("JSON parsing failed, try pure text parsing")
            except Exception as e:
                print(f"JSON parsing exception: {e}, try pure text parsing")
            
            # if JSON parsing successful, use JSON result
            if json_result:
                # convert JSON result to simple format
                simple_result = _convert_json_to_simple_format(json_result)
                if not simple_result:
                    # if JSON conversion failed, use text parsing as fallback
                    simple_result = parse_simple_text_response(response)
            else:
                # parse simple text format
                simple_result = parse_simple_text_response(response)
            
            if simple_result:
                print("successfully parsed text format")
                # 转换为结构化结果
                result = convert_simple_to_structured(simple_result, question_type)
                
                # analyze decomposition quality and generate further question
                analysis = analyze_decomposition_quality(result, question, question_type)
                print(f"decomposition quality analysis: {analysis}")
                
                if analysis["needs_clarification"]:
                    print("detected need clarification, generate further question...")
                    
                    # generate further question
                    further_question = generate_further_question(result, question, question_type)
                    
                    if further_question:
                        print(f"generated further question: {further_question}")
                        return DecompositionResult(
                            subquestions=result.subquestions, 
                            time_vars=result.time_vars,
                            question_type=question_type,
                            further_question=further_question,
                            needs_clarification=True
                        )
                    else:
                        print("cannot generate further question, use current decomposition result")
                
                return result
            else:
                print(f"parsing failed, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    print("retrying...")
                    continue
                else:
                    print("all retries failed")
                    break
                    
        except Exception as e:
            print(f"LLM decomposition failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print("retrying...")
                continue
            else:
                print("all retries failed")
                break
    
    # final fallback: use the whole question as a subquestion
    print("use fallback mode")
    s = SubQuestion(sid="s1", text=question, indicator=Indicator(edges=[], constraints=[]), depends_on=[])
    result = DecompositionResult(subquestions=[s], time_vars=[], question_type=question_type)
    
    # analyze fallback result and generate clarification question
    analysis = analyze_decomposition_quality(result, question, question_type)
    if analysis["needs_clarification"]:
        further_question = generate_further_question(result, question, question_type)
        if further_question:
            print(f"generated further question after fallback: {further_question}")
            return DecompositionResult(
                subquestions=[s], 
                time_vars=[],
                question_type=question_type,
                further_question=further_question,
                needs_clarification=True
            )
    
    return result

# ---- seeds ----

def select_seeds_for_subq(subq: SubQuestion, topic_entities: List[Dict[str, Any]]) -> List[int]:
    """improved seed selection logic"""
    picks: List[int] = []
    
    # 1. extract entities from subquestion text
    subq_text = subq.text.lower()
    
    # 2. extract entities from indicators
    indicator_entities = set()
    for edge in subq.indicator.edges:
        if edge.subj and edge.subj != "?x" and edge.subj != "?y":
            indicator_entities.add(edge.subj.lower())
        if edge.obj and edge.obj != "?x" and edge.obj != "?y":
            indicator_entities.add(edge.obj.lower())
    
    # 3. select seeds based on entity name matching
    for entity in topic_entities:
        entity_name = _norm_name(entity.get("name", "")).lower()
        entity_id = int(entity.get("id", 0))
        
        # check if in indicators
        if entity_name in indicator_entities:
            picks.append(entity_id)
            continue
            
        # check if in subquestion text
        if entity_name in subq_text:
            picks.append(entity_id)
            continue
            
        # check partial matching (for handling variants)
        for indicator_entity in indicator_entities:
            if (entity_name in indicator_entity or 
                indicator_entity in entity_name or
                any(word in entity_name for word in indicator_entity.split() if len(word) > 3)):
                picks.append(entity_id)
                break
    
    # 4. if seeds found, return deduplicated result
    if picks:
        return list(dict.fromkeys(picks))
    
    # 5. if not found, use LLM to select
    print(f"no matching seeds found, use LLM to select")
    ents = json.dumps(topic_entities, ensure_ascii=False, indent=2)
    js = LLM.call(LLM_SYSTEM_PROMPT, LLM_SEED_SELECT_PROMPT.format(subq=subq.text, entities=ents))
    try:
        obj = json.loads(js)
        return [int(x) for x in obj.get("seed_ids", [])]
    except Exception as e:
        print(f"LLM seed selection failed: {e}")
        return []
