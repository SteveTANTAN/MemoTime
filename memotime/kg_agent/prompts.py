
# =============================
# file: kg_agent/prompts.py
# =============================
LLM_SYSTEM_PROMPT = (
"You are a Temporal-KG question analyzer. You MUST operate only on KG evidence. "
"Do NOT suggest or rely on documents or web snippets."
)

LLM_MATCH_PROMPT = (
"Given the question below and a list of available workflow IDs with short intents, "
"decide which template best fits. If none fit, answer 'NONE'.\n\n"
"Question: {question}\n\nAvailable workflows:\n{workflows}\n\n"
"Rules: Choose EXACTLY ONE ID if a fit exists; otherwise 'NONE'. Only return the ID or 'NONE'."
)

LLM_FOLLOW_TEMPLATE_PROMPT = (
"You are to FOLLOW the selected workflow template (YAML-like spec) to analyze the question.\n"
"Constraints: Operate KG-only; respect time_level; apply core_steps in order; return output matching 'output_schema'.\n\n"
"Workflow card:\n{card}\n\nQuestion:\n{question}\n\nReturn JSON only with keys: items, explanations, verification."
)

LLM_GENERATE_TEMPLATE_PROMPT = (
"No existing template fits. Propose a NEW workflow card for this temporal-KG question.\n"
"Constraints: KG-only; include fields: workflow_id, version, status, intent, inputs, evidence, core_steps, verify, output_schema, heuristics, fallbacks.\n"
"The workflow_id must be a concise snake_case verb_noun phrase (e.g., 'first_after_anchor').\n"
"Provide a one-sentence example question for the index.\n\nQuestion:\n{question}\n\nReturn JSON with keys: workflow_id, example_question, yaml. 'yaml' must be the full YAML card."
)

# Type-specific question decomposition prompts based on RTQA MultiTQ project

# AFTER_FIRST type decomposition prompt
LLM_DECOMP_AFTER_FIRST_PROMPT = (
"Decompose 'after_first' temporal questions into subquestions with indicators.\n"
"Pattern: After X, who was the first to Y?\n"
"Strategy: 1) Find reference time when X did Y, 2) Find first entity to do Y after that time\n\n"
"Examples:\n\n"
"Example 1:\n"
"Q: After the Cabinet Council of Ministers of Kazakhstan, who was the first to express the intention to negotiate with China?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"When did Cabinet Council of Ministers of Kazakhstan express the intention to negotiate with China?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"Cabinet Council of Ministers of Kazakhstan\", \"rel\": \"express_intention_to_negotiate\", \"obj\": \"China\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": []\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }},\n"
"    {{\n"
"      \"sid\": \"s2\",\n"
"      \"text\": \"After t1, who was the first to express the intention to negotiate with China?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [\n"
"          {{\"subj\": \"Cabinet Council of Ministers of Kazakhstan\", \"rel\": \"express_intention_to_negotiate\", \"obj\": \"China\", \"time_var\": \"t1\"}},\n"
"          {{\"subj\": \"?x\", \"rel\": \"express_intention_to_negotiate\", \"obj\": \"China\", \"time_var\": \"t2\"}}\n"
"        ],\n"
"        \"constraints\": [\"t2 > t1\", \"first_after(t2, t1)\"]\n"
"      }},\n"
"      \"depends_on\": [\"t1\"]\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\", \"t2\"]\n"
"}}\n\n"
"Example 2:\n"
"Q: After the Danish Ministry of Defence and Security, who was the first to visit Iraq?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"When did Danish Ministry of Defence and Security visit Iraq?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"Danish Ministry of Defence and Security\", \"rel\": \"visit\", \"obj\": \"Iraq\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": []\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }},\n"
"    {{\n"
"      \"sid\": \"s2\",\n"
"      \"text\": \"After t1, who was the first to visit Iraq?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [\n"
"          {{\"subj\": \"?x\", \"rel\": \"visit\", \"obj\": \"Iraq\", \"time_var\": \"t2\"}}\n"
"        ],\n"
"        \"constraints\": [\"t2 > t1\", \"first_after(t2, t1)\"]\n"
"      }},\n"
"      \"depends_on\": [\"t1\"]\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\", \"t2\"]\n"
"}}\n\n"
"Example 3:\n"
"Q: After Devlet Bahçeli, who was the first to want to negotiate with Iraq?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"When did Devlet Bahçeli want to negotiate with Iraq?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"Devlet Bahçeli\", \"rel\": \"want_to_negotiate\", \"obj\": \"Iraq\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": []\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }},\n"
"    {{\n"
"      \"sid\": \"s2\",\n"
"      \"text\": \"After t1, who was the first to want to negotiate with Iraq?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [\n"
"          {{\"subj\": \"?x\", \"rel\": \"want_to_negotiate\", \"obj\": \"Iraq\", \"time_var\": \"t2\"}}\n"
"        ],\n"
"        \"constraints\": [\"t2 > t1\", \"first_after(t2, t1)\"]\n"
"      }},\n"
"      \"depends_on\": [\"t1\"]\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\", \"t2\"]\n"
"}}\n\n"
"Rules: Use ?x for unknown entities; t1, t2 for time variables; t2 > t1; return valid JSON only.\n\n"
"Question: {question}\n"
"Return JSON with the same structure as examples above."
)

# BEFORE_LAST type decomposition prompt
LLM_DECOMP_BEFORE_LAST_PROMPT = (
"Decompose 'before_last' temporal questions into subquestions with indicators.\n"
"Pattern: Before X, which Y did Z last?\n"
"Strategy: 1) Find reference time when Z did X, 2) Find last entity that Z did Y before that time\n\n"
"Examples:\n\n"
"Example 1:\n"
"Q: Before the military of Taiwan, which country did China threaten last?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"When did China threaten the military of Taiwan?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"China\", \"rel\": \"threaten\", \"obj\": \"military of Taiwan\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": []\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }},\n"
"    {{\n"
"      \"sid\": \"s2\",\n"
"      \"text\": \"Before t1, which country did China threaten last?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [\n"
"          {{\"subj\": \"China\", \"rel\": \"threaten\", \"obj\": \"?x\", \"time_var\": \"t2\"}}\n"
"        ],\n"
"        \"constraints\": [\"t2 < t1\", \"last_before(t2, t1)\"]\n"
"      }},\n"
"      \"depends_on\": [\"t1\"]\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\", \"t2\"]\n"
"}}\n\n"
"Example 2:\n"
"Q: With whom did Catherine Ashton last wish to meet before Cambodia?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"When did Catherine Ashton wish to meet Cambodia?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"Catherine Ashton\", \"rel\": \"wish_to_meet\", \"obj\": \"Cambodia\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": []\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }},\n"
"    {{\n"
"      \"sid\": \"s2\",\n"
"      \"text\": \"Before t1, with whom did Catherine Ashton wish to meet last?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [\n"
"          {{\"subj\": \"Catherine Ashton\", \"rel\": \"wish_to_meet\", \"obj\": \"?x\", \"time_var\": \"t2\"}}\n"
"        ],\n"
"        \"constraints\": [\"t2 < t1\", \"last_before(t2, t1)\"]\n"
"      }},\n"
"      \"depends_on\": [\"t1\"]\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\", \"t2\"]\n"
"}}\n\n"
"Example 3:\n"
"Q: Before Thailand, who last wanted to negotiate with the Governor of Thailand?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"When did someone want to negotiate with the Governor of Thailand about Thailand?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"?y\", \"rel\": \"want_to_negotiate\", \"obj\": \"Governor of Thailand\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": []\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }},\n"
"    {{\n"
"      \"sid\": \"s2\",\n"
"      \"text\": \"Before t1, who last wanted to negotiate with the Governor of Thailand?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [\n"
"          {{\"subj\": \"?x\", \"rel\": \"want_to_negotiate\", \"obj\": \"Governor of Thailand\", \"time_var\": \"t2\"}}\n"
"        ],\n"
"        \"constraints\": [\"t2 < t1\", \"last_before(t2, t1)\"]\n"
"      }},\n"
"      \"depends_on\": [\"t1\"]\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\", \"t2\"]\n"
"}}\n\n"
"Rules: Use ?x, ?y for unknown entities; t1, t2 for time variables; t2 < t1; return valid JSON only.\n\n"
"Question: {question}\n"
"Return JSON with the same structure as examples above."
)

# FIRST_LAST type decomposition prompt
LLM_DECOMP_FIRST_LAST_PROMPT = (
"Decompose 'first_last' temporal questions into subquestions with indicators.\n"
"Pattern: When did X last/first Y? or Who was the first/last to Y?\n"
"Strategy: Direct temporal query with ordering constraint\n\n"
"Examples:\n\n"
"Example 1:\n"
"Q: When did Kitti Wasinondh last express an intention to negotiate with Thailand?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"When did Kitti Wasinondh last express an intention to negotiate with Thailand?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"Kitti Wasinondh\", \"rel\": \"express_intention_to_negotiate\", \"obj\": \"Thailand\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": [\"last(t1)\"]\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\"]\n"
"}}\n\n"
"Example 2:\n"
"Q: In which year did Taiwan's Ministry of National Defence and Security last make a request to China?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"When did Taiwan's Ministry of National Defence and Security last make a request to China?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"Taiwan's Ministry of National Defence and Security\", \"rel\": \"make_request_to\", \"obj\": \"China\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": [\"last(t1)\"]\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\"]\n"
"}}\n\n"
"Example 3:\n"
"Q: Who was the first country that Ethiopia expressed optimism about?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"Who was the first country that Ethiopia expressed optimism about?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"Ethiopia\", \"rel\": \"express_optimism_about\", \"obj\": \"?x\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": [\"first(t1)\"]\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\"]\n"
"}}\n\n"
"Rules: Use ?x for unknown entities; t1 for time variable; first(t1) or last(t1) constraints; return valid JSON only.\n\n"
"Question: {question}\n"
"Return JSON with the same structure as examples above."
)

# EQUAL type decomposition prompt
LLM_DECOMP_EQUAL_PROMPT = (
"Decompose 'equal' temporal questions into subquestions with indicators.\n"
"Pattern: Who X in Y? or What happened in Y?\n"
"Strategy: Direct query with time constraint\n\n"
"Examples:\n\n"
"Example 1:\n"
"Q: Who signed an agreement with China in April 2005?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"Who signed an agreement with China in April 2005?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"?x\", \"rel\": \"sign_agreement\", \"obj\": \"China\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": [\"t1 = 2005-04\"]\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\"]\n"
"}}\n\n"
"Example 2:\n"
"Q: Who visited Malaysia on 14 January 2007?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"Who visited Malaysia on 14 January 2007?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"?x\", \"rel\": \"visit\", \"obj\": \"Malaysia\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": [\"t1 = 2007-01-14\"]\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\"]\n"
"}}\n\n"
"Example 3:\n"
"Q: Which country condemned China in August 2013?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"Which country condemned China in August 2013?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"?x\", \"rel\": \"condemn\", \"obj\": \"China\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": [\"t1 = 2013-08\"]\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\"]\n"
"}}\n\n"
"Rules: Use ?x for unknown entities; t1 for time variable; t1 = specific_time; return valid JSON only.\n\n"
"Question: {question}\n"
"Return JSON with the same structure as examples above."
)

# EQUAL_MULTI type decomposition prompt
LLM_DECOMP_EQUAL_MULTI_PROMPT = (
"Decompose 'equal_multi' temporal questions into subquestions with indicators.\n"
"Pattern: Who X in the same Y as Z? or Who X in Y?\n"
"Strategy: 1) Find reference time when Z did X, 2) Find entities that did X at same time\n\n"
"Examples:\n\n"
"Example 1:\n"
"Q: Who visited China in the same month as Oleg Ostapenko?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"When did Oleg Ostapenko visit China?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"Oleg Ostapenko\", \"rel\": \"visit\", \"obj\": \"China\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": []\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }},\n"
"    {{\n"
"      \"sid\": \"s2\",\n"
"      \"text\": \"Who visited China in the same month as t1?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [\n"
"          {{\"subj\": \"?x\", \"rel\": \"visit\", \"obj\": \"China\", \"time_var\": \"t2\"}}\n"
"        ],\n"
"        \"constraints\": [\"same_month(t1, t2)\"]\n"
"      }},\n"
"      \"depends_on\": [\"t1\"]\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\", \"t2\"]\n"
"}}\n\n"
"Example 2:\n"
"Q: In 2012, which country was the first to express interest in cooperation with Cambodia?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"When did Cambodia receive expressions of interest in cooperation?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"?y\", \"rel\": \"express_interest_in_cooperation\", \"obj\": \"Cambodia\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": []\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }},\n"
"    {{\n"
"      \"sid\": \"s2\",\n"
"      \"text\": \"In 2012, which country was the first to express interest in cooperation with Cambodia?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [\n"
"          {{\"subj\": \"?x\", \"rel\": \"express_interest_in_cooperation\", \"obj\": \"Cambodia\", \"time_var\": \"t2\"}}\n"
"        ],\n"
"        \"constraints\": [\"t2 >= t1\", \"first_after(t2, t1)\", \"t1 = 2012\"]\n"
"      }},\n"
"      \"depends_on\": [\"t1\"]\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\", \"t2\"]\n"
"}}\n\n"
"Example 3:\n"
"Q: Who did Ethiopia use conventional military force against on the same day as the Hizbul Islam fighter?\n"
"JSON:\n"
"{{\n"
"  \"subquestions\": [\n"
"    {{\n"
"      \"sid\": \"s1\",\n"
"      \"text\": \"When did the Hizbul Islam fighter use conventional military force?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [{{\"subj\": \"Hizbul Islam fighter\", \"rel\": \"use_conventional_military_force\", \"obj\": \"?y\", \"time_var\": \"t1\"}}],\n"
"        \"constraints\": []\n"
"      }},\n"
"      \"depends_on\": []\n"
"    }},\n"
"    {{\n"
"      \"sid\": \"s2\",\n"
"      \"text\": \"Who did Ethiopia use conventional military force against on the same day as t1?\",\n"
"      \"indicator\": {{\n"
"        \"edges\": [\n"
"          {{\"subj\": \"Ethiopia\", \"rel\": \"use_conventional_military_force\", \"obj\": \"?x\", \"time_var\": \"t2\"}}\n"
"        ],\n"
"        \"constraints\": [\"same_day(t1, t2)\"]\n"
"      }},\n"
"      \"depends_on\": [\"t1\"]\n"
"    }}\n"
"  ],\n"
"  \"time_vars\": [\"t1\", \"t2\"]\n"
"}}\n\n"
"Rules: Use ?x, ?y for unknown entities; t1, t2 for time variables; same_month/same_day constraints; return valid JSON only.\n\n"
"Question: {question}\n"
"Return JSON with the same structure as examples above."
)

# Default decomposition prompt (used when type cannot be identified)
LLM_DECOMP_DEFAULT_PROMPT = (
"Decompose temporal questions into subquestions with indicators.\n"
"Use ?x, ?y for unknown entities; t1, t2 for time variables; return valid JSON only.\n\n"
"Question: {question}\n"
"Return JSON: {{\"subquestions\": [{{\"sid\": \"s1\", \"text\": \"...\", \"indicator\": {{\"edges\": [{{\"subj\": \"...\", \"rel\": \"...\", \"obj\": \"...\", \"time_var\": \"t1\"}}], \"constraints\": []}}, \"depends_on\": []}}], \"time_vars\": [\"t1\"]}}"
)

# Further question generation prompt
LLM_FURTHER_QUESTION_PROMPT = (
"Analyze the decomposed subquestions and identify missing information that needs to be clarified.\n"
"Focus on:\n"
"1. Missing temporal information (when, time constraints)\n"
"2. Incomplete multi-hop reasoning (missing intermediate steps)\n"
"3. Ambiguous entities or relationships\n"
"4. Unclear temporal relationships between events\n\n"
"Original question: {original_question}\n"
"Current decomposition: {current_decomposition}\n"
"Analysis: {analysis}\n\n"
"Generate a follow-up question to clarify the missing information.\n"
"The question should be specific and help complete the reasoning chain.\n\n"
"Return JSON: {{\"further_question\": \"...\", \"reasoning\": \"...\", \"missing_info\": \"...\"}}"
)

# Temporal information clarification prompt
LLM_TEMPORAL_CLARIFICATION_PROMPT = (
"The current question decomposition lacks sufficient temporal information.\n"
"Generate a clarifying question to extract the missing time-related details.\n\n"
"Original question: {original_question}\n"
"Current subquestions: {subquestions}\n"
"Missing temporal info: {missing_temporal_info}\n\n"
"Generate a question that asks for:\n"
"1. Specific time points or periods\n"
"2. Temporal relationships between events\n"
"3. Time constraints or conditions\n"
"4. Chronological ordering of events\n\n"
"Return JSON: {{\"clarification_question\": \"...\", \"expected_info\": \"...\"}}"
)

# Multi-hop reasoning clarification prompt
LLM_MULTI_HOP_CLARIFICATION_PROMPT = (
"The current question decomposition appears to be missing intermediate reasoning steps for multi-hop questions.\n"
"Generate a clarifying question to identify the missing links in the reasoning chain.\n\n"
"Original question: {original_question}\n"
"Current subquestions: {subquestions}\n"
"Missing reasoning steps: {missing_reasoning_steps}\n\n"
"Generate a question that asks for:\n"
"1. Intermediate entities or events\n"
"2. Connecting relationships\n"
"3. Bridge information between steps\n"
"4. Additional context or background\n\n"
"Return JSON: {{\"clarification_question\": \"...\", \"expected_info\": \"...\"}}"
)

LLM_SEED_SELECT_PROMPT = (
"Given a subquestion and a list of topic entities, choose 1-2 SEED entities that are most relevant to answer the subquestion.\n"
"Return format: [id1, id2]\n\n"

"""
{enhanced_prompt}

Exampl 1:
subquestion: When did Kitti Wasinondh last express an intention to negotiate with Thailand?
entities:
1. Kitti Wasinondh
2. Thailand
3. China
output: [1, 2]

Example 2:
subquestion: before 2020, which country expressed interest in cooperation with Cambodia?
entities:
1. Cambodia
2. China
3. Ethiopia
output: [1]

My question is:
"""
"Subquestion: {subq}\n"
"Entities:\n{entities}\n\n"
"IMPORTANT: Return only the entity IDs in the format [id1, id2]. Do not include any other text.\n"
"Example: [1, 3]"
)

LLM_TOOLKIT_SELECT_PROMPT = (
"You are a temporal knowledge graph toolkit expert. Given a subquestion and available toolkit methods, "
"select the most appropriate method and provide the correct parameters.\n\n"
"Available Toolkit Methods:\n"
"{toolkit_methods}\n\n"
"Subquestion: {subquestion}\n"
"Context: {context}\n"
"Seed Entities: {seeds}\n\n"
"Return format: [method_name, param1=value1, param2=value2, ...]\n\n"
"IMPORTANT: Return only the method name and parameters in the exact format shown above.\n"
"Example: [find_after_first, entity=123, reference_time=2010-07-13, limit=10]\n"
"Example: [find_temporal_sequence, entity=123, relation=visit, limit=5]\n"
"Example: [find_entities_after_time, time_point=2010-07-13, limit=20]"
)

LLM_PATH_SELECT_PROMPT = (
"You are a path selection expert. Given a subquestion and candidate paths, select the top 3 most relevant paths.\n"
"Each path has an ID (1, 2, 3, etc.), head entity, relation, tail entity, and time.\n"
"You MUST select exactly 3 paths that best answer the subquestion.\n"
"Return ONLY a JSON object with the selected path IDs.\n\n"
"Subquestion: {subquestion}\n\n"
"Candidate paths:\n{paths}\n\n"
"IMPORTANT: You must select exactly 3 paths. Return format: {{\"selected_paths\": [id1, id2, id3]}}\n"
"Example: {{\"selected_paths\": [1, 3, 5]}}\n"
"Do not return empty selections. Always select 3 paths."
)

LLM_SUFFICIENT_TEST_PROMPT = (
"Please evaluate if this answer is sufficient for the sub-question.\n\n"
"""
Note: the given path is already sorted by the time in first or last, so you don't need to consider the time.
you should consider the sematic information and reasonable. path may the the order, consider who is Active and who is Passive.
"""
"Sub-question: {subquestion}\n\n"
"Answer: {current_answer}\n\n"
"Debate Vote Result: {debate_vote_info}\n\n"
"Top 3 Candidate Paths: {top_paths}\n\n"
"Evidence paths: {retrieved_info}\n\n"
"Previous context: {previous_subquestions}\n\n"
"Check if the answer:\n"
"- Directly answers the sub-question\n"
"- Has complete information (entity, time, relationship)\n"
"- Is supported by the evidence paths and top candidate paths\n"
"- Fits logically with previous answers\n"
"- Is justified by the debate vote result\n\n"
"Return JSON: {{\"sufficient\": true/false, \"reason\": \"explanation\", \"suggestions\": [\"list of improvements if needed\"]}}"
)

LLM_FINAL_SUFFICIENT_TEST_PROMPT = (
"Please reason carefully step-by-step based on the temporal information and evidence from each step to answer the original question.\n\n"

"Please follow this reasoning format:\n"
"1. Analyze the temporal information and key facts from each step\n"
"2. Establish the logical sequence and relationships\n"
"3. Draw conclusions based on the evidence\n"
"4. Provide your final answer in this exact format based on the evidence and your own knowledge: 'So the answer is: <final concise answer>'\n"
"5. The final answer can be multiple entities or time points'\n\n"

"Return JSON: {{\"sufficient\": true/false, \"reasoning\": \"your step-by-step analysis\", \"final_answer\": \"answer in the format 'So the answer is: <answer>'\"}}"
"Examples of good reasoning:\n"

"""

Example 1:
Q:
Original question: after the death of Mahmoud Abbas, who was the first to visit Iraq?
Step-by-step evidence:
Step 1 subquestion: When did Mahmoud Abbas die?
Step 1 answer: Mahmoud Abbas died on 2025-05-14.
Step 2 subquestion: After 2025-05-14, who was the first to visit Iraq?
Step 2 answer: Iran, China, United_States visited Iraq on 2025-05-15.
A:
{{\"sufficient\": true, \"reasoning\": \" Mahmoud Abbas died on 2025-05-14.After 2025-05-14, Iran, China, United_States visited Iraq on 2025-05-15.So the answer is: Iran, China, United_States\", \"final_answer\": \"So the answer is: Iran, China, United_States\"}}

The Question is:
Q:
"""
"Original question: {original_question}\n\n"
"Step-by-step evidence:\n{subquestions_details}\n\n\n"
"Please Given Your answer. "
"A:"
)

LLM_REGENERATE_SUBQUESTION_PROMPT = (
"The current sub-question answer is insufficient. Generate a new sub-question to better address the gap.\n"
"Consider the context of previous successful sub-questions and their answers.\n\n"
"Original question: {original_question}\n"
"Current insufficient sub-question: {original_subquestion}\n"
"Current insufficient answer: {current_answer}\n"
"Gap analysis: {gap_analysis}\n"
"Context: {context}\n"
"Previous successful sub-questions and answers: {previous_subquestions}\n\n"
"Generate a new sub-question that:\n"
"1. Addresses the identified gap\n"
"2. Is more specific and focused\n"
"3. Has a clear temporal scope\n"
"4. Can be answered with KG data\n"
"5. Builds logically on the previous successful sub-questions and their answers\n"
"6. Maintains consistency with the overall question flow\n\n"
"Note: your are asked to rewrite the subquestion, Do not use the original question. \n"
"if you cannot rewrite the subquestion, return the original subquestion.\n"
"Return format (choose one):\n"
"Option 1 - JSON: {{\"new_subquestion\": \"...\", \"indicator\": {{...}}, \"reasoning\": \"...\"}}\n"
"Option 2 - Plain text: Just provide the new sub-question directly\n"
"If using plain text, just return the new sub-question without any additional formatting."
"""
example 1:
Q:
Original question: after the death of Mahmoud Abbas, who was the first to visit Iraq?
Current insufficient sub-question: When did Mahmoud Abbas die?
Current insufficient answer: Boris Johnson died on 2025-05-14.
Gap analysis: The answer is insufficient, because the time is not specific.
Context: The context is the temporal information of the original question.
Previous successful sub-questions and answers: The previous successful sub-questions and answers are the previous successful sub-questions and answers of the original question.
A:
{{\"new_subquestion\": \"When did Mahmoud Abbas die?\", \"indicator\": {{...}}, \"reasoning\": \"...\"}}
"""
)

LLM_REGENERATE_FINAL_QUESTION_PROMPT = (
"The final answer is insufficient. Generate a new question to better address the original intent.\n"
"Original question: {original_question}\n"
"Insufficient final answer: {final_answer}\n"
"Gap analysis: {gap_analysis}\n"
"Available information: {available_info}\n\n"
"Generate a new question that:\n"
"1. Addresses the identified gaps\n"
"2. Is more specific and answerable\n"
"3. Leverages the available information\n"
"4. Maintains the original intent\n\n"
"Return format (choose one):\n"
"Option 1 - JSON: {{\"new_question\": \"...\", \"reasoning\": \"...\"}}\n"
"Option 2 - Plain text: Just provide the new question directly\n"
"If using plain text, just return the new question without any additional formatting."
)

# Answer prompt templates based on RTQA MultiTQ project
LLM_FINAL_ANSWER_PROMPT = (
"You are an expert in temporal knowledge graph question answering. "
"Based on the reasoning path, provide a comprehensive and accurate final answer to the original question.\n\n"
"Answer Guidelines:\n"
"1. Be precise and factual based on the reasoning path\n"
"2. Include temporal context when relevant\n"
"3. Cite specific entities and time points from the reasoning path\n"
"4. If the reasoning path is insufficient, clearly state the limitations\n"
"5. Provide a clear, direct answer to the original question\n\n"
"Original question: {question}\n"
"Reasoning path: {reasoning_path}\n\n"
"Provide a comprehensive final answer:"
)

# Subquestion answer prompt based on RTQA MultiTQ project
LLM_SUBQUESTION_ANSWER_PROMPT = (
"You are an expert in temporal knowledge graph reasoning. "
"Based on the retrieved candidates and context, provide the best answer to the subquestion.\n\n"
"Answer Guidelines:\n"
"1. Select the most relevant candidate based on temporal and semantic relevance\n"
"2. Consider the temporal constraints and ordering requirements\n"
"3. Provide reasoning for your selection\n"
"4. If no suitable candidate exists, state this clearly\n\n"
"Subquestion: {subquestion}\n"
"Candidates: {candidates}\n"
"Context: {context}\n\n"
"Provide the best answer with reasoning:"
)

# Intelligent toolkit selection prompt
LLM_INTELLIGENT_TOOLKIT_SELECT_PROMPT = """
You are an intelligent toolkit selection expert.Please select the most appropriate toolkit for retrieval based on the given subquestion, seed entities, and context information.

## Subquestion
{subquestion}

## Seed Entities
{seeds_info}

## Context Information
{context_info}

## Question Type
{question_type}

## Available Toolkits
{toolkit_info}

## Selection Requirements
1. Select the most appropriate toolkit based on the temporal requirements of the subquestion
2. Consider the type and number of seed entities
3. Can recommend multiple toolkits for combined retrieval
4. Provide specific parameter suggestions for each toolkit

## Output Format
Please output your selection in JSON format as follows:
{{
"selected_toolkits": [
    {{
        "toolkit_name": "EntityFirstAfter",
        "parameters": {{
            "entity": "Iraq",
            "after": "2006-01-05"
        }},
        "reasoning": "Selection reason",
        "priority": 1
    }},
    {{
        "toolkit_name": "EntityEventsAfter",
        "parameters": {{
            "entity": "Iraq",
            "after": "2006-01-05",
            "limit": 10
        }},
        "reasoning": "Selection reason",
        "priority": 2
    }}
]
}}

Please ensure:
1. Toolkit names must be English names from the above list
2. Parameters must meet toolkit requirements
3. Provide clear reasoning process
4. Sort by priority (1 is highest priority)
"""

LLM_FALLBACK_ANSWER_PROMPT = """
You are an expert knowledge graph analyst. Please Using your own knowledge plus available evidence, generate a predict answer for the given question as possible.

Please follow this reasoning format:
1. Provide your final answer in this exact format based on the evidence and your own knowledge: 'So the answer is: <final concise answer>'
2. The final answer can be multiple entities or time points

Return JSON: {{ "final_answer": "answer in the format 'So the answer is: <answer>'", "reasoning": "your step-by-step analysis"}}

Examples of good reasoning:

Example 1:
Q:
Original question: after the death of Mahmoud Abbas, who was the first to visit Iraq?
Step-by-step evidence:
Step 1 subquestion: When did Mahmoud Abbas die?
Step 1 answer: Mahmoud Abbas died on 2025-05-14.
Step 2 subquestion: After 2025-05-14, who was the first to visit Iraq?
Step 2 answer: Iran, China, United_States visited Iraq on 2025-05-15.
A:
{{"final_answer": "So the answer is: Iran, China, United_States", "reasoning": " Mahmoud Abbas died on 2025-05-14.After 2025-05-14, Iran, China, United_States visited Iraq on 2025-05-15.So the answer is: Iran, China, United_States"}}

The Question is:
Q:
Original question: {original_question}

Step-by-step evidence:
{subquestions_details}

A:
"""
