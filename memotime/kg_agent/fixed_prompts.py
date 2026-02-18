# =============================
# file: kg_agent/fixed_prompts.py
# =============================

# fixed simple text format prompts

# AFTER_FIRST type decomposition prompt
LLM_DECOMP_AFTER_FIRST_PROMPT = (
    "Decompose 'after_first' temporal questions into subquestions with indicators.\n"
    "Pattern: After X, who was the first to Y?\n"
    "Strategy: 1) Find reference time when X did Y, 2) Find first entity to do Y after that time\n\n"
    "If X is a time, please refer to Example 4 and using ISO format for X, YYYY-MM-DD\n"
    "Examples:\n\n"
    
    "Example 1:\n"
    "Q: After the Cabinet Council of Ministers of Kazakhstan, who was the first to express the intention to negotiate with Japan?\n"
    "Subquestions: [\"When did Cabinet Council of Ministers of Kazakhstan express the intention to negotiate with Japan?\", \"After t1, who was the first to express the intention to negotiate with Japan?\"]\n"
    "Indicators: [\"Cabinet Council of Ministers of Kazakhstan --[express_intention_to_negotiate]--> Japan (t1)\", \"?x --[express_intention_to_negotiate]--> Japan (t2)\"]\n"
    "Constraints: [\"t2 > t1\", \"first_after(t2, t1)\"]\n"
    "Time_vars: [\"t1\", \"t2\"]\n\n"
    "Example 2:\n"
    "Q: After the Danish Ministry of Defence and Security, who was the first to visit Iraq?\n"
    "Subquestions: [\"When did Danish Ministry of Defence and Security visit Iraq?\", \"After t1, who was the first to visit Iraq?\"]\n"
    "Indicators: [\"Danish Ministry of Defence and Security --[visit]--> Iraq (t1)\", \"?x --[visit]--> Iraq (t2)\"]\n"
    "Constraints: [\"t2 > t1\", \"first_after(t2, t1)\"]\n"
    "Time_vars: [\"t1\", \"t2\"]\n\n"
    "Example 3:\n"
    "Q: After Devlet Bahçeli, who was the first to want to negotiate with Iraq?\n"
    "Subquestions: [\"When did Devlet Bahçeli want to negotiate with Iraq?\", \"After t1, who was the first to want to negotiate with Iraq?\"]\n"
    "Indicators: [\"Devlet Bahçeli --[want_to_negotiate]--> Iraq (t1)\", \"?x --[want_to_negotiate]--> Iraq (t2)\"]\n"
    "Constraints: [\"t2 > t1\", \"first_after(t2, t1)\"]\n"
    "Time_vars: [\"t1\", \"t2\"]\n\n"
    "Example 4:\n"
    "Q: After 2006 July 05, who wanted to negotiate with the Governor of Thailand?\n"
    "Subquestions: [\"After 2006-07-05, who wanted to negotiate with the Governor of Thailand?\"]\n"
    "Indicators: [ \"?x --[want_to_negotiate]--> Governor of Thailand (t2)\"\n"
    "Constraints: [\"t2 > 2006-07-05\", \"after(t2, 2006-07-05)\"\n"
    "Time_vars: [\"t2\", \"2006-07-05\"]\n\n"
    "{enhanced_prompt}"
    "Rules: Use ?x for unknown entities; t1, t2 for time variables; t2 > t1\n"
    "IMPORTANT: Each subquestion must be a complete, standalone question ending with '?'.\n"
    "Do NOT split subquestions into fragments. Each subquestion should be a full sentence.\n\n"
    "Question: {question}\n"
    "Return format:\n"
    "Subquestions: [sub1, sub2, ...]\n"
    "Indicators: [edge1, edge2, ...]\n"
    "Constraints: [constraint1, constraint2, ...]\n"
    "Time_vars: [t1, t2, ...]"
)

# BEFORE_LAST type decomposition prompt
LLM_DECOMP_BEFORE_LAST_PROMPT = (
    "Decompose 'before_last' temporal questions into subquestions with indicators.\n"
    "Pattern: Before X, which Y did Z last?\n"
    "Strategy: 1) Find reference time when Z did X, 2) Find last entity that Z did Y before that time\n\n"
    "If X is a time, please using ISO format for X, YYYY-MM-DD\n"
    "Examples:\n\n"
    "Example 1:\n"
    "Q: Before the military of Taiwan, which country did Japan threaten last?\n"
    "Subquestions: [\"When did Japan threaten the military of Taiwan?\", \"Before t1, which country did Japan threaten last?\"]\n"
    "Indicators: [\"Japan --[threaten]--> military of Taiwan (t1)\", \"Japan --[threaten]--> ?x (t2)\"]\n"
    "Constraints: [\"t2 < t1\", \"last_before(t2, t1)\"]\n"
    "Time_vars: [\"t1\", \"t2\"]\n\n"
    "Example 2:\n"
    "Q: With whom did Catherine Ashton last wish to meet before Cambodia?\n"
    "Subquestions: [\"When did Catherine Ashton wish to meet Cambodia?\", \"Before t1, with whom did Catherine Ashton wish to meet last?\"]\n"
    "Indicators: [\"Catherine Ashton --[wish_to_meet]--> Cambodia (t1)\", \"Catherine Ashton --[wish_to_meet]--> ?x (t2)\"\n"
    "Constraints: [\"t2 < t1\", \"last_before(t2, t1)\"\n"
    "Time_vars: [\"t1\", \"t2\"]\n\n"
    "Example 3:\n"
    "Q: Before Thailand, who last wanted to negotiate with the Governor of Thailand?\n"
    "Subquestions: [\"When did someone want to negotiate with the Governor of Thailand about Thailand?\", \"Before t1, who last wanted to negotiate with the Governor of Thailand?\"]\n"
    "Indicators: [\"?y --[want_to_negotiate]--> Governor of Thailand (t1)\", \"?x --[want_to_negotiate]--> Governor of Thailand (t2)\"\n"
    "Constraints: [\"t2 < t1\", \"last_before(t2, t1)\"\n"
    "Time_vars: [\"t1\", \"t2\"]\n\n"
    "Example 4:\n"
    "Q: Before 2006 July 05, who wanted to negotiate with the Governor of Thailand?\n"
    "Subquestions: [\"Before 2006-07-05, who wanted to negotiate with the Governor of Thailand?\"]\n"
    "Indicators: [ \"?x --[want_to_negotiate]--> Governor of Thailand (t2)\"\n"
    "Constraints: [\"t2 < 2006-07-05\", \"before(t2, 2006-07-05)\"\n"
    "Time_vars: [\"t2\", \"2006-07-05\"]\n\n"
    "{enhanced_prompt}"

    "Rules: Use ?x, ?y for unknown entities; t1, t2 for time variables; t2 < t1\n"
    "IMPORTANT: Each subquestion must be a complete, standalone question ending with '?'.\n"
    "Do NOT split subquestions into fragments. Each subquestion should be a full sentence.\n\n"
    "Question: {question}\n"
    "Return format:\n"
    "Subquestions: [sub1, sub2, ...]\n"
    "Indicators: [edge1, edge2, ...]\n"
    "Constraints: [constraint1, constraint2, ...]\n"
    "Time_vars: [t1, t2, ...]"
)


# BEFORE_LAST type decomposition prompt
LLM_DECOMP_BEFORE_AFTER_PROMPT = (
    "Decompose 'before_after' temporal questions into subquestions with indicators.\n"
    "Pattern: Before X, which Y did Z ? or After X, which Y did Z ?\n"
    "Strategy: 1) Find reference time when Z did X, 2) Find entity that Z did Y before or after that time 3) Sometimes X is a time, then find entity that Z did Y before or after that time\n\n"
    "If X is a time, please refer to Example 4 and using ISO format for X, YYYY-MM-DD\n"
    "Examples:\n\n"
    "Example 1:\n"
    "Q: Before the military of Taiwan, which country did Japan threaten?\n"
    "Subquestions: [\"When did Japan threaten the military of Taiwan?\", \"Before t1, which country did Japan threaten?\"]\n"
    "Indicators: [\"Japan --[threaten]--> military of Taiwan (t1)\", \"Japan --[threaten]--> ?x (t2)\"]\n"
    "Constraints: [\"t2 < t1\", \"before(t2, t1)\"]\n"
    "Time_vars: [\"t1\", \"t2\"]\n\n"
    "Example 2:\n"
    "Q: Before 2006 July 05, which country did Japan threaten?\n"
    "Subquestions: [\"Before 2006-07-05, which country did Japan threaten?\"]\n"
    "Indicators: [ \"?x --[threaten]--> Japan (t2)\"\n"
    "Constraints: [\"t2 < 2006-07-05\", \"before(t2, 2006-07-05)\"\n"
    "Time_vars: [\"t2\", \"2006-07-05\"]\n\n"

    "Example 3:\n"
    "Q: With whom did Catherine Ashton wish to meet before Cambodia?\n"
    "Subquestions: [\"When did Catherine Ashton wish to meet Cambodia?\", \"Before t1, with whom did Catherine Ashton wish to meet?\"]\n"
    "Indicators: [\"Catherine Ashton --[wish_to_meet]--> Cambodia (t1)\", \"Catherine Ashton --[wish_to_meet]--> ?x (t2)\"\n"
    "Constraints: [\"t2 < t1\", \"before(t2, t1)\"\n"
    "Time_vars: [\"t1\", \"t2\"]\n\n"

    "Example 4:\n"
    "Q: After Thailand, who wanted to negotiate with the Governor of Thailand?\n"
    "Subquestions: [\"When did someone want to negotiate with the Governor of Thailand about Thailand?\", \"After t1, who wanted to negotiate with the Governor of Thailand?\"]\n"
    "Indicators: [\"Thailand --[want_to_negotiate]--> Governor of Thailand (t1)\", \"?x --[want_to_negotiate]--> Governor of Thailand (t2)\"\n"
    "Constraints: [\"t2 > t1\", \"after(t2, t1)\"\n"
    "Time_vars: [\"t1\", \"t2\"]\n\n"

    "Example 5:\n"
    "Q: After 2006 July 05, who wanted to negotiate with the Governor of Thailand?\n"
    "Subquestions: [\"After 2006-07-05, who wanted to negotiate with the Governor of Thailand?\"]\n"
    "Indicators: [ \"?x --[want_to_negotiate]--> Governor of Thailand (t2)\"\n"
    "Constraints: [\"t2 > 2006-07-05\", \"after(t2, 2006-07-05)\"\n"
    "Time_vars: [\"t2\", \"2006-07-05\"]\n\n"

    "{enhanced_prompt}"

    "Rules: Use ?x, ?y for unknown entities; t1, t2 for time variables; t2 < t1\n"
    "IMPORTANT: Each subquestion must be a complete, standalone question ending with '?'.\n"
    "Do NOT split subquestions into fragments. Each subquestion should be a full sentence.\n\n"
    "Question: {question}\n"
    "Return format:\n"
    "Subquestions: [sub1, sub2, ...]\n"
    "Indicators: [edge1, edge2, ...]\n"
    "Constraints: [constraint1, constraint2, ...]\n"
    "Time_vars: [t1, t2, ...]"
)

# FIRST_LAST type decomposition prompt
LLM_DECOMP_FIRST_LAST_PROMPT = (
    "Decompose 'first_last' temporal questions into subquestions with indicators.\n"
    "Pattern: When did X last/first Y? or Who was the first/last to Y?\n"
    "Strategy: Direct temporal query with ordering constraint\n\n"
    "Examples:\n\n"
    "Example 1:\n"
    "Q: When did Kitti Wasinondh last express an intention to negotiate with Thailand?\n"
    "Subquestions: [\"When did Kitti Wasinondh last express an intention to negotiate with Thailand?\"]\n"
    "Indicators: [\"Kitti Wasinondh --[express_intention_to_negotiate]--> Thailand (t1)\"\n"
    "Constraints: [\"last(t1)\"\n"
    "Time_vars: [\"t1\"]\n\n"
    "Example 2:\n"
    "Q: In which year did Taiwan's Ministry of National Defence and Security last make a request to Japan?\n"
    "Subquestions: [\"When did Taiwan's Ministry of National Defence and Security last make a request to Japan?\"]\n"
    "Indicators: [\"Taiwan's Ministry of National Defence and Security --[make_request_to]--> Japan (t1)\"\n"
    "Constraints: [\"last(t1)\"\n"
    "Time_vars: [\"t1\"]\n\n"
    "Example 3:\n"
    "Q: Who was the first country that Ethiopia expressed optimism about?\n"
    "Subquestions: [\"Who was the first country that Ethiopia expressed optimism about?\"]\n"
    "Indicators: [\"Ethiopia --[express_optimism_about]--> ?x (t1)\"\n"
    "Constraints: [\"first(t1)\"\n"
    "Time_vars: [\"t1\"]\n\n"
    "{enhanced_prompt}"

    "Rules: Use ?x for unknown entities; t1 for time variable; first(t1) or last(t1) constraints\n"
    "IMPORTANT: Each subquestion must be a complete, standalone question.\n\n"
    "Question: {question}\n"
    "Return format:\n"
    "Subquestions: [sub1, sub2, ...]\n"
    "Indicators: [edge1, edge2, ...]\n"
    "Constraints: [constraint1, constraint2, ...]\n"
    "Time_vars: [t1, t2, ...]"
)

# EQUAL type decomposition prompt
LLM_DECOMP_EQUAL_PROMPT = (
    "Decompose 'equal' temporal questions into subquestions with indicators.\n"
    "Pattern: Who X in Y? or What happened in Y?\n"
    "Strategy: Direct query with time constraint\n\n"
    "Examples:\n\n"
    "Example 1:\n"
    "Q: Who signed an agreement with Japan in April 2005?\n"
    "Subquestions: [\"Who signed an agreement with Japan in April 2005?\"]\n"
    "Indicators: [\"?x --[sign_agreement]--> Japan (t1)\"\n"
    "Constraints: [\"t1 = 2005-04\"]\n"
    "Time_vars: [\"t1\"]\n\n"
    "Example 2:\n"
    "Q: In which year did Malaysia visit Japan?\n"
    "Subquestions: [\"In which year did Malaysia visit Japan?\"]\n"
    "Indicators: [\"Malaysia --[visit]--> Japan (t1)\"\n"
    "Constraints: [\"t1 = specific_time\"]\n"
    "Time_vars: [\"t1\"]\n\n"
    "Example 3:\n"
    "Q: Who visited Malaysia on 14 January 2007?\n"
    "Subquestions: [\"Who visited Malaysia on 14 January 2007?\"]\n"
    "Indicators: [\"?x --[visit]--> Malaysia (t1)\"\n"
    "Constraints: [\"t1 = 2007-01-14\"]\n"
    "Time_vars: [\"t1\"]\n\n"
    "Example 4:\n"
    "Q: Which country condemned Japan in August 2013?\n"
    "Subquestions: [\"Which country condemned Japan in August 2013?\"]\n"
    "Indicators: [\"?x --[condemn]--> Japan (t1)\"\n"
    "Constraints: [\"t1 = 2013-08\"]\n"
    "Time_vars: [\"t1\"]\n\n"
    "{enhanced_prompt}"

    "Rules: Use ?x for unknown entities; t1 for time variable; t1 = specific_time\n"
    "IMPORTANT: Each subquestion must be a complete, standalone question.\n\n"
    "Question: {question}\n"
    "Return format:\n"
    "Subquestions: [sub1, sub2, ...]\n"
    "Indicators: [edge1, edge2, ...]\n"
    "Constraints: [constraint1, constraint2, ...]\n"
    "Time_vars: [t1, t2, ...]"
)

# EQUAL_MULTI type decomposition prompt
LLM_DECOMP_EQUAL_MULTI_PROMPT = (
    "Decompose 'equal_multi' temporal questions into subquestions with indicators.\n"
    "Pattern: Who X in the same Y as Z? or Who X in Y?\n"
    "Strategy: 1) Find reference time when Z did X, 2) Find entities that did X at same time\n\n"
    "Examples:\n\n"
    "Example 1:\n"
    "Q: Who visited Japan in the same month as Oleg Ostapenko?\n"
    "Subquestions: [\"When did Oleg Ostapenko visit Japan?\", \"Who visited Japan in the same month as t1?\"]\n"
    "Indicators: [\"Oleg Ostapenko --[visit]--> Japan (t1)\", \"?x --[visit]--> Japan (t2)\"\n"
    "Constraints: [\"same_month(t1, t2)\"\n"
    "Time_vars: [\"t1\", \"t2\"]\n\n"
    "Example 2:\n"
    "Q: In 2012, which country was the first to express interest in cooperation with Cambodia?\n"
    "Subquestions: [\"When did Cambodia receive expressions of interest in cooperation?\", \"In 2012, which country was the first to express interest in cooperation with Cambodia?\"]\n"
    "Indicators: [\"?y --[express_interest_in_cooperation]--> Cambodia (t1)\", \"?x --[express_interest_in_cooperation]--> Cambodia (t2)\"\n"
    "Constraints: [\"t2 >= t1\", \"first_after(t2, t1)\", \"t1 = 2012\"]\n"
    "Time_vars: [\"t1\", \"t2\"]\n\n"
    "Example 3:\n"
    "Q: Who did Ethiopia use conventional military force against on the same day as the Hizbul Islam fighter?\n"
    "Subquestions: [\"When did the Hizbul Islam fighter use conventional military force?\", \"Who did Ethiopia use conventional military force against on the same day as t1?\"]\n"
    "Indicators: [\"Hizbul Islam fighter --[use_conventional_military_force]--> ?y (t1)\", \"Ethiopia --[use_conventional_military_force]--> ?x (t2)\"\n"
    "Constraints: [\"same_day(t1, t2)\"\n"
    "{enhanced_prompt}"
    "Time_vars: [\"t1\", \"t2\"]\n\n"
    "Rules: Use ?x, ?y for unknown entities; t1, t2 for time variables; same_month/same_day constraints\n"
    "IMPORTANT: Each subquestion must be a complete, standalone question.\n\n"
    "Question: {question}\n"
    "Return format:\n"
    "Subquestions: [sub1, sub2, ...]\n"
    "Indicators: [edge1, edge2, ...]\n"
    "Constraints: [constraint1, constraint2, ...]\n"
    "Time_vars: [t1, t2, ...]"
)

# default decomposition prompt (when cannot identify type)
LLM_DECOMP_DEFAULT_PROMPT = (
    "Decompose temporal questions into subquestions with indicators.\n"
    "Use ?x, ?y for unknown entities; t1, t2 for time variables\n"
    "IMPORTANT: Each subquestion must be a complete, standalone question.\n\n"
    "{enhanced_prompt}"

    "Question: {question}\n"
    "Return format:\n"
    "Subquestions: [sub1, sub2, ...]\n"
    "Indicators: [edge1, edge2, ...]\n"
    "Constraints: [constraint1, constraint2, ...]\n"
    "Time_vars: [t1, t2, ...]"
)

cold_start_toolkit_examples = {
    "equal": [
        {
            "subquestion": "When did Japan threaten Taiwan?",
            "indicator": {
                "edges": [{"subj": "Japan", "rel": "threaten", "obj": "Taiwan", "time_var": "t1"}],
                "constraints": []
            },
            "seed_info": ["ID: 67890, Name: Japan","ID: 7788, Name: Taiwan"],
            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Japan", "entity2": "Taiwan", "direction": "both", "limit": 200},

            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection for direct matching."
        },
        # new: cover more when type scenarios
        {
            "subquestion": "In which year was the peace agreement signed?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "sign", "obj": "peace agreement", "time_var": "t1"}],
                "constraints": []
            },
            "seed_info": ["ID: 11111, Name: peace_agreement"],
            "toolkit": "OneHop",
            "parameters": {"entity": "peace_agreement", "direction": "both", "sort_by_time": True, "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "OneHop + sort_by_time for finding the events happened in the specific year."
        },
        {
            "subquestion": "Who wants to negotiate with China on 16 July 2009?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "negotiate", "obj": "China", "time_var": "t1 = 2009-07-16"}],
                "constraints": []
            },
            "seed_info": ["ID: 67890, Name: China"],
            "toolkit": "DayEvents",
            "parameters": {"entity": "China", "day": "2009-07-16", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DayEvents find the events on the specific day."
        },
        {
            "subquestion": "Who wants to negotiate with China in July 2009?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "negotiate", "obj": "China", "time_var": "t1 = 2009-07"}],
                "constraints": []
            },
            "seed_info": ["ID: 67890, Name: China"],
            "toolkit": "MonthEvents", 
            "parameters": {"entity": "China", "month": "2009-07", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "MonthEvents find the events on the specific month."
        }
    ],
    
    "after_first": [
        {
            "subquestion": "when did the Cabinet Council of Ministers of Kazakhstan express the intention to negotiate with Japan?",
            "indicator": {
                "edges": [{"subj": "Cabinet Council of Ministers of Kazakhstan", "rel": "express_intention_to_negotiate", "obj": "Japan", "time_var": "t1"}],
                "constraints": ["t2 > t1", "first_after(t2, t1)"]
            },
            "entity_names": ['Japan', 'Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)'],
            "seed_info": ["ID: 62, Name: Japan",
            "ID: 4774, Name: Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)"
            ],

            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)", "entity2": "Japan", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection is suitable as it allows querying the relationship between the Cabinet Council of Ministers of Kazakhstan and Japan, focusing on the intention to negotiate."
        },
        {
            "subquestion": "After 2006-01-05, who was the first to visit Iraq?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "visit", "obj": "Iraq", "time_var": "t2"}],
                "constraints": ["t2 > 2006-01-05", "first_after(t2, 2006-01-05)"]
            },
            "seed_info": ["ID: 67890, Name: Iraq"],
            "toolkit": "AfterFirst",
            "parameters": {"entity": "Iraq", "after": "2006-01-05", "limit": 1},
            "context": {"times": {"t1": "2006-01-05"}},
            "time_hints": {"after": "2006-01-05"},
            "reasoning": "AfterFirst can handle the after first situation。"
        },
        # new: more after_first scenarios
        {
            "subquestion": "After the 2007-03-15, who first expressed support for the proposal?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "express_support", "obj": "proposal", "time_var": "t2"}],
                "constraints": ["t2 > 2007-03-15", "first_after(t2, 2007-03-15)"]
            },
            "seed_info": ["ID: 22222, Name: proposal"],
            "toolkit": "AfterFirst",
            "parameters": {"entity": "proposal", "after": "2007-03-15", "limit": 100},
            "context": {"times": {"t1": "2007-03-15"}},
            "time_hints": {"after": "2007-03-15"},
            "reasoning": "AfterFirst can handle the after first situation。"
        }
    ],
    
    
    "before_last": [
        {
            "subquestion": "when did the Cabinet Council of Ministers of Kazakhstan express the intention to negotiate with Japan?",
            "indicator": {
                "edges": [{"subj": "Cabinet Council of Ministers of Kazakhstan", "rel": "express_intention_to_negotiate", "obj": "Japan", "time_var": "t1"}],
                "constraints": ["t2 > t1", "first_after(t2, t1)"]
            },
            "entity_names": ['Japan', 'Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)'],
            "seed_info": ["ID: 62, Name: Japan",
            "ID: 4774, Name: Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)"
            ],

            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)", "entity2": "Japan", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection is suitable as it allows querying the relationship between the Cabinet Council of Ministers of Kazakhstan and Japan, focusing on the intention to negotiate."
        },
        {
            "subquestion": "Before 2006-01-05, which country did Japan threaten last?",
            "indicator": {
                "edges": [{"subj": "Japan", "rel": "threaten", "obj": "?x", "time_var": "t2"}],
                "constraints": ["t2 < 2006-01-05", "last_before(t2, 2006-01-05)"]
            },
            "seed_info": ["ID: 12345, Name: Japan"],
            "toolkit": "BeforeLast",
            "parameters": {"entity": "Japan", "before": "2006-01-05", "limit": 100},

            "context": {"times": {"t1": "2006-01-05"}},
            "time_hints": {"before": "2006-01-05"},
            "reasoning": "BeforeLast can handle the before last situation。"
        }
    ],
    "before_after": [
        {
            "subquestion": "when did the Cabinet Council of Ministers of Kazakhstan express the intention to negotiate with Japan?",
            "indicator": {
                "edges": [{"subj": "Cabinet Council of Ministers of Kazakhstan", "rel": "express_intention_to_negotiate", "obj": "Japan", "time_var": "t1"}],
                "constraints": [ "None(t1)"]
            },
            "entity_names": ['Japan', 'Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)'],
            "seed_info": ["ID: 62, Name: Japan",
            "ID: 4774, Name: Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)"
            ],

            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)", "entity2": "Japan", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection is suitable as it allows querying the relationship between the Cabinet Council of Ministers of Kazakhstan and Japan, focusing on the intention to negotiate."
        },
        {
            "subquestion": "Before 2006-01-05, which country did Japan threaten?",
            "indicator": {
                "edges": [{"subj": "Japan", "rel": "threaten", "obj": "?x", "time_var": "t2"}],
                "constraints": ["t2 < 2006-01-05", "before(t2, 2006-01-05)"]
            },
            "seed_info": ["ID: 12345, Name: Japan"],
            "toolkit": "BeforeLast",
            "parameters": {"entity": "Japan", "before": "2006-01-05", "limit": 100},

            "context": {"times": {"t1": "2006-01-05"}},
            "time_hints": {"before": "2006-01-05"},
            "reasoning": "BeforeLast can handle the before situation。"
        },# new: more after_first scenarios
        {
            "subquestion": "After the 2007-03-15, who expressed support for the proposal?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "express_support", "obj": "proposal", "time_var": "t2"}],
                "constraints": ["t2 > 2007-03-15", "after(t2, 2007-03-15)"]
            },
            "seed_info": ["ID: 22222, Name: proposal"],
            "toolkit": "AfterFirst",
            "parameters": {"entity": "proposal", "after": "2007-03-15", "limit": 100},
            "context": {"times": {"t1": "2007-03-15"}},
            "time_hints": {"after": "2007-03-15"},
            "reasoning": "AfterFirst can handle the after situation。"
        }
    ],
    
    
    "during_between": [
        {
            "subquestion": "What happened between 2005-01-01 and 2005-12-31 in Iraq?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "event", "obj": "Iraq", "time_var": "t1 in between(2005-01-01,2005-12-31)"}],
                "constraints": ["t1 between 2005-01-01 and 2005-12-31"]
            },
            "seed_info": ["ID: 12345, Name: Iraq"],
            "toolkit": "BetweenRange",
            "parameters": {"entity": "Iraq", "between": ["2005-01-01", "2005-12-31"], "limit": 20},
            "context": {},
            "time_hints": {"between": ["2005-01-01", "2005-12-31"]},
            "reasoning": "BetweenRange handle the interval event."
        },
        # new: more between scenarios
        {
            "subquestion": "Who visited Japan during the Olympic Games period between 2008-08-08 and 2008-08-24?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "visit", "obj": "Japan", "time_var": "t1 in between(2008-08-08,2008-08-24)"}],
                "constraints": ["t1 between 2008-08-08 and 2008-08-24"]
            },
            "seed_info": ["ID: 33333, Name: Japan"],
            "toolkit": "BetweenRange",
            "parameters": {"entity": "Japan", "between": ["2008-08-08", "2008-08-24"], "limit": 100},
            "context": {},
            "time_hints": {"between": ["2008-08-08", "2008-08-24"]},
            "reasoning": "BetweenRange suitable for events within a specific period."
        }
    ],
    
    # "direct_connection": [
    #     {
    #         "subquestion": "Is there a direct connection between Japan and Iraq?",
    #         "indicator": {
    #             "edges": [{"subj": "Japan", "rel": "?r", "obj": "Iraq"}],
    #             "constraints": []
    #         },
    #         "seeds": [12345, 67890],
    #         "toolkit": "DirectConnection",
    #         "parameters": {"entity1": "Japan", "entity2": "Iraq", "direction": "both", "limit": 200},
    #         "context": {},
    #         "time_hints": {},
    #         "reasoning": "DirectConnection verify the direct relationship."
    #     },
    #     # new: more direct connection scenarios
    #     {
    #         "subquestion": "What is the relationship between USA and NATO?",
    #         "indicator": {
    #             "edges": [{"subj": "USA", "rel": "?r", "obj": "NATO"}],
    #             "constraints": []
    #         },
    #         "seeds": [44444, 55555],
    #         "toolkit": "DirectConnection",
    #         "parameters": {"entity1": "USA", "entity2": "NATO", "direction": "both", "limit": 150},
    #         "context": {},
    #         "time_hints": {},
    #         "reasoning": "DirectConnection used to find all direct relationships between two entities."
    #     }
    # ],
    
    # "timeline": [
    #     {
    #         "subquestion": "What is the timeline of events for Iraq after 2000-01-01?",
    #         "indicator": {
    #             "edges": [{"subj": "?x", "rel": "?r", "obj": "Iraq", "time_var": "t1"}],
    #             "constraints": ["t1 > 2000-01-01"]
    #         },
    #         "seeds": [12345],
    #         "toolkit": "Timeline",
    #         "parameters": {"entity": "Iraq", "direction": "both", "after": "2000-01-01", "sort_by_time": True, "limit": 100},
    #         "context": {},
    #         "time_hints": {"after": "2000-01-01"},
    #         "reasoning": "Timeline build the time sequence."
    #     },
    #     # new: more comprehensive timeline scenarios
    #     {
    #         "subquestion": "Show me the diplomatic history of Germany",
    #         "indicator": {
    #             "edges": [{"subj": "Germany", "rel": "?r", "obj": "?x", "time_var": "t1"}],
    #             "constraints": []
    #         },
    #         "seeds": [66666],
    #         "toolkit": "Timeline",
    #         "parameters": {"entity": "Germany", "direction": "both", "sort_by_time": True, "limit": 200},
    #         "context": {},
    #         "time_hints": {},
    #         "reasoning": "Timeline provide a complete time sequence view."
    #     }
    # ],
    
    "first_last": [
        {
            "subquestion": "In which year did Taiwan's Ministry of National Defence and Security last make a request to Japan?",
            "indicator": {
                "edges": [{"subj": "Taiwan's Ministry of National Defence and Security", "rel": "make_request", "obj": "Japan", "time_var": "t1"}],
                "constraints": ["last(t1)"]
            },
            "available_toolkits": ["DirectConnection", "BeforeLast"],
            "available_entities": ["Taiwan's National Defence and Security", "Japan"],

            
            "selected_toolkit": "DirectConnection",
            "parameters": {"entity1": "Taiwan's National Defence and Security", "entity2": "Japan", "before": "2025-01-01","direction": "both", "limit": 200},
            "context": {"times": {"t1": "2025-01-01"}},
            "time_hints": {"before": "2025-01-01"},
            "reasoning": "DirectConnection verify the direct relationship. Time set to latest year 2025-01-01",
        },




        {
            "subquestion": "When did Kitti Wasinondh last express an intention to negotiate with Thailand?",
            "indicator": {
                "edges": [{"subj": "Kitti Wasinondh", "rel": "express_intention_to_negotiate", "obj": "Thailand", "time_var": "t1"}],
                "constraints": ["last(t1)"]
            },
            "available_toolkits": ["BeforeLast", "DirectConnection"],
            "available_entities": ["Kitti Wasinondh", "Thailand"],

            
            "selected_toolkit": "DirectConnection",
            "parameters": {"entity1": "Kitti Wasinondh", "entity2": "Thailand", "before": "2025-01-01", "limit": 100},
            "context": {"times": {"t1": "2025-01-01"}},
            "time_hints": {"before": "2025-01-01"},
            "reasoning": "DirectConnection verify the direct relationship. Time set to latest year 2025-01-01"
        },
        {
            "subquestion": "Who was the first country that Ethiopia expressed optimism about?",
            "indicator": {
                "edges": [{"subj": "Ethiopia", "rel": "express_optimism", "obj": "?x", "time_var": "t1"}],
                "constraints": ["first(t1)"]
            },
            "available_toolkits": ["AfterFirst", "DirectConnection"],
            "available_entities": ["Ethiopia"],

            
            "selected_toolkit": "AfterFirst",
            "parameters": {"entity": "Ethiopia", "after": "1800-01-01", "limit": 100},
            "context": {"times": {"t1": "1800-01-01"}},
            "time_hints": {"after": "1800-01-01"},
            "reasoning": "AfterFirst handle the first event. Time set to earliest year 1800-01-01"
        }
    ],
    "equal_multi":[
        {
            "subquestion": "when did the Cabinet Council of Ministers of Kazakhstan express the intention to negotiate with Japan?",
            "indicator": {
                "edges": [{"subj": "Cabinet Council of Ministers of Kazakhstan", "rel": "express_intention_to_negotiate", "obj": "Japan", "time_var": "t1"}],
                "constraints": ["t2 > t1", "first_after(t2, t1)"]
            },
            "entity_names": ['Japan', 'Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)'],
            "seed_info": ["ID: 62, Name: Japan",
            "ID: 4774, Name: Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)"
            ],

            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)", "entity2": "Japan", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection is suitable as it allows querying the relationship between the Cabinet Council of Ministers of Kazakhstan and Japan, focusing on the intention to negotiate."
        },
        {
            "subquestion": "Who did Ethiopia use conventional military force against on 2005-01-01?",
            "indicator": {
                "edges": [{"subj": "Ethiopia", "rel": "use_conventional_military_force", "obj": "?x", "time_var": "t1 = 2005-01-01"}],
                "constraints": ["t1 = 2005-01-01"]
            },
            "seed_info": ["ID: 67890, Name: Ethiopia"],

            "toolkit": "OneHop",
            "parameters": {"entity": "Ethiopia", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "OneHop is suitable as it allows querying the relationship between Ethiopia and the unknown entity, focusing on the use of conventional military force."
        },

        {
            "subquestion": "When did Japan threaten Taiwan?",
            "indicator": {
                "edges": [{"subj": "Japan", "rel": "threaten", "obj": "Taiwan", "time_var": "t1"}],
                "constraints": []
            },
            "seed_info": ["ID: 67890, Name: Japan","ID: 7788, Name: Taiwan"],
            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Japan", "entity2": "Taiwan", "direction": "both", "limit": 200},

            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection for direct matching."
        },
        # new: cover more when type scenarios
        {
            "subquestion": "In which year was the peace agreement signed?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "sign", "obj": "peace agreement", "time_var": "t1"}],
                "constraints": []
            },
            "seed_info": ["ID: 11111, Name: peace_agreement"],
            "toolkit": "OneHop",
            "parameters": {"entity": "peace_agreement", "direction": "both", "sort_by_time": True, "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "OneHop + sort_by_time for finding the events happened in the specific year."
        },
        {
            "subquestion": "Who wants to negotiate with China on 16 July 2009?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "negotiate", "obj": "China", "time_var": "t1 = 2009-07-16"}],
                "constraints": []
            },
            "seed_info": ["ID: 67890, Name: China"],
            "toolkit": "DayEvents",
            "parameters": {"entity": "China", "day": "2009-07-16", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DayEvents find the events on the specific day."
        },
        {
            "subquestion": "Who wants to negotiate with China in July 2009?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "negotiate", "obj": "China", "time_var": "t1 = 2009-07"}],
                "constraints": []
            },
            "seed_info": ["ID: 67890, Name: China"],
            "toolkit": "MonthEvents", 
            "parameters": {"entity": "China", "month": "2009-07", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "MonthEvents find the events on the specific month."
        }
    ],
    "general": [
        {
            "subquestion": "when did the Cabinet Council of Ministers of Kazakhstan express the intention to negotiate with Japan?",
            "indicator": {
                "edges": [{"subj": "Cabinet Council of Ministers of Kazakhstan", "rel": "express_intention_to_negotiate", "obj": "Japan", "time_var": "t1"}],
                "constraints": ["t2 > t1", "first_after(t2, t1)"]
            },
            "entity_names": ['Japan', 'Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)'],
            "seed_info": ["ID: 62, Name: Japan",
            "ID: 4774, Name: Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)"
            ],

            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)", "entity2": "Japan", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection is suitable as it allows querying the relationship between the Cabinet Council of Ministers of Kazakhstan and Japan, focusing on the intention to negotiate."
        },
        {
            "subquestion": "Who did Ethiopia use conventional military force against on 2005-01-01?",
            "indicator": {
                "edges": [{"subj": "Ethiopia", "rel": "use_conventional_military_force", "obj": "?x", "time_var": "t1 = 2005-01-01"}],
                "constraints": ["t1 = 2005-01-01"]
            },
            "seed_info": ["ID: 67890, Name: Ethiopia"],

            "toolkit": "OneHop",
            "parameters": {"entity": "Ethiopia", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "OneHop is suitable as it allows querying the relationship between Ethiopia and the unknown entity, focusing on the use of conventional military force."
        },
        {
            "subquestion": "Before 2006-01-05, which country did Japan threaten last?",
            "indicator": {
                "edges": [{"subj": "Japan", "rel": "threaten", "obj": "?x", "time_var": "t2"}],
                "constraints": ["t2 < 2006-01-05", "last_before(t2, 2006-01-05)"]
            },
            "seed_info": ["ID: 12345, Name: Japan"],
            "toolkit": "BeforeLast",
            "parameters": {"entity": "Japan", "before": "2006-01-05", "limit": 100},

            "context": {"times": {"t1": "2006-01-05"}},
            "time_hints": {"before": "2006-01-05"},
            "reasoning": "BeforeLast get the last event before the specific time."
        }
        # keep existing general examples, they are already comprehensive
    ]
}

def get_general_examples():
    example = []
    question_name = []
    for key, value in cold_start_toolkit_examples.items():
        for item in value:
            question_name = item['subquestion']
            if question_name not in question_name:
                question_name.append(question_name)
                example.append(item)
    return example



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Initial Toolkit Selection Examples
------------------------------------------
enhanced initial example library, solve cold start problem and provide better coverage
"""

import json
import os
from typing import Dict, List, Any

class EnhancedInitialExamples:
    """enhanced initial example manager"""
    
    def __init__(self):
        # use existing example library as base
        self.examples = {
    "equal": [
        {
            "subquestion": "When did Japan threaten Taiwan?",
            "indicator": {
                "edges": [{"subj": "Japan", "rel": "threaten", "obj": "Taiwan", "time_var": "t1"}],
                "constraints": []
            },
            "seed_info": ["ID: 67890, Name: Japan","ID: 7788, Name: Taiwan"],
            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Japan", "entity2": "Taiwan", "direction": "both", "limit": 200},

            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection for direct matching."
        },
        # new: cover more when type scenarios
        {
            "subquestion": "In which year was the peace agreement signed?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "sign", "obj": "peace agreement", "time_var": "t1"}],
                "constraints": []
            },
            "seed_info": ["ID: 11111, Name: peace_agreement"],
            "toolkit": "OneHop",
            "parameters": {"entity": "peace_agreement", "direction": "both", "sort_by_time": True, "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "OneHop + sort_by_time for finding the events happened in the specific year."
        },
        {
            "subquestion": "Who wants to negotiate with China on 16 July 2009?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "negotiate", "obj": "China", "time_var": "t1 = 2009-07-16"}],
                "constraints": []
            },
            "seed_info": ["ID: 67890, Name: China"],
            "toolkit": "DayEvents",
            "parameters": {"entity": "China", "day": "2009-07-16", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DayEvents find the events on the specific day."
        },
        {
            "subquestion": "Who wants to negotiate with China in July 2009?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "negotiate", "obj": "China", "time_var": "t1 = 2009-07"}],
                "constraints": []
            },
            "seed_info": ["ID: 67890, Name: China"],
            "toolkit": "MonthEvents", 
            "parameters": {"entity": "China", "month": "2009-07", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "MonthEvents find the events on the specific month."
        }
    ],
    
    "after_first": [
        {
            "subquestion": "when did the Cabinet Council of Ministers of Kazakhstan express the intention to negotiate with Japan?",
            "indicator": {
                "edges": [{"subj": "Cabinet Council of Ministers of Kazakhstan", "rel": "express_intention_to_negotiate", "obj": "Japan", "time_var": "t1"}],
                "constraints": ["t2 > t1", "first_after(t2, t1)"]
            },
            "entity_names": ['Japan', 'Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)'],
            "seed_info": ["ID: 62, Name: Japan",
            "ID: 4774, Name: Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)"
            ],

            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)", "entity2": "Japan", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection is suitable as it allows querying the relationship between the Cabinet Council of Ministers of Kazakhstan and Japan, focusing on the intention to negotiate."
        },
        {
            "subquestion": "After 2006-01-05, who was the first to visit Iraq?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "visit", "obj": "Iraq", "time_var": "t2"}],
                "constraints": ["t2 > 2006-01-05", "first_after(t2, 2006-01-05)"]
            },
            "seed_info": ["ID: 67890, Name: Iraq"],
            "toolkit": "AfterFirst",
            "parameters": {"entity": "Iraq", "after": "2006-01-05", "limit": 1},
            "context": {"times": {"t1": "2006-01-05"}},
            "time_hints": {"after": "2006-01-05"},
            "reasoning": "AfterFirst can handle the after first situation。"
        },
            # new: more after_first scenarios
        {
            "subquestion": "After the 2007-03-15, who first expressed support for the proposal?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "express_support", "obj": "proposal", "time_var": "t2"}],
                "constraints": ["t2 > 2007-03-15", "first_after(t2, 2007-03-15)"]
            },
            "seed_info": ["ID: 22222, Name: proposal"],
            "toolkit": "AfterFirst",
            "parameters": {"entity": "proposal", "after": "2007-03-15", "limit": 100},
            "context": {"times": {"t1": "2007-03-15"}},
            "time_hints": {"after": "2007-03-15"},
            "reasoning": "AfterFirst can handle the after first situation。"
        }
    ],
    
    
    "before_last": [
        {
            "subquestion": "when did the Cabinet Council of Ministers of Kazakhstan express the intention to negotiate with Japan?",
            "indicator": {
                "edges": [{"subj": "Cabinet Council of Ministers of Kazakhstan", "rel": "express_intention_to_negotiate", "obj": "Japan", "time_var": "t1"}],
                "constraints": ["t2 > t1", "first_after(t2, t1)"]
            },
            "entity_names": ['Japan', 'Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)'],
            "seed_info": ["ID: 62, Name: Japan",
            "ID: 4774, Name: Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)"
            ],

            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)", "entity2": "Japan", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection is suitable as it allows querying the relationship between the Cabinet Council of Ministers of Kazakhstan and Japan, focusing on the intention to negotiate."
        },
        {
            "subquestion": "Before 2006-01-05, which country did Japan threaten last?",
            "indicator": {
                "edges": [{"subj": "Japan", "rel": "threaten", "obj": "?x", "time_var": "t2"}],
                "constraints": ["t2 < 2006-01-05", "last_before(t2, 2006-01-05)"]
            },
            "seed_info": ["ID: 12345, Name: Japan"],
            "toolkit": "BeforeLast",
            "parameters": {"entity": "Japan", "before": "2006-01-05", "limit": 100},

            "context": {"times": {"t1": "2006-01-05"}},
            "time_hints": {"before": "2006-01-05"},
            "reasoning": "BeforeLast can handle the before last situation。"
        }
    ],
    "before_after": [
        {
            "subquestion": "when did the Cabinet Council of Ministers of Kazakhstan express the intention to negotiate with Japan?",
            "indicator": {
                "edges": [{"subj": "Cabinet Council of Ministers of Kazakhstan", "rel": "express_intention_to_negotiate", "obj": "Japan", "time_var": "t1"}],
                "constraints": [ "None(t1)"]
            },
            "entity_names": ['Japan', 'Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)'],
            "seed_info": ["ID: 62, Name: Japan",
            "ID: 4774, Name: Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)"
            ],

            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)", "entity2": "Japan", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection is suitable as it allows querying the relationship between the Cabinet Council of Ministers of Kazakhstan and Japan, focusing on the intention to negotiate."
        },
        {
            "subquestion": "Before 2006-01-05, which country did Japan threaten?",
            "indicator": {
                "edges": [{"subj": "Japan", "rel": "threaten", "obj": "?x", "time_var": "t2"}],
                "constraints": ["t2 < 2006-01-05", "before(t2, 2006-01-05)"]
            },
            "seed_info": ["ID: 12345, Name: Japan"],
            "toolkit": "BeforeLast",
            "parameters": {"entity": "Japan", "before": "2006-01-05", "limit": 100},

            "context": {"times": {"t1": "2006-01-05"}},
            "time_hints": {"before": "2006-01-05"},
            "reasoning": "BeforeLast can handle the before situation。"
        },# new: more after_first scenarios
        {
            "subquestion": "After the 2007-03-15, who expressed support for the proposal?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "express_support", "obj": "proposal", "time_var": "t2"}],
                "constraints": ["t2 > 2007-03-15", "after(t2, 2007-03-15)"]
            },
            "seed_info": ["ID: 22222, Name: proposal"],
            "toolkit": "AfterFirst",
            "parameters": {"entity": "proposal", "after": "2007-03-15", "limit": 100},
            "context": {"times": {"t1": "2007-03-15"}},
            "time_hints": {"after": "2007-03-15"},
            "reasoning": "AfterFirst can handle the after situation。"
        }
    ],
    
    
    "during_between": [
        {
            "subquestion": "What happened between 2005-01-01 and 2005-12-31 in Iraq?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "event", "obj": "Iraq", "time_var": "t1 in between(2005-01-01,2005-12-31)"}],
                "constraints": ["t1 between 2005-01-01 and 2005-12-31"]
            },
            "seed_info": ["ID: 12345, Name: Iraq"],
            "toolkit": "BetweenRange",
            "parameters": {"entity": "Iraq", "between": ["2005-01-01", "2005-12-31"], "limit": 20},
            "context": {},
            "time_hints": {"between": ["2005-01-01", "2005-12-31"]},
            "reasoning": "BetweenRange handle the interval event."
        },
        # new: more between scenarios
        {
            "subquestion": "Who visited Japan during the Olympic Games period between 2008-08-08 and 2008-08-24?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "visit", "obj": "Japan", "time_var": "t1 in between(2008-08-08,2008-08-24)"}],
                "constraints": ["t1 between 2008-08-08 and 2008-08-24"]
            },
            "seed_info": ["ID: 33333, Name: Japan"],
            "toolkit": "BetweenRange",
            "parameters": {"entity": "Japan", "between": ["2008-08-08", "2008-08-24"], "limit": 100},
            "context": {},
            "time_hints": {"between": ["2008-08-08", "2008-08-24"]},
            "reasoning": "BetweenRange suitable for events within a specific period."
        }
    ],
    
    # "direct_connection": [
    #     {
    #         "subquestion": "Is there a direct connection between Japan and Iraq?",
    #         "indicator": {
    #             "edges": [{"subj": "Japan", "rel": "?r", "obj": "Iraq"}],
    #             "constraints": []
    #         },
    #         "seeds": [12345, 67890],
    #         "toolkit": "DirectConnection",
    #         "parameters": {"entity1": "Japan", "entity2": "Iraq", "direction": "both", "limit": 200},
    #         "context": {},
    #         "time_hints": {},
    #         "reasoning": "DirectConnection verify the direct relationship."
    #     },
    #     # new: more direct connection scenarios
    #     {
    #         "subquestion": "What is the relationship between USA and NATO?",
    #         "indicator": {
    #             "edges": [{"subj": "USA", "rel": "?r", "obj": "NATO"}],
    #             "constraints": []
    #         },
    #         "seeds": [44444, 55555],
    #         "toolkit": "DirectConnection",
    #         "parameters": {"entity1": "USA", "entity2": "NATO", "direction": "both", "limit": 150},
    #         "context": {},
    #         "time_hints": {},
    #         "reasoning": "DirectConnection used to find all direct relationships between two entities."
    #     }
    # ],
    
    # "timeline": [
    #     {
    #         "subquestion": "What is the timeline of events for Iraq after 2000-01-01?",
    #         "indicator": {
    #             "edges": [{"subj": "?x", "rel": "?r", "obj": "Iraq", "time_var": "t1"}],
    #             "constraints": ["t1 > 2000-01-01"]
    #         },
    #         "seeds": [12345],
    #         "toolkit": "Timeline",
    #         "parameters": {"entity": "Iraq", "direction": "both", "after": "2000-01-01", "sort_by_time": True, "limit": 100},
    #         "context": {},
    #         "time_hints": {"after": "2000-01-01"},
    #         "reasoning": "Timeline build the time sequence."
    #     },
    #     # 新增：更全面的timeline场景
    #     {
    #         "subquestion": "Show me the diplomatic history of Germany",
    #         "indicator": {
    #             "edges": [{"subj": "Germany", "rel": "?r", "obj": "?x", "time_var": "t1"}],
    #             "constraints": []
    #         },
    #         "seeds": [66666],
    #         "toolkit": "Timeline",
    #         "parameters": {"entity": "Germany", "direction": "both", "sort_by_time": True, "limit": 200},
    #         "context": {},
    #         "time_hints": {},
    #         "reasoning": "Timeline provide a complete time sequence view."
    #     }
    # ],
    
    "first_last": [
        {
            "subquestion": "In which year did Taiwan's Ministry of National Defence and Security last make a request to Japan?",
            "indicator": {
                "edges": [{"subj": "Taiwan's Ministry of National Defence and Security", "rel": "make_request", "obj": "Japan", "time_var": "t1"}],
                "constraints": ["last(t1)"]
            },
            "available_toolkits": ["DirectConnection", "BeforeLast"],
            "available_entities": ["Taiwan's National Defence and Security", "Japan"],

            
            "selected_toolkit": "DirectConnection",
            "parameters": {"entity1": "Taiwan's National Defence and Security", "entity2": "Japan", "before": "2025-01-01","direction": "both", "limit": 200},
            "context": {"times": {"t1": "2025-01-01"}},
            "time_hints": {"before": "2025-01-01"},
            "reasoning": "DirectConnection verify the direct relationship. Time set to latest year 2025-01-01",
        },




        {
            "subquestion": "When did Kitti Wasinondh last express an intention to negotiate with Thailand?",
            "indicator": {
                "edges": [{"subj": "Kitti Wasinondh", "rel": "express_intention_to_negotiate", "obj": "Thailand", "time_var": "t1"}],
                "constraints": ["last(t1)"]
            },
            "available_toolkits": ["BeforeLast", "DirectConnection"],
            "available_entities": ["Kitti Wasinondh", "Thailand"],

            
            "selected_toolkit": "DirectConnection",
            "parameters": {"entity1": "Kitti Wasinondh", "entity2": "Thailand", "before": "2025-01-01", "limit": 100},
            "context": {"times": {"t1": "2025-01-01"}},
            "time_hints": {"before": "2025-01-01"},
            "reasoning": "DirectConnection verify the direct relationship. Time set to latest year 2025-01-01"
        },
        {
            "subquestion": "Who was the first country that Ethiopia expressed optimism about?",
            "indicator": {
                "edges": [{"subj": "Ethiopia", "rel": "express_optimism", "obj": "?x", "time_var": "t1"}],
                "constraints": ["first(t1)"]
            },
            "available_toolkits": ["AfterFirst", "DirectConnection"],
            "available_entities": ["Ethiopia"],

            
            "selected_toolkit": "AfterFirst",
            "parameters": {"entity": "Ethiopia", "after": "1800-01-01", "limit": 100},
            "context": {"times": {"t1": "1800-01-01"}},
            "time_hints": {"after": "1800-01-01"},
            "reasoning": "AfterFirst handle the first event. Time set to earliest year 1800-01-01"
        }
    ],
    "equal_multi":[
        {
            "subquestion": "when did the Cabinet Council of Ministers of Kazakhstan express the intention to negotiate with Japan?",
            "indicator": {
                "edges": [{"subj": "Cabinet Council of Ministers of Kazakhstan", "rel": "express_intention_to_negotiate", "obj": "Japan", "time_var": "t1"}],
                "constraints": ["t2 > t1", "first_after(t2, t1)"]
            },
            "entity_names": ['Japan', 'Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)'],
            "seed_info": ["ID: 62, Name: Japan",
            "ID: 4774, Name: Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)"
            ],

            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)", "entity2": "Japan", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection is suitable as it allows querying the relationship between the Cabinet Council of Ministers of Kazakhstan and Japan, focusing on the intention to negotiate."
        },
        {
            "subquestion": "Who did Ethiopia use conventional military force against on 2005-01-01?",
            "indicator": {
                "edges": [{"subj": "Ethiopia", "rel": "use_conventional_military_force", "obj": "?x", "time_var": "t1 = 2005-01-01"}],
                "constraints": ["t1 = 2005-01-01"]
            },
            "seed_info": ["ID: 67890, Name: Ethiopia"],

            "toolkit": "OneHop",
            "parameters": {"entity": "Ethiopia", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "OneHop is suitable as it allows querying the relationship between Ethiopia and the unknown entity, focusing on the use of conventional military force."
        },

        {
            "subquestion": "When did Japan threaten Taiwan?",
            "indicator": {
                "edges": [{"subj": "Japan", "rel": "threaten", "obj": "Taiwan", "time_var": "t1"}],
                "constraints": []
            },
            "seed_info": ["ID: 67890, Name: Japan","ID: 7788, Name: Taiwan"],
            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Japan", "entity2": "Taiwan", "direction": "both", "limit": 200},

            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection for direct matching."
        },
        # new: cover more when type scenarios
        {
            "subquestion": "In which year was the peace agreement signed?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "sign", "obj": "peace agreement", "time_var": "t1"}],
                "constraints": []
            },
            "seed_info": ["ID: 11111, Name: peace_agreement"],
            "toolkit": "OneHop",
            "parameters": {"entity": "peace_agreement", "direction": "both", "sort_by_time": True, "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "OneHop + sort_by_time for finding the events happened in the specific year."
        },
        {
            "subquestion": "Who wants to negotiate with China on 16 July 2009?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "negotiate", "obj": "China", "time_var": "t1 = 2009-07-16"}],
                "constraints": []
            },
            "seed_info": ["ID: 67890, Name: China"],
            "toolkit": "DayEvents",
            "parameters": {"entity": "China", "day": "2009-07-16", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DayEvents find the events on the specific day."
        },
        {
            "subquestion": "Who wants to negotiate with China in July 2009?",
            "indicator": {
                "edges": [{"subj": "?x", "rel": "negotiate", "obj": "China", "time_var": "t1 = 2009-07"}],
                "constraints": []
            },
            "seed_info": ["ID: 67890, Name: China"],
            "toolkit": "MonthEvents", 
            "parameters": {"entity": "China", "month": "2009-07", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "MonthEvents find the events on the specific month."
        }
    ],
    "general": [
        {
            "subquestion": "when did the Cabinet Council of Ministers of Kazakhstan express the intention to negotiate with Japan?",
            "indicator": {
                "edges": [{"subj": "Cabinet Council of Ministers of Kazakhstan", "rel": "express_intention_to_negotiate", "obj": "Japan", "time_var": "t1"}],
                "constraints": ["t2 > t1", "first_after(t2, t1)"]
            },
            "entity_names": ['Japan', 'Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)'],
            "seed_info": ["ID: 62, Name: Japan",
            "ID: 4774, Name: Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)"
            ],

            "toolkit": "DirectConnection",
            "parameters": {"entity1": "Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)", "entity2": "Japan", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "DirectConnection is suitable as it allows querying the relationship between the Cabinet Council of Ministers of Kazakhstan and Japan, focusing on the intention to negotiate."
        },
        {
            "subquestion": "Who did Ethiopia use conventional military force against on 2005-01-01?",
            "indicator": {
                "edges": [{"subj": "Ethiopia", "rel": "use_conventional_military_force", "obj": "?x", "time_var": "t1 = 2005-01-01"}],
                "constraints": ["t1 = 2005-01-01"]
            },
            "seed_info": ["ID: 67890, Name: Ethiopia"],

            "toolkit": "OneHop",
            "parameters": {"entity": "Ethiopia", "direction": "both", "limit": 200},
            "context": {},
            "time_hints": {},
            "reasoning": "OneHop is suitable as it allows querying the relationship between Ethiopia and the unknown entity, focusing on the use of conventional military force."
        },
        {
            "subquestion": "Before 2006-01-05, which country did Japan threaten last?",
            "indicator": {
                "edges": [{"subj": "Japan", "rel": "threaten", "obj": "?x", "time_var": "t2"}],
                "constraints": ["t2 < 2006-01-05", "last_before(t2, 2006-01-05)"]
            },
            "seed_info": ["ID: 12345, Name: Japan"],
            "toolkit": "BeforeLast",
            "parameters": {"entity": "Japan", "before": "2006-01-05", "limit": 100},

            "context": {"times": {"t1": "2006-01-05"}},
            "time_hints": {"before": "2006-01-05"},
            "reasoning": "BeforeLast get the last event before the specific time."
        }
        # 保留你现有的general示例，它们已经很全面了
    ]
}

        
        # 添加缺失的问题类型映射
        self.template_mapping = {
            "WHEN": "equal",
            "AFTER": "after_first", 
            "BEFORE": "before_last",
            "BETWEEN": "during_between",
            "SAME_DAY": "equal",
            "SAME_MONTH": "equal", 
            "DIRECT": "direct_connection",
            "TIMELINE": "timeline",
            "FIRST_AFTER": "after_first",
            "LAST_BEFORE": "before_last",
            "FIRST_LAST": "first_last",  # add missing mapping
            "GENERAL": "general"
        }
    
    def get_examples_by_question_type(self, qtype: str, max_examples: int = 3) -> List[Dict]:
        """get examples by question type"""
        template_key = self.template_mapping.get(qtype, "general")
        examples = self.examples.get(template_key, [])
        return examples[:max_examples]
    
    def get_examples_by_template_key(self, template_key: str, max_examples: int = 3) -> List[Dict]:
        """get examples by template key"""
        examples = self.examples.get(template_key, [])
        return examples[:max_examples]
    
    def format_examples_for_prompt(self, examples: List[Dict]) -> str:
        """format examples for prompt"""
        if not examples:
            return "No examples available"
        
        formatted_examples = []
        for example in examples:
            formatted = f"""————————————————————————————————————————————————————————————————————————————————
Subquestion: {example.get('subquestion', '')}
Indicator: {json.dumps(example.get('indicator', {}), ensure_ascii=False)}
Seeds: {example.get('seeds', [])}
Selected Toolkit: {example.get('toolkit', '')}
Parameters: {json.dumps(example.get('parameters', {}), ensure_ascii=False)}
Context: {json.dumps(example.get('context', {}), ensure_ascii=False)}
Time Hints: {json.dumps(example.get('time_hints', {}), ensure_ascii=False)}
Reasoning: {example.get('reasoning', '')}
————————————————————————————————————————————————————————————————————————————————"""
            formatted_examples.append(formatted)
        
        return "\n\n".join(formatted_examples)
    
    def add_dynamic_example(self, template_key: str, example: Dict):
        """add dynamic new example (from successful history)"""
        if template_key not in self.examples:
            self.examples[template_key] = []
        
        # limit each type to save at most 10 examples
        if len(self.examples[template_key]) >= 10:
            self.examples[template_key].pop(0)  # remove oldest
        
        self.examples[template_key].append(example)
    
    def get_coverage_report(self) -> Dict[str, int]:
        """get example coverage report"""
        coverage = {}
        for qtype, template_key in self.template_mapping.items():
            example_count = len(self.examples.get(template_key, []))
            coverage[qtype] = example_count
        return coverage
    
    def validate_examples(self) -> Dict[str, List[str]]:
        """validate example completeness"""
        validation_errors = {}
        required_fields = ['subquestion', 'toolkit', 'parameters', 'reasoning']
        
        for template_key, examples in self.examples.items():
            errors = []
            for i, example in enumerate(examples):
                for field in required_fields:
                    if field not in example:
                        errors.append(f"Example {i}: Missing field '{field}'")
                
                # validate toolkit is in supported list
                toolkit = example.get('toolkit', '')
                supported_toolkits = ['OneHop', 'AfterFirst', 'BeforeLast', 'BetweenRange', 
                                    'DayEvents', 'MonthEvents', 'DirectConnection', 'Timeline']
                if toolkit not in supported_toolkits:
                    errors.append(f"Example {i}: Unsupported toolkit '{toolkit}'")
            
            if errors:
                validation_errors[template_key] = errors
        
        return validation_errors

# create global instance
enhanced_examples = EnhancedInitialExamples()

def get_initial_examples_enhanced(template_key: str, max_examples: int = 3) -> List[Dict]:
    """get enhanced initial examples"""
    return enhanced_examples.get_examples_by_template_key(template_key, max_examples)

def get_formatted_examples_enhanced(template_key: str, max_examples: int = 3) -> str:
    """get formatted enhanced examples"""
    examples = get_initial_examples_enhanced(template_key, max_examples)
    return enhanced_examples.format_examples_for_prompt(examples)

def get_examples_by_question_type_enhanced(qtype: str, max_examples: int = 3) -> str:
    """get formatted examples by question type"""
    examples = enhanced_examples.get_examples_by_question_type(qtype, max_examples)
    return enhanced_examples.format_examples_for_prompt(examples)

if __name__ == "__main__":
    # test coverage
    print("Coverage Report:")
    coverage = enhanced_examples.get_coverage_report()
    for qtype, count in coverage.items():
        print(f"  {qtype}: {count} examples")
    
    print("\nValidation Report:")
    errors = enhanced_examples.validate_examples()
    if errors:
        for template_key, error_list in errors.items():
            print(f"  {template_key}: {len(error_list)} errors")
            for error in error_list[:3]:  # only show first 3 errors
                print(f"    - {error}")
    else:
        print("  All examples are valid!")