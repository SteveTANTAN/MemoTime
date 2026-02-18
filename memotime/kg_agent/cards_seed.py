
# =============================
# file: kg_agent/cards_seed.py
# =============================
SEED_TEMPLATES = {
    "after_first.yaml": {
        "workflow_id": "after_first",
        "version": "0.1.0",
        "status": "stable",
        "intent": {
            "short": "Find the earliest subject after anchor",
            "long": "Given an anchor (entity/event/time), find the first subject that satisfies target_relation after the anchor time at the specified time_level. KG-only."
        },
        "inputs": {
            "required": [
                {"name": "anchor_entity", "type": "entity"},
                {"name": "target_relation", "type": "relation"},
                {"name": "time_level", "type": "enum(day|month|year)"}
            ],
            "optional": [
                {"name": "anchor_time", "type": "time|null"},
                {"name": "target_entity_filter", "type": "entity|set|null"},
                {"name": "constraints", "type": "dict|null"}
            ]
        },
        "evidence": {
            "must_have": ["time-stamped triples in KG"],
            "preferred": ["multiple KG paths if available"],
            "indexing": ["entity index", "time buckets by time_level"]
        },
        "core_steps": [
            {"id": "S1", "name": "Locate/Confirm Anchor", "actions": [
                "If anchor_time missing: locate anchor event time via KG at time_level",
                "Normalize to time bucket (day|month|year)"
            ]},
            {"id": "S2", "name": "Collect Candidates", "actions": [
                "Retrieve subjects s.t. target_relation(subject, X)",
                "Attach earliest valid time at time_level"
            ]},
            {"id": "S3", "name": "Temporal Filter", "actions": [
                "Keep candidates with time > anchor_time (same time_level)"
            ]},
            {"id": "S4", "name": "Rank/Select", "actions": [
                "Sort by time asc; tie-break by path reliability",
                "Select K per cardinality"
            ]},
            {"id": "S5", "name": "Compose Answer", "actions": [
                "Emit items with entity, time, path, provenance"
            ]}
        ],
        "verify": {
            "checks": ["temporal monotonicity", "KG path consistency", "time granularity match"]
        },
        "output_schema": {
            "items": [{"entity": "str", "time": "str|null", "path": "list", "provenance": "dict"}],
            "explanations": [],
            "verification": {"passed": "bool", "details": []}
        },
        "heuristics": [
            "Prefer explicit time properties/edges over inferred ordering"
        ],
        "fallbacks": [
            "Coarsen time_level day->month->year",
            "Relax to 'after' then pick earliest from remaining",
            "Widen anchor context if anchor_time missing"
        ],
        "notes": [
            "Cache KG neighborhood to avoid duplicate expansion"
        ],
        "todo": {
            "eval": "Edge-cases if candidate time equals anchor"
        }
    },
    "before_last.yaml": {
        "workflow_id": "before_last",
        "version": "0.1.0",
        "status": "stable",
        "intent": {"short": "Find last before anchor", "long": "Find last subject satisfying relation before anchor time. KG-only."},
        "inputs": {"required": [
            {"name": "anchor_entity", "type": "entity"},
            {"name": "target_relation", "type": "relation"},
            {"name": "time_level", "type": "enum(day|month|year)"}
        ]},
        "evidence": {"must_have": ["time-stamped triples in KG"]},
        "core_steps": [
            {"id": "S1", "name": "Locate/Confirm Anchor", "actions": ["Resolve anchor_time; normalize"]},
            {"id": "S2", "name": "Collect Candidates", "actions": ["Retrieve subjects for relation"]},
            {"id": "S3", "name": "Temporal Filter", "actions": ["Keep time < anchor_time"]},
            {"id": "S4", "name": "Rank/Select", "actions": ["Sort by time desc; pick top-K"]},
            {"id": "S5", "name": "Compose Answer", "actions": ["Emit schema-compliant answer"]}
        ],
        "verify": {"checks": ["temporal monotonicity", "granularity match"]}
    },
    "same_bucket.yaml": {
        "workflow_id": "same_bucket",
        "version": "0.1.0",
        "status": "stable",
        "intent": {"short": "Same day/month/year bucket", "long": "Find subjects sharing the same bucket with anchor at given time_level. KG-only."},
        "inputs": {"required": [
            {"name": "anchor_entity", "type": "entity"},
            {"name": "target_relation", "type": "relation"},
            {"name": "time_level", "type": "enum(day|month|year)"}
        ]},
        "core_steps": [
            {"id": "S1", "name": "Locate Anchor Bucket", "actions": ["Resolve anchor_time; get bucket"]},
            {"id": "S2", "name": "Collect in Bucket", "actions": ["Retrieve candidates with time in same bucket"]},
            {"id": "S3", "name": "Rank", "actions": ["Order by path reliability if needed"]},
            {"id": "S4", "name": "Compose", "actions": ["Emit items"]}
        ],
        "verify": {"checks": ["bucket equality"]}
    },
    "during_between.yaml": {
        "workflow_id": "during_between",
        "version": "0.1.0",
        "status": "stable",
        "intent": {"short": "Within period", "long": "Return subjects satisfying relation within [t1, t2]. KG-only."},
        "inputs": {"required": [
            {"name": "period", "type": "interval"},
            {"name": "target_relation", "type": "relation"},
            {"name": "time_level", "type": "enum(day|month|year)"}
        ]},
        "core_steps": [
            {"id": "S1", "name": "Normalize Period", "actions": ["Align to time_level buckets"]},
            {"id": "S2", "name": "Collect", "actions": ["Retrieve candidates with time ∈ [t1,t2]"]},
            {"id": "S3", "name": "Rank", "actions": ["Order by time asc or reliability"]},
            {"id": "S4", "name": "Compose", "actions": ["Emit list or top-K"]}
        ],
        "verify": {"checks": ["period inclusion", "granularity match"]}
    }
}
