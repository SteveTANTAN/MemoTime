#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: kg_agent/toolkit_selector.py
"""
Toolkit selector
Automatically select the appropriate toolkit based on the subquestion type and time requirements
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .decompose import SubQuestion
from .temporal_kg_toolkit import TemporalKGToolkit

@dataclass
class ToolkitConfig:
    """Toolkit configuration"""
    toolkit_name: str
    method_name: str
    parameters: Dict[str, Any]
    description: str

class ToolkitSelector:
    """Toolkit selector"""
    
    def __init__(self, toolkit: TemporalKGToolkit):
        self.toolkit = toolkit
    
    def select_toolkit_for_subquestion(self, subq: SubQuestion, ctx: Dict[str, Any]) -> ToolkitConfig:
        """
        Select the appropriate toolkit based on the subquestion type and time requirements
        """
        # Analyze the constraints of the subquestion
        constraints = subq.indicator.constraints
        time_vars = self._extract_time_vars(subq)
        
        # Select the toolkit based on the constraint type
        if self._has_first_constraint(constraints):
            return self._select_first_toolkit(subq, ctx)
        elif self._has_last_constraint(constraints):
            return self._select_last_toolkit(subq, ctx)
        elif self._has_after_constraint(constraints):
            return self._select_after_toolkit(subq, ctx)
        elif self._has_before_constraint(constraints):
            return self._select_before_toolkit(subq, ctx)
        elif self._has_same_time_constraint(constraints):
            return self._select_same_time_toolkit(subq, ctx)
        elif self._has_equal_constraint(constraints):
            return self._select_equal_toolkit(subq, ctx)
        else:
            # Use basic retrieval by default
            return self._select_basic_toolkit(subq, ctx)
    
    def _extract_time_vars(self, subq: SubQuestion) -> List[str]:
        """Extract time variables"""
        time_vars = []
        for edge in subq.indicator.edges:
            if edge.time_var and edge.time_var not in time_vars:
                time_vars.append(edge.time_var)
        return time_vars
    
    def _has_first_constraint(self, constraints: List[str]) -> bool:
        """Check if there is a first constraint"""
        return any("first" in constraint.lower() for constraint in constraints)
    
    def _has_last_constraint(self, constraints: List[str]) -> bool:
        """Check if there is a last constraint"""
        return any("last" in constraint.lower() for constraint in constraints)
    
    def _has_after_constraint(self, constraints: List[str]) -> bool:
        """Check if there is an after constraint"""
        return any("after" in constraint.lower() or ">" in constraint for constraint in constraints)
    
    def _has_before_constraint(self, constraints: List[str]) -> bool:
        """Check if there is a before constraint"""
        return any("before" in constraint.lower() or "<" in constraint for constraint in constraints)
    
    def _has_same_time_constraint(self, constraints: List[str]) -> bool:
        """Check if there is a same_time constraint"""
        return any("same" in constraint.lower() for constraint in constraints)
    
    def _has_equal_constraint(self, constraints: List[str]) -> bool:
        """Check if there is an equal constraint"""
        return any("=" in constraint for constraint in constraints)
    
    def _select_first_toolkit(self, subq: SubQuestion, ctx: Dict[str, Any]) -> ToolkitConfig:
        """Select the first toolkit"""
        # Extract entity information
        entities = self._extract_entities_from_edges(subq.indicator.edges)
        
        # Check if it is a first_last type (no after constraint)
        if not self._has_after_constraint(subq.indicator.constraints):
            return ToolkitConfig(
                toolkit_name="temporal_kg_toolkit",
                method_name="find_first_last",
                parameters={
                    "entity": entities.get("subject"),
                    "relation": entities.get("relation"),
                    "target": entities.get("object")
                },
                description="Find the first or last occurrence of an event for an entity"
            )
        else:
            return ToolkitConfig(
                toolkit_name="temporal_kg_toolkit",
                method_name="find_after_first",
                parameters={
                    "entity": entities.get("subject"),
                    "relation": entities.get("relation"),
                    "target": entities.get("object"),
                    "reference_time": self._get_reference_time(ctx, subq)
                },
                description="Find the first entity to perform an action after a reference time"
            )
    
    def _select_last_toolkit(self, subq: SubQuestion, ctx: Dict[str, Any]) -> ToolkitConfig:
        """Select the last toolkit"""
        entities = self._extract_entities_from_edges(subq.indicator.edges)
        
        return ToolkitConfig(
            toolkit_name="temporal_kg_toolkit",
            method_name="find_before_last",
            parameters={
                "entity": entities.get("subject"),
                "relation": entities.get("relation"),
                "target": entities.get("object"),
                "reference_time": self._get_reference_time(ctx, subq)
            },
            description="Find the last entity to perform an action before a reference time"
        )
    
    def _select_after_toolkit(self, subq: SubQuestion, ctx: Dict[str, Any]) -> ToolkitConfig:
        """Select the after toolkit"""
        entities = self._extract_entities_from_edges(subq.indicator.edges)
        
        return ToolkitConfig(
            toolkit_name="temporal_kg_toolkit",
            method_name="find_after_first",
            parameters={
                "entity": entities.get("subject"),
                "relation": entities.get("relation"),
                "target": entities.get("object"),
                "reference_time": self._get_reference_time(ctx, subq)
            },
            description="Find entities after a reference time"
        )
    
    def _select_before_toolkit(self, subq: SubQuestion, ctx: Dict[str, Any]) -> ToolkitConfig:
        """Select the before toolkit"""
        entities = self._extract_entities_from_edges(subq.indicator.edges)
        
        return ToolkitConfig(
            toolkit_name="temporal_kg_toolkit",
            method_name="find_before_last",
            parameters={
                "entity": entities.get("subject"),
                "relation": entities.get("relation"),
                "target": entities.get("object"),
                "reference_time": self._get_reference_time(ctx, subq)
            },
            description="Find entities before a reference time"
        )
    
    def _select_same_time_toolkit(self, subq: SubQuestion, ctx: Dict[str, Any]) -> ToolkitConfig:
        """Select the same_time toolkit"""
        entities = self._extract_entities_from_edges(subq.indicator.edges)
        
        return ToolkitConfig(
            toolkit_name="temporal_kg_toolkit",
            method_name="find_same_time_events",
            parameters={
                "entity": entities.get("subject"),
                "relation": entities.get("relation"),
                "target": entities.get("object"),
                "reference_time": self._get_reference_time(ctx, subq)
            },
            description="Find events at the same time"
        )
    
    def _select_equal_toolkit(self, subq: SubQuestion, ctx: Dict[str, Any]) -> ToolkitConfig:
        """Select the equal toolkit"""
        entities = self._extract_entities_from_edges(subq.indicator.edges)
        time_constraint = self._extract_time_constraint(subq.indicator.constraints)
        
        return ToolkitConfig(
            toolkit_name="temporal_kg_toolkit",
            method_name="events_on_day",
            parameters={
                "entity": entities.get("subject"),
                "relation": entities.get("relation"),
                "target": entities.get("object"),
                "time": time_constraint
            },
            description="Find events at a specific time"
        )
    
    def _select_basic_toolkit(self, subq: SubQuestion, ctx: Dict[str, Any]) -> ToolkitConfig:
        """Select the basic toolkit"""
        entities = self._extract_entities_from_edges(subq.indicator.edges)
        
        return ToolkitConfig(
            toolkit_name="temporal_kg_toolkit",
            method_name="retrieve_one_hop",
            parameters={
                "entity": entities.get("subject"),
                "relation": entities.get("relation"),
                "target": entities.get("object")
            },
            description="Basic one-hop retrieval"
        )
    
    def _extract_entities_from_edges(self, edges: List) -> Dict[str, str]:
        """Extract entity information from edges"""
        entities = {"subject": None, "relation": None, "object": None}
        
        if edges:
            edge = edges[0]  # Take the first edge
            entities["subject"] = edge.subj if edge.subj != "?x" and edge.subj != "?y" else None
            entities["relation"] = edge.rel
            entities["object"] = edge.obj if edge.obj != "?x" and edge.obj != "?y" else None
        
        return entities
    
    def _get_reference_time(self, ctx: Dict[str, Any], subq: SubQuestion) -> Optional[str]:
        """Get the reference time"""
        # Get the time of the previous answer from the context
        if ctx.get("answers"):
            for answer in ctx["answers"].values():
                if answer.get("time"):
                    return answer["time"]
        return None
    
    def _extract_time_constraint(self, constraints: List[str]) -> Optional[str]:
        """Extract time constraints"""
        for constraint in constraints:
            if "=" in constraint:
                # Extract the time value
                parts = constraint.split("=")
                if len(parts) == 2:
                    time_value = parts[1].strip()
                    # Try to convert natural language time format
                    try:
                        from kg_agent.natural_time_parser import parse_natural_time_to_iso
                        iso_time = parse_natural_time_to_iso(time_value)
                        if iso_time:
                            print(f"✅ Constraint time conversion: '{time_value}' -> '{iso_time}'")
                            return iso_time
                    except Exception as e:
                        print(f"⚠️ Constraint time conversion failed: {e}")
                    return time_value
            elif "<" in constraint or ">" in constraint:
                # Handle comparative constraints, e.g. "t1 < 22 October 2008"
                import re
                # Match the time value
                time_match = re.search(r'[<>]\s*(.+)$', constraint)
                if time_match:
                    time_value = time_match.group(1).strip()
                    try:
                        from kg_agent.natural_time_parser import parse_natural_time_to_iso
                        iso_time = parse_natural_time_to_iso(time_value)
                        if iso_time:
                            print(f"✅ Constraint time conversion: '{time_value}' -> '{iso_time}'")
                            return iso_time
                    except Exception as e:
                        print(f"⚠️ Constraint time conversion failed: {e}")
                    return time_value
        return None

# Toolkit method mapping
TOOLKIT_METHODS = {
    "find_after_first": "find_after_first",
    "find_before_last": "find_before_last", 
    "find_same_time_events": "find_same_time_events",
    "events_on_day": "events_on_day",
    "events_in_month": "events_in_month",
    "retrieve_one_hop": "retrieve_one_hop"
}
