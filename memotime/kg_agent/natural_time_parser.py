#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Natural Time Parser for TPKG System
natural language time format parser, convert various time expressions to standard ISO format
"""

import re
import calendar
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union, Tuple
import json

# import existing time parsing function
from kg_agent.temporal_kg_toolkit import parse_time_to_range

class NaturalTimeParser:
    """natural language time parser"""
    
    def __init__(self):
        # month mapping
        self.month_names = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        
        # common time format regex expressions
        self.patterns = [
            # DD Month YYYY (e.g., "22 October 2008")
            (r'(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})', self._parse_dd_month_yyyy),
            # Month DD, YYYY (e.g., "October 22, 2008")
            (r'([a-zA-Z]+)\s+(\d{1,2}),?\s+(\d{4})', self._parse_month_dd_yyyy),
            # Month YYYY (e.g., "October 2008")
            (r'([a-zA-Z]+)\s+(\d{4})', self._parse_month_yyyy),
            # DD/MM/YYYY or MM/DD/YYYY
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', self._parse_slash_date),
            # DD-MM-YYYY or MM-DD-YYYY
            (r'(\d{1,2})-(\d{1,2})-(\d{4})', self._parse_dash_date)
        ]
        
        # LLM fallback flag
        self.use_llm_fallback = True
        
    def parse_natural_time(self, time_str: str) -> Optional[str]:
        """
        parse natural language time format, return ISO format string
        
        Args:
            time_str: natural language time string
            
        Returns:
            ISO format time string, if parsing fails return None
        """
        if not time_str or not isinstance(time_str, str):
            return None
            
        time_str = time_str.strip()
        
        # first try existing ISO format parsing
        try:
            parse_time_to_range(time_str)
            return time_str  # already standard format
        except ValueError:
            pass
        
        # try various natural language formats
        for pattern, parser_func in self.patterns:
            match = re.search(pattern, time_str, re.IGNORECASE)
            if match:
                try:
                    iso_date = parser_func(match)
                    if iso_date:
                        return iso_date
                except Exception as e:
                    print(f"⚠️ parse time format failed: {e}")
                    continue
        
        # if all patterns fail, try LLM parsing
        if self.use_llm_fallback:
            return self._llm_parse_time(time_str)
        
        return None
    
    def _parse_dd_month_yyyy(self, match) -> Optional[str]:
        """parse DD Month YYYY format"""
        day, month_name, year = match.groups()
        month = self.month_names.get(month_name.lower())
        if not month:
            return None
        
        try:
            # validate date validity
            datetime(int(year), month, int(day))
            return f"{year}-{month:02d}-{int(day):02d}"
        except ValueError:
            return None
    
    def _parse_month_dd_yyyy(self, match) -> Optional[str]:
        """parse Month DD, YYYY format"""
        month_name, day, year = match.groups()
        month = self.month_names.get(month_name.lower())
        if not month:
            return None
        
        try:
            # validate date validity
            datetime(int(year), month, int(day))
            return f"{year}-{month:02d}-{int(day):02d}"
        except ValueError:
            return None
    
    def _parse_month_yyyy(self, match) -> Optional[str]:
        """parse Month YYYY format"""
        month_name, year = match.groups()
        month = self.month_names.get(month_name.lower())
        if not month:
            return None
        
        try:
            # return month format
            datetime(int(year), month, 1)
            return f"{year}-{month:02d}"
        except ValueError:
            return None
    
    def _parse_slash_date(self, match) -> Optional[str]:
        """parse DD/MM/YYYY or MM/DD/YYYY format"""
        part1, part2, year = match.groups()
        
        # try two interpretations: DD/MM/YYYY and MM/DD/YYYY
        for day, month in [(part1, part2), (part2, part1)]:
            try:
                if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
                    datetime(int(year), int(month), int(day))
                    return f"{year}-{int(month):02d}-{int(day):02d}"
            except ValueError:
                continue
        
        return None
    
    def _parse_dash_date(self, match) -> Optional[str]:
        """parse DD-MM-YYYY or MM-DD-YYYY format"""
        return self._parse_slash_date(match)  # logic same
    
    def _parse_chinese_date(self, match) -> Optional[str]:
        """parse Chinese date format YYYY-MM-DD"""
        year, month, day = match.groups()
        try:
            datetime(int(year), int(month), int(day))
            return f"{year}-{int(month):02d}-{int(day):02d}"
        except ValueError:
            return None
    
    def _parse_chinese_month(self, match) -> Optional[str]:
        """parse Chinese month format YYYY-MM"""
        year, month = match.groups()
        try:
            datetime(int(year), int(month), 1)
            return f"{year}-{int(month):02d}"
        except ValueError:
            return None
    
    def _parse_chinese_year(self, match) -> Optional[str]:
        """parse Chinese year format YYYY"""
        year = match.groups()[0]
        try:
            datetime(int(year), 1, 1)
            return year
        except ValueError:
            return None
    
    def _llm_parse_time(self, time_str: str) -> Optional[str]:
        """use LLM to parse complex time expressions"""
        try:
            # import LLM module
            from kg_agent.llm import LLM
            
            system_prompt = "You are a time format converter. Convert natural language time expressions to ISO format."
            prompt = f"""
Convert the following time expression to ISO format. Only return the converted time, no other explanation.

Supported ISO formats:
- YYYY-MM-DD (specific date)
- YYYY-MM (month)
- YYYY (year)

Time expression: "{time_str}"

If you cannot parse it, please return "UNPARSEABLE"
"""
            
            response = LLM.call(system_prompt, prompt)
            if response and response.strip() != "UNPARSEABLE":
                # validate LLM returned format
                iso_time = response.strip()
                try:
                    parse_time_to_range(iso_time)
                    return iso_time
                except ValueError:
                    pass
        
        except Exception as e:
            print(f"⚠️ LLM time parsing failed: {e}")
        
        return None
    
    def extract_time_constraints_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        extract time constraints information from text
        
        Args:
            text: text containing time information
            
        Returns:
            time constraints list, each constraint contains type and value information
        """
        constraints = []
        
        # find "before" constraint
        before_patterns = [
            r'before\s+([^,\.\?!]+?)(?:\s*,|\s*\.|$|\?|!)',
            r'prior\s+to\s+([^,\.\?!]+?)(?:\s*,|\s*\.|$|\?|!)',
            r'earlier\s+than\s+([^,\.\?!]+?)(?:\s*,|\s*\.|$|\?|!)'
        ]
        
        for pattern in before_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                time_expr = match.group(1).strip()
                iso_time = self.parse_natural_time(time_expr)
                if iso_time:
                    constraints.append({
                        'type': 'before',
                        'original': time_expr,
                        'iso_format': iso_time,
                        'constraint': f't1 < {iso_time}'
                    })
        
        # find "after" constraint
        after_patterns = [
            r'after\s+([^,\.\?!]+?)(?:\s*,|\s*\.|$|\?|!)',
            r'following\s+([^,\.\?!]+?)(?:\s*,|\s*\.|$|\?|!)',
            r'later\s+than\s+([^,\.\?!]+?)(?:\s*,|\s*\.|$|\?|!)'
        ]
        
        for pattern in after_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                time_expr = match.group(1).strip()
                iso_time = self.parse_natural_time(time_expr)
                if iso_time:
                    constraints.append({
                        'type': 'after',
                        'original': time_expr,
                        'iso_format': iso_time,
                        'constraint': f't1 > {iso_time}'
                    })
        
        # find "on" or "in" constraint (exact time)
        exact_patterns = [
            r'on\s+([^,\.\?!]+?)(?:\s*,|\s*\.|$|\?|!)',
            r'in\s+([^,\.\?!]+?)(?:\s*,|\s*\.|$|\?|!)',
            r'during\s+([^,\.\?!]+?)(?:\s*,|\s*\.|$|\?|!)'
        ]
        
        for pattern in exact_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                time_expr = match.group(1).strip()
                iso_time = self.parse_natural_time(time_expr)
                if iso_time:
                    constraints.append({
                        'type': 'exact',
                        'original': time_expr,
                        'iso_format': iso_time,
                        'constraint': f't1 = {iso_time}'
                    })
        
        return constraints

# global instance
_time_parser = None

def get_time_parser() -> NaturalTimeParser:
    """get time parser instance (singleton pattern)"""
    global _time_parser
    if _time_parser is None:
        _time_parser = NaturalTimeParser()
    return _time_parser

def parse_natural_time_to_iso(time_str: str) -> Optional[str]:
    """
    convenient function: convert natural language time to ISO format
    
    Args:
        time_str: natural language time string
        
    Returns:
        ISO format time string, if fails return None
    """
    parser = get_time_parser()
    return parser.parse_natural_time(time_str)

def extract_time_constraints(text: str) -> List[Dict[str, Any]]:
    """
        convenient function: extract time constraints from text
    
    Args:
        text: text containing time information
        
    Returns:
        time constraints list
    """
    parser = get_time_parser()
    return parser.extract_time_constraints_from_text(text)
