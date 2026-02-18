#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Answer Verifier for TPKG System
Answer Verifier for TPKG System
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class AnswerVerifier:
    """Answer Verifier"""
    
    def __init__(self, golden_answers_path: str = None):
        """
        Initialize Answer Verifier
        
        Args:
            golden_answers_path: golden answers file path
                                If None, automatically use current dataset's test data path
        """
        # Dynamic path based on current dataset
        if golden_answers_path is None:
            try:
                from config import TPKGConfig
            except:
                try:
                    from ..config import TPKGConfig
                except:
                    # Fallback: use relative path from this file
                    from pathlib import Path
                    project_root = Path(__file__).parent.parent.parent.absolute()
                    golden_answers_path = str(project_root / "Data" / "MultiTQ" / "questions_with_candidates_multitq.json")
                    TPKGConfig = None
            
            if TPKGConfig:
                golden_answers_path = TPKGConfig.get_test_data_path()
                print(f"📁 AnswerVerifier auto-loaded test data: {golden_answers_path}")
        
        self.golden_answers_path = golden_answers_path
        self.golden_answers = self._load_golden_answers()
    
    def _load_golden_answers(self) -> Dict[int, Dict[str, Any]]:
        """Load golden answers (supports MultiTQ and TimeQuestions formats)"""
        try:
            with open(self.golden_answers_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            golden_dict = {}
            
            # Detect data format
            if data and isinstance(data, list):
                first_item = data[0]
                
                # TimeQuestions format detection
                if 'Id' in first_item and 'Answer' in first_item:
                    print("📊 Detected TimeQuestions format")
                    for item in data:
                        qid = item.get('Id')
                        if qid is not None:
                            # Extract answers
                            answers = []
                            answer_list = item.get('Answer', [])
                            answer_type = 'entity'  # default
                            
                            for ans in answer_list:
                                if isinstance(ans, dict):
                                    ans_type = ans.get('AnswerType', 'Entity')
                                    if ans_type == 'Value':
                                        # Time answer
                                        answer_type = 'time'
                                        arg = ans.get('AnswerArgument', '')
                                        # Extract year or date
                                        if arg:
                                            # Format: "1994-03-01T00:00:00Z"
                                            if 'T' in arg:
                                                date_part = arg.split('T')[0]
                                                answers.append(date_part)  # "1994-03-01"
                                                # Also add year-only version
                                                year = date_part.split('-')[0]
                                                if year not in answers:
                                                    answers.append(year)  # "1994"
                                            else:
                                                answers.append(arg)
                                    elif ans_type == 'Entity':
                                        # Entity answer
                                        answer_type = 'entity'
                                        label = ans.get('WikidataLabel', '')
                                        if label:
                                            answers.append(label)
                            
                            golden_dict[qid] = {
                                'answers': answers,
                                'answer_type': answer_type,
                                'question': item.get('Question', ''),
                                'qtype': item.get('Temporal question type', ['unknown'])[0] if isinstance(item.get('Temporal question type'), list) else 'unknown'
                            }
                
                # MultiTQ format
                elif 'quid' in first_item:
                    print("📊 Detected MultiTQ format")
                    for item in data:
                        quid = item.get('quid')
                        if quid is not None:
                            golden_dict[quid] = {
                                'answers': item.get('answers', []),
                                'answer_type': item.get('answer_type', 'entity'),
                                'question': item.get('question', ''),
                                'qtype': item.get('qtype', 'unknown')
                            }
            
            print(f"✅ Loaded {len(golden_dict)} golden answers")
            return golden_dict
        except Exception as e:
            print(f"❌ Failed to load golden answers: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    @staticmethod
    def _normalize_entity(entity: str) -> str:
        """
        Normalize entity name: remove symbols, convert to lowercase
        
        Args:
            entity: original entity name
        
        Returns:
            Normalized entity name
        """
        if not entity:
            return ""
        
        # Remove all symbols: underscore, slash, quote, parenthesis, etc.
        # Keep letters, numbers and spaces
        normalized = re.sub(r'[_/\'"()\[\]{}<>.,;:!?@#$%^&*+=|\\~`-]', ' ', entity)
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    @staticmethod
    def _normalize_time(time_str: str) -> Tuple[Optional[str], str]:
        """
        Normalize time format
        
        Args:
            time_str: original time string
        
        Returns:
            (normalized time string, time granularity) e.g. ("2000-01", "month")
        """
        if not time_str:
            return None, "unknown"
        
        time_str = str(time_str).strip()
        
        # Try matching different time formats
        #  YYYY-MM-DD
        match = re.match(r'(\d{4})-(\d{2})-(\d{2})', time_str)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}", "day"
        
        #  YYYY-MM
        match = re.match(r'(\d{4})-(\d{2})', time_str)
        if match:
            year, month = match.groups()
            return f"{year}-{month}", "month"
        
        # YYYY
        match = re.match(r'(\d{4})', time_str)
        if match:
            year = match.group(1)
            return year, "year"
        
        # (January, Feb, etc.)
        month_names = {
            'january': '01', 'jan': '01',
            'february': '02', 'feb': '02',
            'march': '03', 'mar': '03',
            'april': '04', 'apr': '04',
            'may': '05',
            'june': '06', 'jun': '06',
            'july': '07', 'jul': '07',
            'august': '08', 'aug': '08',
            'september': '09', 'sep': '09', 'sept': '09',
            'october': '10', 'oct': '10',
            'november': '11', 'nov': '11',
            'december': '12', 'dec': '12'
        }
        
        time_lower = time_str.lower()
        for month_name, month_num in month_names.items():
            if month_name in time_lower:
                # Try to extract year
                year_match = re.search(r'\d{4}', time_str)
                if year_match:
                    year = year_match.group()
                    return f"{year}-{month_num}", "month"
                else:
                    return month_num, "month"
        
        # Cannot recognize format, return original string
        return time_str, "unknown"
    
    @staticmethod
    def _time_matches(predicted: str, golden: str) -> bool:

        pred_norm, pred_level = AnswerVerifier._normalize_time(predicted)
        gold_norm, gold_level = AnswerVerifier._normalize_time(golden)
        
        if not pred_norm or not gold_norm:
            return False
        
        if pred_norm == gold_norm:
            return True
        

        if pred_norm.startswith(gold_norm) or gold_norm.startswith(pred_norm):
            return True
        
        return False
    
    @staticmethod
    def _extract_answer(answer_text: str) -> List[str]:

        if not answer_text:
            return []
        
        main_answer_part = answer_text
        if 'answer is' in answer_text.lower():
            match = re.search(r'(?:So )?[Tt]he answer is[:\s]+(.+)', answer_text, re.DOTALL)
            if match:
                main_answer_part = match.group(1)
        
        main_answer_part = re.sub(r'\s+on\s+\d{4}[-/]\d{2}[-/]\d{2}[^\s,]*', '', main_answer_part)
        
        parts = re.split(r'\s*,\s*(?:or\s+)?|\s+or\s+', main_answer_part)
        
        candidates = []
        seen = set()  
        
        for part in parts:
            cleaned = part.strip()
            
            if re.match(r'^on\s+\d{4}', cleaned, re.IGNORECASE):
                continue
            
            cleaned = re.sub(r'\s*\[.*?\]\s*', ' ', cleaned).strip()
            
            # Remove date/explanation in parentheses (but keep parentheses in entity name)
            # e.g. keep "Citizen (North Korea)", but remove "Japan (2005-11-18)"
            # Strategy: if parentheses contain date format, delete
            cleaned = re.sub(r'\s*\(\d{4}[-/]\d{2}[-/]\d{2}\)\s*', '', cleaned)
            cleaned = re.sub(r'\s*\(\d{4}[-/]\d{2}\)\s*', '', cleaned)
            cleaned = re.sub(r'\s*\(\d{4}\)\s*', '', cleaned)
            
            # Remove "on YYYY-MM-DD"
            cleaned = re.sub(r'\s+on\s+\d{4}[-/]\d{2}[-/]\d{2}.*$', '', cleaned)
            
            # Remove extra spaces
            cleaned = ' '.join(cleaned.split())
            
            # Normalize for deduplication (lowercase + remove underscore)
            normalized = cleaned.lower().replace('_', ' ')
            
            if cleaned and cleaned.lower() not in ['or', 'and', 'the'] and normalized not in seen:
                candidates.append(cleaned)
                seen.add(normalized)
        
        # If no candidates extracted, return the cleaned version of the original answer
        if not candidates:
            cleaned = answer_text.strip()
            cleaned = cleaned.split(',')[0].strip()
            if cleaned:
                candidates.append(cleaned)
        
        return candidates if candidates else [answer_text.strip()]
    
    def normalize_answer_advanced(self, answer: str) -> str:
        """
        Advanced answer normalization (from verify_accuracy_universal.py)
        Handles various formats and edge cases more flexibly
        """
        if not answer:
            return ""
        
        # Remove common prefixes
        prefixes_to_remove = [
            "So the answer is:",
            "The answer is:",
            "Answer:",
            "Answer is:",
            "Answer:"
        ]
        
        normalized = answer.strip()
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        # Handle "or [" cases but keep bracket content
        if "or [" in normalized:
            or_bracket_pos = normalized.find("or [")
            if or_bracket_pos != -1:
                before_or = normalized[:or_bracket_pos].strip()
                bracket_start = normalized.find("[", or_bracket_pos)
                bracket_end = normalized.find("]", bracket_start)
                if bracket_start != -1 and bracket_end != -1:
                    bracket_content = normalized[bracket_start+1:bracket_end]
                    if before_or and bracket_content:
                        if before_or in bracket_content:
                            normalized = bracket_content
                        else:
                            normalized = f"{before_or} {bracket_content}".strip()
                    elif bracket_content:
                        normalized = bracket_content
                    else:
                        normalized = before_or
                else:
                    normalized = before_or
        
        # Remove "or No answer provided" suffix
        if "or No answer provided" in normalized:
            normalized = normalized.split("or No answer provided")[0].strip()
        
        # Remove duplicates
        if normalized:
            parts = normalized.split(',')
            unique_parts = []
            for part in parts:
                part = part.strip()
                if part and part not in unique_parts:
                    unique_parts.append(part)
            normalized = ', '.join(unique_parts)
        
        # Keep longest entity if multiple entities
        if normalized and ',' in normalized:
            parts = [part.strip() for part in normalized.split(',')]
            longest_part = max(parts, key=len)
            normalized = longest_part
        
        # Remove date suffix "on 20"
        if "on 20" in normalized:
            normalized = normalized.split("on 20")[0].strip()
        
        # Handle bracket content
        if "(" in normalized and ")" in normalized:
            bracket_content = normalized[normalized.find("(")+1:normalized.find(")")]
            main_part = normalized.split("(")[0].strip()
            
            if main_part and bracket_content:
                normalized = f"{main_part} {bracket_content}"
            elif bracket_content:
                normalized = bracket_content
            else:
                normalized = main_part
        
        # Remove punctuation and extra spaces
        normalized = normalized.replace(",", "").replace(".", "").replace("!", "").replace("?", "")
        normalized = normalized.replace("_", " ")
        normalized = normalized.replace("/", " ")
        normalized = " ".join(normalized.split())
        
        return normalized.lower()
    
    @staticmethod
    def _is_time_format_match(predicted: str, golden: str) -> bool:
        """Check if time formats match (from verify_accuracy_universal.py)"""
        import re
        
        # Extract year and month
        pred_year_month = re.search(r'(\d{4})[-\s]*(\d{1,2})', predicted)
        golden_year_month = re.search(r'(\d{4})[-\s]*(\d{1,2})', golden)
        
        if pred_year_month and golden_year_month:
            pred_ym = f"{pred_year_month.group(1)}-{pred_year_month.group(2).zfill(2)}"
            golden_ym = f"{golden_year_month.group(1)}-{golden_year_month.group(2).zfill(2)}"
            return pred_ym == golden_ym
        
        # Check month name matching
        month_names = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        for month_name, month_num in month_names.items():
            if month_name in predicted and month_num in golden:
                return True
        
        return False
    
    @staticmethod
    def _is_multi_answer_match(predicted: str, golden: str) -> bool:
        """Check multi-answer matching (from verify_accuracy_universal.py)"""
        if ',' in golden:
            golden_parts = [part.strip() for part in golden.split(',')]
            for part in golden_parts:
                if predicted in part or part in predicted:
                    return True
        
        if ',' in predicted:
            pred_parts = [part.strip() for part in predicted.split(',')]
            for part in pred_parts:
                if part in golden or golden in part:
                    return True
        
        return False
    
    @staticmethod
    def _is_semantic_match(predicted: str, golden: str) -> bool:
        """Check semantic matching (from verify_accuracy_universal.py)"""
        # Remove common modifiers
        pred_clean = predicted.replace('(china)', '').replace('(japan)', '').replace('(thailand)', '').replace('(south korea)', '').strip()
        golden_clean = golden.replace('(china)', '').replace('(japan)', '').replace('(thailand)', '').replace('(south korea)', '').strip()
        
        if pred_clean and golden_clean:
            if pred_clean in golden_clean or golden_clean in pred_clean:
                return True
        
        # Check word overlap
        pred_words = set(pred_clean.split())
        golden_words = set(golden_clean.split())
        
        if pred_words and golden_words:
            overlap = len(pred_words & golden_words)
            total_unique = len(pred_words | golden_words)
            if total_unique > 0 and overlap / total_unique > 0.5:
                return True
        
        return False
    
    def is_answer_correct_flexible(self, quid: int, predicted_answer: str, 
                                   matching_strategies: List[str] = None) -> Tuple[bool, str]:
        """
        Flexible answer verification with multiple matching strategies
        (Integrated from verify_accuracy_universal.py and verify_accuracy.py)
        
        Args:
            quid: question ID  
            predicted_answer: predicted answer
            matching_strategies: list of matching strategies to use
                Available strategies:
                - 'exact': exact matching
                - 'contain': substring matching
                - 'advanced_normalize': advanced normalization + matching
                - 'time_format': time format matching
                - 'multi_answer': multi-answer matching
                - 'semantic': semantic matching
                - 'loose': very loose matching (remove all spaces/underscores)
                Default: all strategies
        
        Returns:
            (is_correct, match_details)
        """
        if matching_strategies is None:
            matching_strategies = ['exact', 'contain', 'advanced_normalize', 
                                  'time_format', 'multi_answer', 'semantic', 'loose']
        
        # Get golden answers
        if quid not in self.golden_answers:
            return False, "No golden answer found"
        
        golden_answers = self.golden_answers[quid]['answers']
        if not golden_answers:
            return False, "Empty golden answers"
        
        if not predicted_answer or not predicted_answer.strip():
            return False, "Empty predicted answer"
        
        # Try each matching strategy
        for golden in golden_answers:
            for strategy in matching_strategies:
                if strategy == 'exact':
                    # Exact matching
                    pred_norm = self._normalize_entity(predicted_answer)
                    gold_norm = self._normalize_entity(golden)
                    if pred_norm == gold_norm:
                        return True, f"Exact match: '{pred_norm}' == '{gold_norm}'"
                
                elif strategy == 'contain':
                    # Substring matching
                    pred_norm = self._normalize_entity(predicted_answer)
                    gold_norm = self._normalize_entity(golden)
                    if gold_norm in pred_norm or pred_norm in gold_norm:
                        return True, f"Substring match: '{pred_norm}' contains '{gold_norm}'"
                
                elif strategy == 'advanced_normalize':
                    # Advanced normalization
                    pred_adv = self.normalize_answer_advanced(predicted_answer)
                    gold_adv = self.normalize_answer_advanced(golden)
                    if pred_adv == gold_adv:
                        return True, f"Advanced normalized match: '{pred_adv}' == '{gold_adv}'"
                    if gold_adv in pred_adv or pred_adv in gold_adv:
                        return True, f"Advanced normalized substring: '{pred_adv}' contains '{gold_adv}'"
                
                elif strategy == 'time_format':
                    # Time format matching
                    pred_norm = self._normalize_entity(predicted_answer)
                    gold_norm = self._normalize_entity(golden)
                    if self._is_time_format_match(pred_norm, gold_norm):
                        return True, f"Time format match: '{pred_norm}' matches '{gold_norm}'"
                
                elif strategy == 'multi_answer':
                    # Multi-answer matching
                    pred_norm = self._normalize_entity(predicted_answer)
                    gold_norm = self._normalize_entity(golden)
                    if self._is_multi_answer_match(pred_norm, gold_norm):
                        return True, f"Multi-answer match: '{pred_norm}' matches '{gold_norm}'"
                
                elif strategy == 'semantic':
                    # Semantic matching
                    pred_norm = self._normalize_entity(predicted_answer)
                    gold_norm = self._normalize_entity(golden)
                    if self._is_semantic_match(pred_norm, gold_norm):
                        return True, f"Semantic match: '{pred_norm}' ~ '{gold_norm}'"
                
                elif strategy == 'loose':
                    # Very loose matching (remove all spaces and underscores)
                    pred_simple = predicted_answer.replace('_', '').replace(' ', '').lower()
                    gold_simple = golden.replace('_', '').replace(' ', '').lower()
                    if pred_simple in gold_simple or gold_simple in pred_simple:
                        return True, f"Loose match: '{pred_simple}' ~ '{gold_simple}'"
        
        return False, f"No match found with strategies: {matching_strategies}"
    
    def verify_answer(self, quid: int, final_answer: str, use_advanced_matching: bool = True) -> Dict[str, Any]:
        """
        Verify answer with flexible matching strategies
        
        Args:
            quid: question ID
            final_answer: predicted answer
            use_advanced_matching: if True, use all available matching strategies
                                  if False, use only basic exact/contain matching
        
        Returns:
            Verification result dictionary:
            {
                'is_correct': bool,
                'status': str,
                'answer_type': str,
                'question_type': str,
                'predicted_answer': str,
                'golden_answers': List[str],
                'match_details': str,
                'matching_strategy': str  # which strategy matched
            }
        """
        # Check if golden answer exists
        if quid not in self.golden_answers:
            return {
                'is_correct': False,
                'status': f'Question ID {quid} not in golden answers',
                'answer_type': 'unknown',
                'question_type': 'unknown',
                'predicted_answer': final_answer,
                'golden_answers': [],
                'match_details': 'No golden answer found',
                'matching_strategy': 'none'
            }
        
        golden_data = self.golden_answers[quid]
        golden_answers = golden_data['answers']
        answer_type = golden_data['answer_type']
        question_type = golden_data['qtype']
        
        if not final_answer:
            return {
                'is_correct': False,
                'status': 'Predicted answer is empty',
                'answer_type': answer_type,
                'question_type': question_type,
                'predicted_answer': final_answer,
                'golden_answers': golden_answers,
                'match_details': 'Empty prediction',
                'matching_strategy': 'none'
            }
        
        # Choose matching strategies based on use_advanced_matching flag
        if use_advanced_matching:
            strategies = ['exact', 'contain', 'advanced_normalize', 'time_format', 
                         'multi_answer', 'semantic', 'loose']
        else:
            strategies = ['exact', 'contain']
        
        # Use flexible matching system
        is_correct, match_details = self.is_answer_correct_flexible(quid, final_answer, strategies)
        
        # Extract which strategy worked from match_details
        matching_strategy = 'none'
        if is_correct and ':' in match_details:
            matching_strategy = match_details.split(':')[0].strip()
        
        return {
            'is_correct': is_correct,
            'status': 'Verification completed',
            'answer_type': answer_type,
            'question_type': question_type,
            'predicted_answer': final_answer,
            'golden_answers': golden_answers,
            'match_details': match_details,
            'matching_strategy': matching_strategy
        }
    
    def _verify_entity_answer(self, predicted: str, golden_list: List[str]) -> Tuple[bool, str]:
        """
        Verify entity type answer
        
        Rules:
        - After normalization, predicted in golden or golden in predicted
        
        Args:
            predicted: predicted answer
            golden_list: golden answer list
        
        Returns:
            (is correct, match details)
        """
        pred_norm = self._normalize_entity(predicted)
        
        for golden in golden_list:
            gold_norm = self._normalize_entity(golden)
            
            # Check containment relationship
            if pred_norm in gold_norm or gold_norm in pred_norm:
                return True, f"Match success: '{predicted}' <-> '{golden}'"
        
        # No match
        return False, f"No match: '{predicted}' not in {golden_list}"
    
    def _verify_time_answer(self, predicted: str, golden_list: List[str]) -> Tuple[bool, str]:

        for golden in golden_list:
            if self._time_matches(predicted, golden):
                return True, f"Time match: '{predicted}' <-> '{golden}'"
        
        # No match
        return False, f"Time not match: '{predicted}' not in {golden_list}"


# Global instance
_answer_verifier = None
_last_golden_path = None


def get_answer_verifier(golden_answers_path: str = None) -> AnswerVerifier:
    """
    Get global answer verifier instance
    Automatically switches based on current dataset
    
    Args:
        golden_answers_path: Optional path to golden answers file
                            If None, uses current dataset's test data path from TPKGConfig
    
    Returns:
        AnswerVerifier instance
    """
    global _answer_verifier, _last_golden_path
    
    if golden_answers_path is None:
        try:
            from config import TPKGConfig
            golden_answers_path = TPKGConfig.get_test_data_path()
        except:
            try:
                from ..config import TPKGConfig
                golden_answers_path = TPKGConfig.get_test_data_path()
            except:
                # Fallback: use relative path from this file
                from pathlib import Path
                project_root = Path(__file__).parent.parent.parent.absolute()
                golden_answers_path = str(project_root / "Data" / "MultiTQ" / "questions_with_candidates_multitq.json")
    
    if _answer_verifier is None or _last_golden_path != golden_answers_path:
        print(f"🔄 AnswerVerifierRe-initialize: {golden_answers_path}")
        _answer_verifier = AnswerVerifier(golden_answers_path)
        _last_golden_path = golden_answers_path
    
    return _answer_verifier

