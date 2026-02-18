#!/usr/bin/env python3



from typing import List, Dict, Any, Optional
from kg_agent.llm import LLM
import json

class DebateVoteSystem:
    
    def __init__(self):
        self.llm = LLM()
    
    def collect_toolkit_results(self, toolkit_results: List[Dict[str, Any]]) -> Dict[str, Any]:

        collected_results = {
            "toolkit_count": len(toolkit_results),
            "toolkit_results": [],
            "all_candidates": [],
            "all_parameters": []
        }
        
        for i, result in enumerate(toolkit_results):
            toolkit_info = {
                "toolkit_id": i + 1,
                "toolkit_name": result.get("toolkit_name", f"toolkit_{i+1}"),
                "parameters": result.get("parameters", {}),
                "chosen_answer": result.get("chosen", {}),
                "candidates": result.get("candidates", []),
                "success": result.get("ok", False),
                "explanations": result.get("explanations", [])
            }
            
            chosen = result.get("chosen", {})
            if chosen:
                toolkit_info["chosen_details"] = {
                    "entity": chosen.get("entity", "Unknown"),
                    "time": chosen.get("time", "Unknown"),
                    "path": chosen.get("path", []),
                    "provenance": chosen.get("provenance", {}),
                    "score": chosen.get("provenance", {}).get("similarity", 0)
                }
            
            collected_results["toolkit_results"].append(toolkit_info)
            collected_results["all_candidates"].extend(result.get("candidates", []))
            collected_results["all_parameters"].append(result.get("parameters", {}))
        
        return collected_results
    
    def generate_debate_prompt(self, subquestion: str, collected_results: Dict[str, Any]) -> str:

        prompt = f"""As a expert, you need to evaluate the results of multiple toolkits and choose the most correct answer.
Subquestion: {subquestion}
Toolkit results: {collected_results}
Return the most correct answer in JSON format.
"""

        
        for i, toolkit_result in enumerate(collected_results["toolkit_results"]):
            prompt += f"\nToolkit {i+1}: {toolkit_result['toolkit_name']}\n"
            prompt += f"Parameters: {toolkit_result['parameters']}\n"
            prompt += f"Success: {toolkit_result['success']}\n"
            
            if toolkit_result.get("chosen_details"):
                chosen = toolkit_result["chosen_details"]
                prompt += f"Chosen answer: {chosen['entity']} (Time: {chosen['time']}, Score: {chosen['score']:.3f})\n"
                if chosen.get("path"):
                    # Handle new path structure [heads_list, relation, tails_list]
                    path = chosen['path']
                    if isinstance(path, list) and len(path) >= 3:
                        heads = path[0] if isinstance(path[0], list) else [path[0]]
                        relation = path[1]
                        tails = path[2] if isinstance(path[2], list) else [path[2]]
                        heads_str = ', '.join(heads) if len(heads) > 1 else heads[0]
                        tails_str = ', '.join(tails) if len(tails) > 1 else tails[0]
                        prompt += f"Path: {heads_str} -> {relation} -> {tails_str}\n"
                    else:
                        # Fallback for old path format
                        path_str = ' -> '.join(str(item) for item in path)
                        prompt += f"Path: {path_str}\n"
                if chosen.get("provenance", {}).get("selection_reason"):
                    prompt += f"Selection reason: {chosen['provenance']['selection_reason']}\n"
            
            prompt += f"Candidate number: {len(toolkit_result['candidates'])}\n"
            prompt += f"Explanation: {', '.join(toolkit_result['explanations'])}\n"
            prompt += "-" * 50 + "\n"
        
        prompt += """
Please evaluate the results of each toolkit based on the following criteria:
1. Whether the answer directly answers the subquestion
2. Whether the time information is accurate and complete
3. Whether the path information is reasonable and complete
4. Whether the similarity score is high
5. Whether the selection reason is reasonable
6. Whether the toolkit parameters are suitable for the subquestion

Please return the evaluation result in JSON format:
{
    "winning_toolkit": Toolkit number,
    "winning_answer": {
        "entity": "Selected entity",
        "time": "Time",
        "path": ["head", "relation", "tail"],
        "score": "Similarity score",
        "reason": "Selection reason"
    },
    "evaluation": {
        "criteria_scores": {
            "toolkit_1": {"relevance": score, "accuracy": score, "completeness": score},
            "toolkit_2": {"relevance": score, "accuracy": score, "completeness": score}
        },
        "overall_winner": "Toolkit number",
        "reasoning": "Detailed evaluation reasoning process"
    }
}
"""
        
        return prompt
    
    def conduct_debate_vote(self, subquestion: str, toolkit_results: List[Dict[str, Any]]) -> Dict[str, Any]:

        print(f"🗳️ Debate Vote: {subquestion}")
        print(f"📊 Toolkit number: {len(toolkit_results)}")
        
        # Collect results
        collected_results = self.collect_toolkit_results(toolkit_results)
        
        # Generate debate prompt
        debate_prompt = self.generate_debate_prompt(subquestion, collected_results)
        
        try:
            # Call LLM for debate vote
            llm_response = self.llm.call("", debate_prompt)
            
            # Parse LLM response
            try:
                vote_result = json.loads(llm_response)
                print(f"✅ Debate Vote completed: Winning toolkit {vote_result.get('winning_toolkit')}")
                print(f"🏆 Winning answer: {vote_result.get('winning_answer', {}).get('entity')}")
                
                # Add original results information
                vote_result["original_results"] = collected_results
                vote_result["subquestion"] = subquestion
                
                return vote_result
                
            except json.JSONDecodeError:
                print("⚠️ LLM response JSON parsing failed, using heuristic selection")
                return self._fallback_selection(toolkit_results, subquestion)
                
        except Exception as e:
            print(f"❌ Debate Vote failed: {e}")
            return self._fallback_selection(toolkit_results, subquestion)
    
    def _fallback_selection(self, toolkit_results: List[Dict[str, Any]], subquestion: str) -> Dict[str, Any]:

        print("🔄 Using heuristic fallback selection")
        
        # Select the first successful toolkit result
        for i, result in enumerate(toolkit_results):
            if result.get("ok") and result.get("chosen"):
                chosen = result["chosen"]
                return {
                    "winning_toolkit": i + 1,
                    "winning_answer": {
                        "entity": chosen.get("entity", "Unknown"),
                        "time": chosen.get("time", "Unknown"),
                        "path": chosen.get("path", []),
                        "score": chosen.get("provenance", {}).get("similarity", 0),
                        "reason": f"Heuristic selection toolkit {i+1}"
                    },
                    "evaluation": {
                        "overall_winner": str(i + 1),
                        "reasoning": "Heuristic selection the first successful result"
                    },
                    "subquestion": subquestion
                }
        
        # If no successful result, return an empty result
        return {
            "winning_toolkit": 0,
            "winning_answer": {
                "entity": "Unknown",
                "time": "Unknown", 
                "path": [],
                "score": 0,
                "reason": "No valid result found"
            },
            "evaluation": {
                "overall_winner": "0",
                "reasoning": "All toolkits failed"
            },
            "subquestion": subquestion
        }

def test_debate_vote():
    """Test debate vote system"""
    print("=== Test Debate Vote system ===")
    
    # Mock toolkit results.
    mock_results = [
        {
            "toolkit_name": "intelligent_retrieve_one_hop",
            "parameters": {"query": "China", "limit": 20},
            "ok": True,
            "chosen": {
                "entity": "China",
                "time": "2010-02-20",
                "path": ["Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)", "Express_intent_to_meet_or_negotiate", "China"],
                "provenance": {
                    "similarity": 0.788,
                    "selection_reason": "The path clearly indicates the specific time when Kazakhstan Cabinet Council of Ministers expressed the intention to negotiate with China"
                }
            },
            "candidates": [{"entity": "China", "time": "2010-02-20"}],
            "explanations": ["Smart retrieval found 1 candidate"]
        },
        {
            "toolkit_name": "intelligent_find_direct_connection", 
            "parameters": {"entity1": 62, "entity2": 4774},
            "ok": True,
            "chosen": {
                "entity": "China",
                "time": "2010-02-20", 
                "path": ["Cabinet_/_Council_of_Ministers_/_Advisors_(Kazakhstan)", "Express_intent_to_meet_or_negotiate", "China"],
                "provenance": {
                    "similarity": 0.788,
                    "selection_reason": "The path directly mentions the intention of Kazakhstan Cabinet Council of Ministers to negotiate with China"
                }
            },
            "candidates": [{"entity": "China", "time": "2010-02-20"}],
            "explanations": ["Direct connection found 1 candidate"]
        }
    ]
    
    # Create debate vote system
    debate_system = DebateVoteSystem()
    
    # Conduct debate vote
    subquestion = "When did Cabinet Council of Ministers of Kazakhstan express the intention to negotiate with China?"
    vote_result = debate_system.conduct_debate_vote(subquestion, mock_results)
    
    print(f"Vote result: {vote_result}")

if __name__ == "__main__":
    test_debate_vote()

