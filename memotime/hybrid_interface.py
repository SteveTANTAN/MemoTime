#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid retrieval interface - simple and easy-to-use question to triple retrieval interface
"""

import asyncio
from typing import List, Dict, Any, Optional
from memotime.kg_agent.hybrid_retrieval import hybrid_retrieve_triples


class HybridInterface:
    """Hybrid retrieval interface class"""
    
    def __init__(self, db_path: str, raw_data_path: str = None):
        """
        Initialize hybrid retrieval interface
        
        Args:
            db_path: database path
            raw_data_path: original data file path (optional)
        """
        self.db_path = db_path
        self.raw_data_path = raw_data_path
    
    async def retrieve_triples(self, 
                             question: str, 
                             top_k: int = 50,
                             re_rank: bool = True) -> Dict[str, Any]:
        """
        Retrieve related triples based on question
        
        Args:
            question: input question
            top_k: number of triples to return
            re_rank: whether to use time re-ranking
            
        Returns:
            dictionary containing related triples
        """
        return await hybrid_retrieve_triples(
            question=question,
            db_path=self.db_path,
            top_k=top_k,
            re_rank=re_rank,
            raw_data_path=self.raw_data_path
        )
    
    def retrieve_triples_sync(self, 
                            question: str, 
                            top_k: int = 50,
                            re_rank: bool = True) -> Dict[str, Any]:
        """
        Synchronous version of triple retrieval
        
        Args:
            question: input question
            top_k: number of triples to return
            re_rank: whether to use time re-ranking
            
        Returns:
            dictionary containing related triples
        """
        return asyncio.run(self.retrieve_triples(question, top_k, re_rank))
    
    async def batch_retrieve_triples(self, 
                                   questions: List[str], 
                                   top_k: int = 50,
                                   re_rank: bool = True) -> List[Dict[str, Any]]:
        """
                Batch retrieve related triples for multiple questions
        
        Args:
            questions: question list
            top_k: number of triples to return for each question
            re_rank: whether to use time re-ranking
            
        Returns:
            list of dictionaries containing related triples
        """
        tasks = [
            self.retrieve_triples(question, top_k, re_rank) 
            for question in questions
        ]
        return await asyncio.gather(*tasks)
    
    def batch_retrieve_triples_sync(self, 
                                  questions: List[str], 
                                  top_k: int = 50,
                                  re_rank: bool = True) -> List[Dict[str, Any]]:
        """
        Synchronous version of batch triple retrieval
        
        Args:
            questions: question list
            top_k: number of triples to return for each question
            re_rank: whether to use time re-ranking
            
        Returns:
            list of dictionaries containing related triples
        """
        return asyncio.run(self.batch_retrieve_triples(questions, top_k, re_rank))


# Convenient function
def retrieve_triples_for_question(question: str, 
                                db_path: str,
                                top_k: int = 50,
                                re_rank: bool = True,
                                raw_data_path: str = None) -> Dict[str, Any]:
    """
    Convenient function: retrieve related triples for question (synchronous version)
    
    Args:
        question: input question
        db_path: database path
        top_k: number of triples to return
        re_rank: whether to use time re-ranking
        raw_data_path: original data file path (optional)
        
    Returns:
        dictionary containing related triples
    """
    interface = HybridInterface(db_path, raw_data_path)
    return interface.retrieve_triples_sync(question, top_k, re_rank)


async def retrieve_triples_for_question_async(question: str, 
                                            db_path: str,
                                            top_k: int = 50,
                                            re_rank: bool = True) -> Dict[str, Any]:
    """
    Convenient function: retrieve related triples for question (asynchronous version)
    
    Args:
        question: input question
        db_path: database path
        top_k: number of triples to return
        re_rank: whether to use time re-ranking
        
    Returns:
        dictionary containing related triples
    """
    interface = HybridInterface(db_path)
    return await interface.retrieve_triples(question, top_k, re_rank)


def batch_retrieve_triples_for_questions(questions: List[str], 
                                        db_path: str,
                                        top_k: int = 50,
                                        re_rank: bool = True) -> List[Dict[str, Any]]:
    """
        Convenient function: batch retrieve related triples for multiple questions (synchronous version)
    
    Args:
        questions: question list
        db_path: database path
        top_k: number of triples to return for each question
        re_rank: whether to use time re-ranking
        
    Returns:
        list of dictionaries containing related triples
    """
    interface = HybridInterface(db_path)
    return interface.batch_retrieve_triples_sync(questions, top_k, re_rank)


