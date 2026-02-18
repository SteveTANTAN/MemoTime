#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================
# file: main.py
# =============================
from data_process import load_questions
from kg_agent import TemplateRegistry, Agent, solve_with_decomposition
from kg_agent.performance_monitor import get_performance_monitor, start_question_monitoring, end_question_monitoring
from tqdm import tqdm
import json
import sys

# import unified configuration
from config import TPKGConfig, load_preset, apply_config
from experiment_database import get_experiment_db


def run_experiment(args=None):
    """
    Main function to run experiment
    
    Args:
        args: command line argument object (passed from run.py), if None then read from environment variables (backward compatibility)
    """
    
    # ========== apply configuration (without using environment variables) ==========
    
    if args:
        # new way: read configuration from command line arguments
        config_dict = {}
        
        # first handle dataset configuration (highest priority)
        if hasattr(args, 'dataset') and args.dataset:
            config_dict['dataset'] = args.dataset
        
        # preset configuration
        if args.preset:
            from config import ExperimentPresets
            presets = {
                "baseline": ExperimentPresets.baseline(),
                "fast_mode": ExperimentPresets.fast_mode(),
                "accuracy_mode": ExperimentPresets.accuracy_mode(),
                "full_optimization": ExperimentPresets.full_optimization(),
                "equal_type_test": ExperimentPresets.equal_type_test(),
                "experience_pool_test": ExperimentPresets.experience_pool_test(),
            }
            if args.preset in presets:
                config_dict.update(presets[args.preset])
        
        # apply command line arguments (override preset)
        if args.retries is not None:
            config_dict['max_retries'] = args.retries
        if args.depth is not None:
            config_dict['max_depth'] = args.depth
        if args.branches is not None:
            config_dict['max_total_branches'] = args.branches
        if args.hybrid is not None:
            config_dict['use_hybrid_retrieval'] = args.hybrid
        # experience pool and template learning parameters have been removed
        if hasattr(args, 'unified_knowledge') and args.unified_knowledge is not None:
            config_dict['use_unified_knowledge_store'] = args.unified_knowledge
        if args.questions is not None:
            config_dict['max_questions'] = args.questions
        if args.skip is not None:
            config_dict['skip_questions'] = args.skip
        if args.type:
            config_dict['filter_question_type'] = args.type
        if args.entities is not None:
            config_dict['max_candidate_entities'] = args.entities
        if args.model:
            config_dict['default_llm_model'] = args.model
        
        # storage mode configuration
        if hasattr(args, 'storage_mode'):
            config_dict['storage_mode'] = args.storage_mode
        if hasattr(args, 'enable_shared_fallback'):
            config_dict['enable_shared_fallback'] = args.enable_shared_fallback
        if hasattr(args, 'config_name') and args.config_name:
            config_dict['config_name'] = args.config_name
        
        # apply configuration
        apply_config(config_dict)
        
        # experiment metadata
        EXPERIMENT_NAME = args.name if hasattr(args, 'name') and args.name else None
        EXPERIMENT_DESC = args.desc if hasattr(args, 'desc') and args.desc else ""
        EXPERIMENT_TAGS = args.tags.split(",") if hasattr(args, 'tags') and args.tags else []
        AUTO_SAVE = not (hasattr(args, 'no_save') and args.no_save)
        
    else:
        # old way: read from environment variables (backward compatibility)
        import os
        
        # read preset from environment variables
        EXPERIMENT_PRESET = os.getenv("EXPERIMENT_PRESET", None)
        if EXPERIMENT_PRESET:
            load_preset(EXPERIMENT_PRESET)
        
        # experiment metadata
        EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", None)
        EXPERIMENT_DESC = os.getenv("EXPERIMENT_DESC", "")
        EXPERIMENT_TAGS = os.getenv("EXPERIMENT_TAGS", "").split(",") if os.getenv("EXPERIMENT_TAGS") else []
        AUTO_SAVE = os.getenv("AUTO_EXPERIMENT", "true").lower() == "true"
    
    # ========== database management (based on configuration) ==========
    exp_db = None
    config_id = None
    config_name = EXPERIMENT_NAME or "default"

    if AUTO_SAVE:
        exp_db = get_experiment_db()
        
        # automatically generate configuration name (ensure always has a clear name)
        if not EXPERIMENT_NAME:
            if args and hasattr(args, 'preset') and args.preset:
                config_name = args.preset  # directly use preset name
            else:
                # generate configuration name according to fixed parameter order, core parameters and feature switches use fixed placeholders
                parts = []
                
                # 0. dataset (fixed position 0) - ✅新增
                parts.append(TPKGConfig.DATASET.lower())  # timequestions 或 multitq
                
                # 1. retries (fixed position 1)
                parts.append(f"retry{TPKGConfig.MAX_RETRIES}")
                
                # 2. decomposition depth (fixed position 2)
                parts.append(f"depth{TPKGConfig.MAX_DEPTH}")
                
                # 3. branch number (fixed position 3)
                parts.append(f"branch{TPKGConfig.MAX_TOTAL_BRANCHES}")
                
                # 4. feature switches (fixed position 4-7)
                if TPKGConfig.USE_HYBRID_RETRIEVAL:
                    parts.append("hybrid")
                else:
                    parts.append("nohybrid")
                    
                # experience pool and template learning have been removed
                    
                if TPKGConfig.USE_UNIFIED_KNOWLEDGE_STORE:
                    parts.append("unified")
                else:
                    parts.append("nounified")
                
                # 5. storage mode (fixed position 8)
                if hasattr(TPKGConfig, 'STORAGE_MODE') and TPKGConfig.STORAGE_MODE:
                    parts.append(TPKGConfig.STORAGE_MODE)
                else:
                    parts.append("shared")
                
                # 6. shared fallback (fixed position 9)
                if hasattr(TPKGConfig, 'ENABLE_SHARED_FALLBACK') and TPKGConfig.ENABLE_SHARED_FALLBACK:
                    parts.append("fallback")
                else:
                    parts.append("nofallback")
                
                # 7. default model (fixed position 10)
                default_model = TPKGConfig.DEFAULT_LLM_MODEL
                # simplify model name
                if 'gpt-4o-mini' in default_model:
                    parts.append("gpt4mini")
                elif 'deepseek-v3' in default_model:
                    parts.append("deepseek")
                elif 'gpt-4' in default_model:
                    parts.append("gpt4")
                else:
                    # for other models, take the first 8 characters
                    model_short = default_model.replace('-', '').replace('_', '')[:8]
                    parts.append(model_short)
                
                # experimental range parameters are not included in the configuration name, because they do not affect the core configuration
                
                config_name = "_".join(parts)
        
        # get or create configuration (based on configuration content hash)
        config_id = exp_db.get_or_create_config(
            config_name=config_name,
            config=TPKGConfig.get_all_config(),
            description=EXPERIMENT_DESC,
            tags=EXPERIMENT_TAGS
        )

    # print current configuration
    print("\n" + "=" * 80)
    print("current running configuration:")
    print("=" * 80)
    if config_id:
        print(f"configuration ID: {config_id}")
        print(f"configuration name: {config_name}")
        print("-" * 80)
    print(f"maximum retries: {TPKGConfig.MAX_RETRIES}")
    print(f"maximum depth: {TPKGConfig.MAX_DEPTH}")
    print(f"maximum branch number: {TPKGConfig.MAX_TOTAL_BRANCHES}")
    print(f"LLM model: {TPKGConfig.DEFAULT_LLM_MODEL}")
    print(f"use hybrid retrieval: {TPKGConfig.USE_HYBRID_RETRIEVAL}")
    # experience pool and template learning have been removed
    print(f"use unified knowledge store: {TPKGConfig.USE_UNIFIED_KNOWLEDGE_STORE}")
    print(f"process question number: {TPKGConfig.MAX_QUESTIONS}")
    print(f"skip question number: {TPKGConfig.SKIP_QUESTIONS}")
    if TPKGConfig.FILTER_QUESTION_TYPE:
        print(f"filter question type: {TPKGConfig.FILTER_QUESTION_TYPE}")
    print("=" * 80 + "\n")

    # ========== load data and run experiment ==========
    # use configuration dataset path (automatically select based on current dataset)
    datas, q_name, Qid, t_e = load_questions()

    registry = TemplateRegistry()
    agent = Agent(registry)

    # use configuration parameters
    processed_count = 0
    type_matched_count = 0
    success_count = 0
    fail_count = 0
    
    skip_end = TPKGConfig.SKIP_QUESTIONS
    process_end = TPKGConfig.SKIP_QUESTIONS + TPKGConfig.MAX_QUESTIONS
    
    for data in tqdm(datas[skip_end:process_end], desc="Processing questions"):
        if processed_count >= TPKGConfig.MAX_QUESTIONS:
            break
            
        question = data[q_name]
        q_id = data[Qid]
        topic_entities = data.get(t_e) 
        topic_entities = topic_entities[:TPKGConfig.MAX_CANDIDATE_ENTITIES]
        
        # check question type (using correct field names)
        with open(TPKGConfig.TEST_DATA_PATH, 'r') as f:
            test_data_list = json.load(f)
        
        dataset_config = TPKGConfig.get_dataset_config()
        id_field = dataset_config['question_id_field']
        
        question_type = None
        for test_item in test_data_list:
            if test_item.get(id_field) == q_id:
                # MultiTQ has qtype field, TimeQuestions may not have
                question_type = test_item.get('qtype', None)
                break
        
        # if question type filtering is configured, then filter
        if TPKGConfig.FILTER_QUESTION_TYPE:
            if question_type != TPKGConfig.FILTER_QUESTION_TYPE:
                continue
            type_matched_count += 1
        
        processed_count += 1
        
        # check if this question has a correct answer in the current answer DB, if so then skip
        if exp_db and config_id:
            existing_result = exp_db.get_question_result(config_id, str(q_id))
            if existing_result and existing_result.get('is_correct'):
                print(f"\n{'='*100}")
                print(f"⏭️  skip question #{processed_count} | QID={q_id} (already has correct answer)")
                print(f"   saved answer: {existing_result.get('final_answer', '')[:80]}...")
                print('='*100)
                success_count += 1  # count into success number
                continue

        print(f"\n{'='*100}")
        print(f"question #{processed_count} | QID={q_id}")
        print(f"question type: {question_type}")
        print(f"question: {question}")
        if TPKGConfig.FILTER_QUESTION_TYPE:
            print(f"type matched number: {type_matched_count}")
        print('='*100)

        # start performance monitoring
        if TPKGConfig.ENABLE_PERFORMANCE_MONITORING:
            start_question_monitoring(q_id, question, question_type)

        # use decomposition to solve
        print("Using decomposition with configuration:")
        print(f"  - Max retries: {TPKGConfig.MAX_RETRIES}")
        print(f"  - Max depth: {TPKGConfig.MAX_DEPTH}")
        print(f"  - Hybrid retrieval: {TPKGConfig.USE_HYBRID_RETRIEVAL}")
        # experience pool and template learning have been removed
        
        result = solve_with_decomposition(
            agent, 
            question, 
            topic_entities, 
            quid=q_id,
            max_retries=TPKGConfig.MAX_RETRIES,
            max_depth=TPKGConfig.MAX_DEPTH,
            max_branch=TPKGConfig.MAX_TOTAL_BRANCHES,
            use_hybrid=TPKGConfig.USE_HYBRID_RETRIEVAL,
            storage_mode=TPKGConfig.STORAGE_MODE,
            llm_model=TPKGConfig.DEFAULT_LLM_MODEL,
            # template learning have been removed
        )
        
        # end performance monitoring
        is_correct = None
        if TPKGConfig.ENABLE_PERFORMANCE_MONITORING:
            learning_info = result.get('learning', {})
            if 'verified' in learning_info:
                is_correct = learning_info.get('verified', False)
            else:
                is_correct = result.get('ok', False) and result.get('final_answer', '') != ''
            
            final_answer = result.get('final_answer', '')
            answer_path = result.get('answer_path', [])
            end_question_monitoring(is_correct, final_answer, answer_path)
            
            # count success and failure
            if is_correct:
                success_count += 1
            else:
                fail_count += 1
        
        # immediately save data (save immediately after each question, to prevent loss due to interruption)
        if exp_db and config_id and is_correct is not None:
            trajectory = result.get('trajectory', [])
            
                # extract evidence edges
            evidence_edges = []
            for step_idx, step in enumerate(trajectory):
                step_result = step.get('result', {})
                candidates = step_result.get('candidates', [])
                for cand in candidates:
                    edge = {
                        'head_entity': cand.get('head', cand.get('entity', '')),
                        'relation': cand.get('relation', ''),
                        'tail_entity': cand.get('tail', cand.get('tail_entity', '')),
                        'time_info': cand.get('time', cand.get('date', '')),
                        'score': cand.get('score', cand.get('similarity', None)),
                        'source': cand.get('provenance', {}).get('method', 'unknown'),
                        'is_used': cand.get('entity') == step_result.get('chosen', {}).get('entity'),
                        'step_index': step_idx,
                        'metadata': {'candidate_full': cand}
                    }
                    evidence_edges.append(edge)
            
            answer_path = result.get('answer_path', [])
            for path_item in answer_path:
                if isinstance(path_item, dict):
                    edge = {
                        'head_entity': path_item.get('head', path_item.get('entity', '')),
                        'relation': path_item.get('relation', ''),
                        'tail_entity': path_item.get('tail', ''),
                        'time_info': path_item.get('time', path_item.get('date', '')),
                        'score': path_item.get('score', 1.0),
                        'source': 'final_answer_path',
                        'is_used': True,
                        'metadata': {'from_answer_path': True}
                    }
                    evidence_edges.append(edge)
            
            # check if should save
            should_save = exp_db.should_save_question(config_id, str(q_id), is_correct)
            
            if should_save:
                # immediately save (do not wait for all questions to complete)
                exp_db.save_best_results(
                    config_id=config_id,
                    qid=str(q_id),
                    question=question,
                    question_type=question_type,
                    final_answer=result.get('final_answer', ''),
                    is_correct=is_correct,
                    trajectory=trajectory,
                    evidence_edges=evidence_edges
                )
                print(f"💾 already saved question {q_id} result")
            else:
                print(f"⏭️ skip question {q_id} (keep previous correct answer)")
    
    print("\n" + "=" * 80)
    print("processing completed statistics:")
    print("=" * 80)
    print(f"total processed questions: {processed_count}")
    if TPKGConfig.FILTER_QUESTION_TYPE:
        print(f"matched type '{TPKGConfig.FILTER_QUESTION_TYPE}': {type_matched_count} questions")
    if TPKGConfig.ENABLE_PERFORMANCE_MONITORING:
        print(f"success: {success_count} questions")
        print(f"failure: {fail_count} questions")
        if processed_count > 0:
            current_success_rate = success_count / processed_count
            print(f"success rate: {current_success_rate * 100:.2f}%")
    print("=" * 80)
    
    # record this run statistics
    if exp_db and config_id:
        # record this run
        run_id = exp_db.record_run(
            config_id=config_id,
            total_questions=processed_count,
            success_count=success_count,
            fail_count=fail_count
        )
        
        # show configuration statistics
        print(f"\n📊 configuration '{config_name}' statistics:")
        stats = exp_db.get_config_stats(config_id)
        print(f"   - configuration ID: {config_id}")
        print(f"   - total run count: {stats['config']['run_count']}")
        print(f"   - best success rate: {stats['config']['best_success_rate']*100:.2f}%")
        print(f"   - total saved questions: {stats['stats'].get('total_questions', 0) or 0}")
        
        print(f"\n💾 database location: {exp_db.db_path}")
    
    print("\n✅ experiment completed!")


if __name__ == "__main__":
    # when running directly, use environment variables (backward compatibility)
    run_experiment(None)
