#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    TPKG simplified running script  
"""

import argparse
import os
import sys
import json
import os

# Set CUDA_VISIBLE_DEVICES to make only GPU 0 visible
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

def main():
    parser = argparse.ArgumentParser(
        description='TPKG experiment running tool - simplified version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # use preset configuration
  python run.py --preset baseline
  python run.py --preset fast_mode
  
  # custom configuration
  python run.py --name my_test --retries 2 --hybrid --unified-knowledge
  
  # test specific question type 
  python run.py --type equal --questions 30
  
  # complete example
  python run.py --name hybrid_test --retries 1 --depth 2 --hybrid --unified-knowledge --questions 50 --skip 10
  
Preset configurations:
  baseline           - baseline (no optimization)
  fast_mode          - fast mode
  accuracy_mode      - accuracy mode
  full_optimization  - full optimization
        """
    )
    
    # ========== experiment configuration ==========
    exp_group = parser.add_argument_group('experiment configuration')
    exp_group.add_argument('--dataset', type=str, 
                          choices=['TimeQuestions', 'MultiTQ'],
                          default='MultiTQ',
                          help='select dataset: TimeQuestions or MultiTQ (default: MultiTQ)')
    exp_group.add_argument('--name', '-n', type=str, default=None,
                          help='experiment name (default: auto-generated)')
    exp_group.add_argument('--desc', type=str, default='',
                          help='experiment description')
    exp_group.add_argument('--tags', type=str, default='',
                          help='experiment tags (comma separated)') 
    
    # ========== preset configuration ==========
    preset_group = parser.add_argument_group('preset configuration')
    preset_group.add_argument('--preset', '-p', type=str, 
                             choices=['baseline', 'fast_mode', 'accuracy_mode', 
                                     'full_optimization', 'equal_type_test', 
                                     'experience_pool_test',
                                     'retry2_hybrid_pool_template_unified_shared_1q',
                                     'retry2_hybrid_pool_template_individual_1q',
                                     'retry3_hybrid_pool_template_unified_shared_5q',
                                     'retry1_fast_mode_shared_1q',
                                     'retry2_accuracy_mode_individual_10q'],
                             help='use preset configuration')
    
    # ========== core parameters ==========
    core_group = parser.add_argument_group('core parameters')
    core_group.add_argument('--retries', '-r', type=int, default=None,
                           help='maximum retries (default: 1)')
    core_group.add_argument('--depth', '-d', type=int, default=None,
                           help='maximum decomposition depth (default: 3)')
    core_group.add_argument('--branches', '-b', type=int, default=None,
                           help='maximum branches (default: 5)')
    
    # ========== feature switches ==========
    feature_group = parser.add_argument_group('feature switches')
    feature_group.add_argument('--hybrid', action='store_true',
                              help='enable hybrid retrieval')
    feature_group.add_argument('--no-hybrid', dest='hybrid', action='store_false',
                              help='disable hybrid retrieval')
    # experience pool and template learning have been removed, using unified knowledge store
    # feature_group.add_argument('--pool', action='store_true', help='启用经验池')
    # feature_group.add_argument('--template', action='store_true', help='启用模板学习')
    feature_group.add_argument('--unified-knowledge', action='store_true',
                              help='enable unified knowledge store')
    feature_group.add_argument('--no-unified-knowledge', dest='unified_knowledge', action='store_false',
                              help='disable unified knowledge store')
    
    # ========== storage mode ==========
    storage_group = parser.add_argument_group('storage mode')   
    storage_group.add_argument('--storage-mode', type=str, choices=['shared', 'individual'],
                              default='shared', help='storage mode: shared(shared) or individual(independent)')
    storage_group.add_argument('--enable-shared-fallback', action='store_true',
                              help='enable shared storage fallback (also access shared data in independent mode)')
    storage_group.add_argument('--config-name', type=str, default=None,
                              help='configuration name (used for database path), for example: retry2_depth3_branch20_hybrid_unified_individual_nofallback_gpt4mini')
    
    # ========== experiment range ==========
    range_group = parser.add_argument_group('experiment range') 
    range_group.add_argument('--questions', '-q', type=int, default=None,
                            help='process question number (default: 50)')
    range_group.add_argument('--skip', '-s', type=int, default=None,
                            help='skip first N questions (default: 5)')
    range_group.add_argument('--type', '-t', type=str, default=None,
                            choices=['equal', 'before_last', 'first_last', 'after_first'],
                            help='filter question type')
    range_group.add_argument('--entities', '-e', type=int, default=None,
                            help='candidate entity number (default: 10)')
    
    # ========== LLM configuration ==========
    llm_group = parser.add_argument_group('LLM configuration')
    llm_group.add_argument('--model', '-m', type=str, default=None,
                          help='LLM model (default: gpt-4o-mini)')
    
    # ========== other options ==========
    other_group = parser.add_argument_group('other options')
    other_group.add_argument('--result', action='store_true',
                            help='view experiment result statistics for this configuration (do not run experiment)')
    other_group.add_argument('--no-save', action='store_true',
                            help='do not save experiment data (temporary test)')
    other_group.add_argument('--verbose', '-v', action='store_true',
                            help='display detailed logs')
    
    # set default values
    parser.set_defaults(hybrid=None, unified_knowledge=None)
    
    args = parser.parse_args()
    
    # ========== if it is query result mode, directly query and exit ==========
    if args.result:
        from config import TPKGConfig, apply_config, ExperimentPresets
        from experiment_database import get_experiment_db
        
        # ✅ first apply dataset configuration (must be processed first)
        if hasattr(args, 'dataset') and args.dataset:
            apply_config({'dataset': args.dataset})
        
        # build configuration dictionary (only include core configuration, not include experiment range)
        config_dict = {}
        
        # preset configuration
        if args.preset:
            from config import load_preset
            preset_config = load_preset(args.preset)
            if preset_config:
                config_dict.update(preset_config)
        
        # apply core parameters (use default values in config.py)
        from config import TPKGConfig
        
        # initialize all configurations to default values
        config_dict.update({
            'dataset': TPKGConfig.DATASET,  # ✅ 添加dataset
            'max_retries': TPKGConfig.MAX_RETRIES,
            'max_depth': TPKGConfig.MAX_DEPTH,
            'max_branches': TPKGConfig.MAX_TOTAL_BRANCHES,
            'use_hybrid_retrieval': TPKGConfig.USE_HYBRID_RETRIEVAL,
            # experience pool and template learning have been removed
            'use_unified_knowledge_store': TPKGConfig.USE_UNIFIED_KNOWLEDGE_STORE,
            'storage_mode': TPKGConfig.STORAGE_MODE,
            'enable_shared_fallback': TPKGConfig.ENABLE_SHARED_FALLBACK,
            'default_llm_model': TPKGConfig.DEFAULT_LLM_MODEL
        })
        
        # use user-specified parameters to override default values
        if args.retries is not None:
            config_dict['max_retries'] = args.retries
        if args.depth is not None:
            config_dict['max_depth'] = args.depth
        if args.branches is not None:
            config_dict['max_branches'] = args.branches
        if args.hybrid is not None:
            config_dict['use_hybrid_retrieval'] = args.hybrid
        # experience pool and template learning parameters have been removed
        if args.unified_knowledge is not None:
            config_dict['use_unified_knowledge_store'] = args.unified_knowledge
        if args.storage_mode:
            config_dict['storage_mode'] = args.storage_mode
        if hasattr(args, 'enable_shared_fallback'):
            config_dict['enable_shared_fallback'] = args.enable_shared_fallback
        if args.model:
            config_dict['default_llm_model'] = args.model
        
        # determine configuration name (ensure always has a clear name)
        config_name = None
        if args.preset:
            config_name = args.preset
        elif args.name:
            config_name = args.name
        else:
            # generate configuration name according to fixed parameter order, core parameters and feature switches use fixed placeholders
            parts = []
            
            # 0. dataset (fixed position 0) - ✅ added
            dataset = config_dict.get('dataset', 'MultiTQ')
            parts.append(dataset.lower())  # timequestions 或 multitq
            
            # 1. retries (fixed position 1)
            retries = config_dict.get('max_retries', 1)
            parts.append(f"retry{retries}")
            
            # 2. decomposition depth (fixed position 2)
            depth = config_dict.get('max_depth', 1)
            parts.append(f"depth{depth}")
            
            # 3. branch number (fixed position 3)
            branches = config_dict.get('max_branches', 5)
            parts.append(f"branch{branches}")
            
            # 4. feature switches (fixed position 4-7)
            if config_dict.get('use_hybrid_retrieval'):
                parts.append("hybrid")
            else:
                parts.append("nohybrid")
                
            # experience pool and template learning have been removed
                
            if config_dict.get('use_unified_knowledge_store'):
                parts.append("unified")
            else:
                parts.append("nounified")
            
            # 5. storage mode (fixed position 8)
            storage_mode = config_dict.get('storage_mode', 'shared')
            parts.append(storage_mode)
            
            # 6. shared fallback (fixed position 9)
            if config_dict.get('enable_shared_fallback'):
                parts.append("fallback")
            else:
                parts.append("nofallback")
            
            # 7. default model (fixed position 10)
            default_model = config_dict.get('default_llm_model', 'gpt-4o-mini')
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
        
        # query the result of this configuration
        print("\n" + "=" * 80)
        print("query the result of this configuration")
        print("=" * 80)
        print(f"configuration name: {config_name}")
        
        # check if the configuration exists
        import sqlite3
        db = get_experiment_db()
        conn = sqlite3.connect(db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # directly find by configuration name
        cursor.execute("SELECT * FROM configurations WHERE config_name = ? ORDER BY created_at DESC LIMIT 1", (config_name,))
        config_row = cursor.fetchone()
        
        if not config_row:
            print(f"\n❌ did not find the experiment record of '{config_name}'")
            print(f"\nwarning: this configuration has not run any experiment")
            
            # display available configurations
            cursor.execute("SELECT DISTINCT config_name FROM configurations ORDER BY config_name")
            available_configs = cursor.fetchall()
            if available_configs:
                print(f"\navailable configuration names:")
                for config in available_configs:
                    print(f"  - {config[0]}")
            
            conn.close()
            sys.exit(0)
        
        config_info = dict(config_row)
        config_info['config_json'] = json.loads(config_info['config_json'])
        config_info['tags'] = json.loads(config_info['tags'])
        
        config_id = config_info['config_id']  # get configuration ID
        
        print(f"\nconfiguration information:")  
        print(f"  configuration ID: {config_info['config_id']}")
        print(f"  configuration name: {config_info['config_name']}")
        print(f"  run count: {config_info['run_count']}")
        print(f"  configuration table best success rate: {config_info['best_success_rate']*100:.2f}%")
        if config_info['best_run_at']:
            print(f"  best run time: {config_info['best_run_at']}")
        
        # query run history
        cursor.execute("""
        SELECT run_id, run_at, total_questions, success_rate, is_best
        FROM runs
        WHERE config_id = ?
        ORDER BY run_at DESC
        LIMIT 10
        """, (config_id,))
        
        runs = cursor.fetchall()
        if runs:
            print(f"\nrun history (last 10 times):")
            print(f"  {'running time':<20} | {'question number':<8} | {'success rate':<10} | mark")
            print("  " + "-" * 50)
            for run in runs:
                is_best_mark = " 🏆 best" if run[4] else ""
                print(f"  {run[1]:<20} | {run[2]:<8} | {run[3]*100:>6.2f}%  |{is_best_mark}")
        
        # query saved question statistics
        cursor.execute("""
        SELECT 
            COUNT(*) as total_saved,
            SUM(is_correct) as correct_count,
            AVG(is_correct) as avg_correct
        FROM question_results
        WHERE config_id = ?
        """, (config_id,))
        
        stats = cursor.fetchone()
        print(f"\nsaved question statistics (real-time calculation):")
        print(f"  total saved question number: {stats[0]}")
        print(f"  correct answer number: {stats[1] or 0}")
        print(f"  incorrect answer number: {stats[0] - (stats[1] or 0)}")
        if stats[0] > 0:
            real_time_accuracy = (stats[1] or 0) / stats[0] * 100
            print(f"  real-time success rate: {real_time_accuracy:.2f}%")
            
            # ✅ compare configuration table best success rate with actual correct rate
            config_best_rate = config_info['best_success_rate'] * 100
            if abs(real_time_accuracy - config_best_rate) > 0.01:
                print(f"  ⚠️  warning: real-time success rate ({real_time_accuracy:.2f}%) is not consistent with the best success rate in the configuration table ({config_best_rate:.2f}%)")
                print(f"      please run an experiment to synchronize the configuration table data")
        
        # query trajectory and edges statistics
        cursor.execute("""
        SELECT COUNT(DISTINCT qid) FROM trajectories WHERE config_id = ?
        """, (config_id,))
        traj_count = cursor.fetchone()[0]
        
        cursor.execute("""
        SELECT COUNT(*) FROM evidence_edges WHERE config_id = ?
        """, (config_id,))
        edge_count = cursor.fetchone()[0]
        
        print(f"\ndata integrity:")
        print(f"  Trajectory coverage: {traj_count} questions")
        print(f"  Evidence Edges: {edge_count} edges")
        
        # calculation based question types (real-time load question types from test.json)
        print(f"\calculation based question types:")
        print(f"  {'Type':<15} | {'Total':<6} | {'Correct':<6} | {'Success Rate'}")
        print("  " + "-" * 50)
        
        # get all question results
        cursor.execute("""
        SELECT qid, is_correct FROM question_results WHERE config_id = ?
        """, (config_id,))
        
        all_results = cursor.fetchall()
        
        # load question types mapping from test.json (use current dataset path)
        try:
            test_data_path = TPKGConfig.get_test_data_path()
            print(f"📊 load test data for statistics: {test_data_path}")
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data_list = json.load(f)
            
            # ✅ use correct field names based on dataset
            dataset_config = TPKGConfig.get_dataset_config()
            id_field = dataset_config['question_id_field']  # 'Id' or 'quid'
            
            qtype_map = {}
            for test_item in test_data_list:
                qid = str(test_item.get(id_field))
                
                # question type field processing
                if TPKGConfig.DATASET == 'TimeQuestions':
                    # TimeQuestions: 'Temporal question type' is a list
                    qtype_list = test_item.get('Temporal question type', ['unknown'])
                    qtype = qtype_list[0] if isinstance(qtype_list, list) and qtype_list else 'unknown'
                else:
                    # MultiTQ: 'qtype' is a string
                    qtype = test_item.get('qtype', 'unknown')
                
                qtype_map[qid] = qtype
        except Exception as e:
            print(f"  ⚠️  failed to load test data: {e}")
            import traceback
            traceback.print_exc()
            qtype_map = {}
        
        # calculation based question types (real-time load question types from test.json)
        type_stats = {}
        for qid, is_correct in all_results:
            qtype = qtype_map.get(qid, 'unknown')
            if qtype not in type_stats:
                type_stats[qtype] = {'total': 0, 'correct': 0}
            type_stats[qtype]['total'] += 1
            if is_correct:
                type_stats[qtype]['correct'] += 1
        
        # sort by success rate and display
        sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0, reverse=True)
        for qtype, stats in sorted_types:
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total'] * 100
                print(f"  {qtype:<15} | {stats['total']:<6} | {stats['correct']:<6} | {accuracy:>6.2f}%")
        
        # calculation based question labels (Multiple/Single) - real-time calculation
        print(f"\ncalculation based question labels (Multiple/Single) - real-time calculation:")
        print(f"  {'Label':<10} | {'Total':<6} | {'Correct':<6} | {'Success Rate'}")
        print("  " + "-" * 40)
        
        # load question labels mapping from test.json
        try:
            qlabel_map = {}
            for test_item in test_data_list:
                qid = str(test_item.get(id_field))
                # ✅ TimeQuestions has no qlabel field, use qtype as replacement
                if TPKGConfig.DATASET == 'TimeQuestions':
                    qlabel = 'N/A'  # TimeQuestions has no qlabel field
                else:
                    qlabel = test_item.get('qlabel', 'unknown')
                qlabel_map[qid] = qlabel
        except:
            qlabel_map = {}
        
        # calculation based question labels (Multiple/Single) - real-time calculation
        label_stats = {}
        for qid, is_correct in all_results:
            qlabel = qlabel_map.get(qid, 'unknown')
            if qlabel not in label_stats:
                label_stats[qlabel] = {'total': 0, 'correct': 0}
            label_stats[qlabel]['total'] += 1
            if is_correct:
                label_stats[qlabel]['correct'] += 1
        
        # sort by success rate and display
        sorted_labels = sorted(label_stats.items(), key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0, reverse=True)
        for qlabel, stats in sorted_labels:
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total'] * 100
                print(f"  {qlabel:<10} | {stats['total']:<6} | {stats['correct']:<6} | {accuracy:>6.2f}%")
        
        # calculation based answer types (entity/time) - real-time calculation
        print(f"\ncalculation based answer types (entity/time) - real-time calculation:")
        print(f"  {'Answer Type':<10} | {'Total':<6} | {'Correct':<6} | {'Success Rate'}")
        print("  " + "-" * 40)
        
        # load answer types mapping from test.json
        try:
            answer_type_map = {}
            for test_item in test_data_list:
                qid = str(test_item.get(id_field))
                
                # ✅ answer type field processing
                if TPKGConfig.DATASET == 'TimeQuestions':
                    # TimeQuestions: infer answer type from AnswerType field
                    answer_list = test_item.get('Answer', [])
                    if answer_list and isinstance(answer_list[0], dict):
                        ans_type = answer_list[0].get('AnswerType', 'Entity')
                        answer_type = 'time' if ans_type == 'Value' else 'entity'
                    else:
                        answer_type = 'unknown'
                else:
                    # MultiTQ: directly use answer_type field
                    answer_type = test_item.get('answer_type', 'unknown')
                
                answer_type_map[qid] = answer_type
        except:
            answer_type_map = {}
        
        # calculation based answer types (entity/time) - real-time calculation
        answer_stats = {}
        for qid, is_correct in all_results:
            answer_type = answer_type_map.get(qid, 'unknown')
            if answer_type not in answer_stats:
                answer_stats[answer_type] = {'total': 0, 'correct': 0}
            answer_stats[answer_type]['total'] += 1
            if is_correct:
                answer_stats[answer_type]['correct'] += 1
        
        # sort by success rate and display
        sorted_answers = sorted(answer_stats.items(), key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0, reverse=True)
        for answer_type, stats in sorted_answers:
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total'] * 100
                print(f"  {answer_type:<10} | {stats['total']:<6} | {stats['correct']:<6} | {accuracy:>6.2f}%")
        
        # calculation based time levels (day/month/year) - real-time calculation
        print(f"\ncalculation based time levels (day/month/year) - real-time calculation:")
        print(f"  {'Time Level':<10} | {'Total':<6} | {'Correct':<6} | {'Success Rate'}")
        print("  " + "-" * 40)
        
        # load time levels mapping from test.json
        try:
            time_level_map = {}
            for test_item in test_data_list:
                qid = str(test_item.get(id_field))
                # ✅ TimeQuestions has no time_level field
                if TPKGConfig.DATASET == 'TimeQuestions':
                    time_level = 'N/A'  # TimeQuestions has no time_level field
                else:
                    time_level = test_item.get('time_level', 'unknown')
                time_level_map[qid] = time_level
        except:
            time_level_map = {}
        
        # calculation based time levels (day/month/year) - real-time calculation
        time_stats = {}
        for qid, is_correct in all_results:
            time_level = time_level_map.get(qid, 'unknown')
            if time_level not in time_stats:
                time_stats[time_level] = {'total': 0, 'correct': 0}
            time_stats[time_level]['total'] += 1
            if is_correct:
                time_stats[time_level]['correct'] += 1
        
        # sort by success rate and display
        sorted_times = sorted(time_stats.items(), key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0, reverse=True)
        for time_level, stats in sorted_times:
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total'] * 100
                print(f"  {time_level:<10} | {stats['total']:<6} | {stats['correct']:<6} | {accuracy:>6.2f}%")
        
        conn.close()
        
        print("\n" + "=" * 80)
        print("✅ query completed")
        print("=" * 80)
        
        sys.exit(0)
    
    # ========== build configuration dictionary (do not use environment variables)==========
    
    # preset configuration
    if args.preset:
        print(f"🔧 use preset configuration: {args.preset}")
    
    # ========== print configuration summary ==========
    print("\n" + "=" * 80)
    print("running configuration summary")
    print("=" * 80)
    
    if args.preset:
        print(f"preset: {args.preset}")
    if args.name:
        print(f"experiment name: {args.name}")
    
    # core parameters
    config_items = []
    if args.retries is not None:
        config_items.append(f"retry={args.retries}")
    if args.depth is not None:
        config_items.append(f"depth={args.depth}")
    if args.branches is not None:
        config_items.append(f"branches={args.branches}")
    
    # feature switches
    features = []
    if args.hybrid is True:
        features.append("hybrid retrieval")
    # experience pool and template learning have been removed
    if args.unified_knowledge is True:
        features.append("unified knowledge store")
    
    if config_items:
        print(f"parameters: {', '.join(config_items)}")
    if features:
        print(f"features: {', '.join(features)}")
    
    # experiment range
    range_items = []
    if args.questions is not None:
        range_items.append(f"{args.questions} questions")
    if args.skip is not None:
        range_items.append(f"skip {args.skip} questions")
    if args.type:
        range_items.append(f"type={args.type}")
    
    if range_items:
        print(f"range: {', '.join(range_items)}")
    
    if args.model:
        print(f"model: {args.model}")
    
    print("=" * 80 + "\n")
    
    # ========== directly import and run main module (pass parameters)==========
    
    # set sys.argv so that main.py can access parameters
    sys.argv = ['main.py']
    
    # directly import main module and pass configuration
    try:
        # import main module
        import main as main_module
        
        # directly call main, pass args
        main_module.run_experiment(args)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ keyboard interrupt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

