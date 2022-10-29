import os
import pickle
import json
import numpy as np
import sys
from ogb.lsc import WikiKG90Mv2Dataset, WikiKG90Mv2Evaluator
import pdb
from collections import defaultdict
import torch.nn.functional as F
import torch

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


# python save_test_submission.py $SAVE_PATH $NUM_PROC $MODE
#def get_test_predictions(path,valid_candidate_path,test_candidate_path,mode="test-dev",num_proc=1,with_test = True): 
def get_test_predictions(args): 
    if args.aml:
        from azureml.core import Run
        run = Run.get_context()
    else:
        run = None
    
    path = os.path.join(args.path,args.model_prefix)
    valid_candidate_path =os.path.join(args.cand_path, "val_t_candidate_50000.npy")
    test_candidate_path =os.path.join(args.cand_path, "test-dev_t_candidate_50000.npy")
    #valid_candidates = np.load(valid_candidate_path,allow_pickle=True).item()
    #valid_candidates = list(valid_candidates.values())
    #test_candidates = np.load(test_candidate_path,allow_pickle=True).item()
    #test_candidates = list(test_candidates.values())
    valid_candidates = np.load(valid_candidate_path)
    test_candidates = np.load(test_candidate_path)
    all_file_names = os.listdir(path)
    test_file_names = [name for name in all_file_names if '.pkl' in name and 'test' in name]
    valid_file_names = [name for name in all_file_names if '.pkl' in name and 'valid' in name]
    steps = [(name.split('.')[0].split('_')[-1])
             for name in valid_file_names if 'valid_0' in name]
    steps.sort()
    print(steps)
    evaluator = WikiKG90Mv2Evaluator()
    device = torch.device('cpu')

    all_test_dicts = []
    best_valid_mrr = -1
    best_valid_idx = -1
    
    for i, step in enumerate(steps):
        print(step)
        valid_result_dict = defaultdict(lambda: defaultdict(list))
        test_result_dict = defaultdict(lambda: defaultdict(list))
        for proc in range(args.num_proc):
            valid_result_dict_proc = torch.load(os.path.join(
                path, "valid_{}_{}.pkl".format(proc, step)), map_location=device)
            for result_dict_proc, result_dict in zip([valid_result_dict_proc], [valid_result_dict]):
                for key in result_dict_proc['h,r->t']:
                    result_dict['h,r->t'][key].append(result_dict_proc['h,r->t'][key].numpy())
            if args.with_test:
                test_result_dict_proc = torch.load(os.path.join(
                    path, "test_{}_{}.pkl".format(proc, step)), map_location=device)
                for result_dict_proc, result_dict in zip([test_result_dict_proc], [test_result_dict]):
                    for key in result_dict_proc['h,r->t']:
                        result_dict['h,r->t'][key].append(result_dict_proc['h,r->t'][key].numpy())

        for result_dict in [valid_result_dict]:
            for key in result_dict['h,r->t']:
                if key == 't_pred_top10':
                    index = np.concatenate(result_dict['h,r->t'][key], 0)
                    temp = []
                    for ii in range(index.shape[0]):
                        temp.append(np.array(valid_candidates[ii])[index[ii]])
                    result_dict['h,r->t'][key] = np.concatenate(np.expand_dims(temp, 0))
                else:
                    result_dict['h,r->t'][key] = np.concatenate(result_dict['h,r->t'][key], 0)
        if args.with_test:
            for result_dict in [test_result_dict]:
                for key in result_dict['h,r->t']:
                    if key == 't_pred_top10':
                        index = np.concatenate(result_dict['h,r->t'][key], 0)
                        temp = []
                        for ii in range(index.shape[0]):
                            temp.append(np.array(test_candidates[ii])[index[ii]])
                        result_dict['h,r->t'][key] = np.concatenate(np.expand_dims(temp, 0))
                    else:
                        result_dict['h,r->t'][key] = np.concatenate(result_dict['h,r->t'][key], 0) 
        if args.with_test:
            all_test_dicts.append(test_result_dict)
        metrics = evaluator.eval(valid_result_dict)
        metric = 'mrr'
        print("valid-{} at step {}: {}".format(metric, step, metrics[metric]))
        if metrics[metric] > best_valid_mrr:
            best_valid_mrr = metrics[metric]
            best_valid_idx = i
    print(f"Best valid mrr:{best_valid_mrr}")
    """
     if args.with_test:
        best_test_dict = all_test_dicts[best_valid_idx]
        evaluator.save_test_submission(best_test_dict, path, mode)
    """
   

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aml", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether to use AML")
    parser.add_argument("--with_test", type=lambda x: (str(x).lower() == 'true'), default=True, help="whether to use test")
    parser.add_argument("--path", type=str, default="../model_output", help="model output")
    parser.add_argument("--gpu", type=str, default="0,1", help="gpu id, e.g., '0,1,2,3'")
    parser.add_argument('--cand_path', type=str, default='cand_data', help="indicate the candidate path")
    parser.add_argument('--num_proc', type=int, default=1,
                          help='The number of processes to evaluate the model in parallel.'\
                                  'For multi-GPU, the number of processes by default is set to match the number of GPUs.'\
                                  'If set explicitly, the number of processes needs to be divisible by the number of GPUs.')
    parser.add_argument('--model_prefix', type=str, default='OTE_wikikg90m_concat_d_240_g_12.00')

    args = parser.parse_args()
    get_test_predictions(args)
