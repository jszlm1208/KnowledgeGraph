import argparse
import os
import sys
import json
import shutil
import datetime
import numpy as np
import pickle
from tqdm import tqdm
from azureml.core import Run
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core import Dataset
from feature_walk_probability import h2r, h2t, r2h, r2t, t2h, t2r
from feature_dump import get_rrt_feat, get_h2t_t2h_feat, get_t2h_h2t_feat, get_h2t_h2t_feat, get_hht_feat, get_r2t_h2r_feat, get_r2t_feat, get_rrh_feat

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default="outputs", help="folder path for all outputs")
parser.add_argument("--gpu", type=str, default="0,1,2", help="gpu id")
parser.add_argument("--aml", type=bool, default=True)
parser.add_argument("--data_path", type=str, default=False)

args = parser.parse_args()

print('output path is {}'.format(args.output_path))
print('gpu is {}'.format(args.gpu))
print('aml is {}'.format(args.aml))
print('data_path is {}'.format(args.data_path))

def main(gpu, output_path, data_path):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    script_path = os.path.dirname(os.path.abspath(sys.argv[0])) ##dataset file path?
    print('script_path is {}'.format(script_path))
    output_path = output_path
    
    #for feature_walk_probability
    
    hrt_path = os.path.join(data_path, "wikikg90m-v2/processed/train_hrt.npy")
    train_hrt = np.load(hrt_path, mmap_mode="r")
    
    data_h2r = h2r(train_hrt)
   # data_h2t = h2t(train_hrt)
   # data_r2h = r2h(train_hrt)
   # data_r2t = r2t(train_hrt)
   # data_t2h = t2h(train_hrt)
   # data_t2r = t2r(train_hrt)

    feature_output_path = os.path.join(output_path,"feature_output")
    

    if not os.path.exists(output_path):
            os.mkdir(output_path)
    if not os.path.exists(feature_output_path):
            os.mkdir(feature_output_path)
    pickle.dump(data_h2r, open(os.path.join(output_path,"feature_output/h2r_prob.pkl"), "wb"))
   # pickle.dump(data_h2t, open(os.path.join(output_path,"feature_output/h2t_prob.pkl"), "wb"))
   # pickle.dump(data_r2h, open(os.path.join(output_path,"feature_output/r2h_prob.pkl"), "wb"))
   # pickle.dump(data_r2t, open(os.path.join(output_path,"feature_output/r2t_prob.pkl"), "wb"))
   # pickle.dump(data_t2h, open(os.path.join(output_path,"feature_output/t2h_prob.pkl"), "wb"))
   # pickle.dump(data_t2r, open(os.path.join(output_path,"feature_output/t2r_prob.pkl"), "wb"))

    del data_h2r
   # del data_h2t
   # del data_r2h
   # del data_r2t
   # del data_t2h
   # del data_t2r


    #--------------------------------for dump feature---------------------------------------------------------------------
    """
    val_hr_path = os.path.join(data_path, "wikikg90m-v2/processed/val_hr.npy")
    val_t_candidate_path = os.path.join(data_path, "wikikg90m-v2/processed/val_t_candidate.npy")
    test_hr_path = os.path.join(data_path, "wikikg90m-v2/processed/test-dev_hr.npy")
    test_t_candidate_path = os.path.join(data_path, "wikikg90m-v2/processed/test_t_candidate.npy")
    val_hr = np.load(val_hr_path, mmap_mode="r")
    val_t_candidate = np.load(val_t_candidate_path, mmap_mode="r")
    test_hr = np.load(test_hr_path, mmap_mode="r")
    test_t_candidate = np.load(test_t_candidate_path, mmap_mode="r")

    val_output_path = os.path.join(output_path,"valid_feats")
    test_output_path = os.path.join(output_path,"test_feats")

    data_rrt_val = get_rrt_feat(val_t_candidate,val_hr,data_t2r,data_r2t)
    data_h2t_t2h_val = get_h2t_h2t_feat(val_t_candidate,val_hr,data_h2t,data_t2h)
    data_t2h_h2t_val = get_t2h_h2t_feat(val_t_candidate,val_hr,data_h2t,data_t2h)
    data_h2t_h2t_val = get_h2t_h2t_feat(val_t_candidate,val_hr,data_h2t,data_t2h)
    data_hht_val = get_hht_feat(val_t_candidate,val_hr,data_h2t,data_t2h)
    data_r2t_h2r_val = get_r2t_h2r_feat(val_t_candidate,val_hr,data_h2r,data_r2t,data_r2h)
    data_r2t_val = get_r2t_feat(val_t_candidate,val_hr,data_r2t)
    data_rrh_val = get_rrh_feat(val_t_candidate,val_hr,data_h2r,data_r2h,data_t2r,data_r2t)

    if not os.path.exists(val_output_path):
        os.mkdir(val_output_path)
    np.save(os.path.join(val_output_path,"rrt_feat.npy"),data_rrt_val)
    np.save(os.path.join(val_output_path,"h2t_t2h_feat.npy"),data_h2t_t2h_val)
    np.save(os.path.join(val_output_path,"t2h_h2t_feat.npy"),data_t2h_h2t_val)
    np.save(os.path.join(val_output_path,"h2t_h2t_feat.npy"),data_h2t_h2t_val)
    np.save(os.path.join(val_output_path,"hht_feat.npy"),data_hht_val)
    np.save(os.path.join(val_output_path,"r2t_h2r_feat.npy"),data_r2t_h2r_val)
    np.save(os.path.join(val_output_path,"r2t_feat.npy"),data_r2t_val)
    np.save(os.path.join(val_output_path,"rrh_feat.npy"),data_rrh_val)
    del data_rrt_val
    del data_h2t_t2h_val
    del data_t2h_h2t_val
    del data_h2t_h2t_val
    del data_hht_val
    del data_r2t_h2r_val
    del data_r2t_val
    del data_rrh_val

    data_rrt_test = get_rrt_feat(test_t_candidate,test_hr,data_t2r,data_r2t)
    data_h2t_t2h_test = get_h2t_h2t_feat(test_t_candidate,test_hr,data_h2t,data_t2h)
    data_t2h_h2t_test = get_t2h_h2t_feat(test_t_candidate,test_hr,data_h2t,data_t2h)
    data_h2t_h2t_test = get_h2t_h2t_feat(test_t_candidate,test_hr,data_h2t,data_t2h)
    data_hht_test = get_hht_feat(test_t_candidate,test_hr,data_h2t,data_t2h)
    data_r2t_h2r_test = get_r2t_h2r_feat(test_t_candidate,test_hr,data_h2r,data_r2t,data_r2h)
    data_r2t_test = get_r2t_feat(test_t_candidate,test_hr,data_r2t)
    data_rrh_test = get_rrh_feat(test_t_candidate,test_hr,data_h2r,data_r2h,data_t2r,data_r2t)


    if not os.path.exists(test_output_path):
        os.mkdir(test_output_path)
    np.save(os.path.join(test_output_path,"rrt_feat.npy"),data_rrt_test)
    np.save(os.path.join(test_output_path,"h2t_t2h_feat.npy"),data_h2t_t2h_test)
    np.save(os.path.join(test_output_path,"t2h_h2t_feat.npy"),data_t2h_h2t_test)
    np.save(os.path.join(test_output_path,"h2t_h2t_feat.npy"),data_h2t_h2t_test)
    np.save(os.path.join(test_output_path,"hht_feat.npy"),data_hht_test)
    np.save(os.path.join(test_output_path,"r2t_h2r_feat.npy"),data_r2t_h2r_test)
    np.save(os.path.join(test_output_path,"r2t_feat.npy"),data_r2t_test)
    np.save(os.path.join(test_output_path,"rrh_feat.npy"),data_rrh_test)

    del data_rrt_test
    del data_h2t_t2h_test
    del data_t2h_h2t_test
    del data_h2t_h2t_test
    del data_hht_test
    del data_r2t_h2r_test
    del data_r2t_test
    del data_rrh_test
    
    """
    """
    aml_run = Run.get_context()
    ws=aml_run.experiment.workspace
    datastore = ws.get_default_datastore()
    datastore.upload(output_path, 'OTE_manual_features/feature_output/',
        overwrite=True)    
    """


if __name__ == '__main__':
    if args.aml:
        from azureml.core import Run

        run = Run.get_context()
    else:
        run = None

    main(args.gpu,
         args.output_path,
         args.data_path)