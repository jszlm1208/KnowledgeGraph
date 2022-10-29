from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core import Dataset
from azureml.core import Datastore
from azureml.core import Run
from os import path
import argparse

def main():
    """
        run_note: experiment name
        data_folder: dataset name
        model_data_folder:
    """
    run_tags = {
    "dataset": "wikikg90m_v2",
    "method": "on full dataset",
    "exp": "evaluation"
    }
    ws = Workspace.from_config('config.json') 
    ds = ws.get_default_datastore()
   # data_path = Dataset.get_by_name(ws, name='OTE_model_output').as_named_input('data_folder').as_mount()
   # cand_path = Dataset.get_by_name(ws, name='cand_data_path').as_named_input('cand_folder').as_mount()
    data_path = Dataset.get_by_name(ws, name='TransE_1msteps_model').as_named_input('data_folder').as_mount()
    cand_path = Dataset.get_by_name(ws, name='cand_data_path').as_named_input('cand_folder').as_mount()
    
    # run experiments
    
    
    config = ScriptRunConfig(source_directory='./',
                                     script='python/dglke/main.py',
                                     arguments=['save_test_submission',
                                                '--aml', True,
                                                '--with_test', True,
                                                '--path', data_path,
                                                '--model_prefix', "TransE_l2_wikikg90m_shallow_d_600_g_10.00",
                                                '--cand_path', cand_path,
                                                '--num_proc', 1
                                                ],
                                     compute_target='BizQADevWUS3CPU')

    #config.run_config.data_references[data_path.data_reference_name] = data_path.to_config()
    env = ws.environments['dgl']
    experiment = Experiment(workspace=ws, name='NOTE_eval')
    config.run_config.environment = env
    run = experiment.submit(config,tags=run_tags)
    aml_url = run.get_portal_url()
    print(aml_url)

if __name__ == '__main__':
    main()