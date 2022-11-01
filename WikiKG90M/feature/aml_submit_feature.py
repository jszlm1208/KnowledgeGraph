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
    "num_proc":"1"
    }
    subscription_id = '389384f8-9747-48b4-80a2-09f64d0a0dd7'
    resource_group = 'BizQA-WUS3-RG-GpuClusterA100'
    workspace_name = 'BizQA-Dev-WUS3-AML'

    ws = Workspace(subscription_id, resource_group, workspace_name)
    # ws = Workspace.from_config('config.json')
    ds = ws.get_default_datastore()
    #data_path = ds.path('ogb/ogbl_wikikg2/smore_folder_full/sample/wikikg90m-v2/processed/').as_mount()
    #data_path.path_on_compute = 'tmp/data'
    #path = Dataset.get_by_name(ws, name=data_folder).as_named_input('data_folder').as_mount()
    data_path = Dataset.get_by_name(ws, name='Wikikg_v2_full').as_named_input('data_folder').as_mount()
    
    # run experiments
    
    
    config = ScriptRunConfig(source_directory='./',
                                     script='main_feature.py',
                                     arguments=['--aml',True,
                                                '--data_path',data_path
                                                ],
                                     compute_target='BizQADevWUS3CPU')

    #config.run_config.data_references[data_path.data_reference_name] = data_path.to_config()
    env = ws.environments['NOTE']
    experiment = Experiment(workspace=ws, name='NOTE_features')
    config.run_config.environment = env
    run = experiment.submit(config,tags=run_tags)
    aml_url = run.get_portal_url()
    print(aml_url)

if __name__ == '__main__':
    main()