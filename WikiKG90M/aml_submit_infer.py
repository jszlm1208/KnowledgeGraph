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
    "dataset": "wikikg90m_v2"
    }
    subscription_id = 'xxx'
    resource_group = 'xxx'
    workspace_name = 'xxx'

    ws = Workspace(subscription_id, resource_group, workspace_name)
    #ws = Workspace.from_config('config.json') ####need point the config.json path, set .from_config() if run in Notbook
    ds = ws.get_default_datastore()
    #data_path = ds.path('ogb/ogbl_wikikg2/smore_folder_full/sample/wikikg90m-v2/processed/').as_mount()
    #data_path.path_on_compute = 'tmp/data'
    #path = Dataset.get_by_name(ws, name=data_folder).as_named_input('data_folder').as_mount()
    data_path = Dataset.get_by_name(ws, name='Wikikg_v2_full').as_named_input('data_folder').as_mount()
    model_path = Dataset.get_by_name(ws, name='OTE_model_output').as_named_input('model_folder').as_mount()
    
    # run experiments
    
    
    config = ScriptRunConfig(source_directory='./',
                                     script='python/infer.py',
                                     arguments=['--model_path',model_path,
                                                '--data_path',data_path,
                                                '--infer_test',
                                                '--model_prefix','OTE_wikikg90m_concat_d_240_g_12.00'
                                                ],
                                     compute_target='BizQADevWUS3A100')

    #config.run_config.data_references[data_path.data_reference_name] = data_path.to_config()
    env = ws.environments['dgl']
    experiment = Experiment(workspace=ws, name='NOTE_eval')
    config.run_config.environment = env
    run = experiment.submit(config,tags=run_tags)
    aml_url = run.get_portal_url()
    print(aml_url)

if __name__ == '__main__':
    main()