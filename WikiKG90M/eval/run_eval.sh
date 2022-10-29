data_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full"
model_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/saved_models"
cand_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/wikikg90m-v2/processed"

python eval.py \
    '--data_path', data_path,
    '--cand_path', cand_path,
    '--cand_size', -1,
    '--model_name', "TransE_l2",
    '--model_prefix', "TransE_l2_wikikg90m_shallow_d_600_g_10.00",
    '--model_path', model_path,
    '--dataset', "wikikg90m",
    '--neg_sample_size_eval', 200,
    '--batch_size_eval', 200,
    '--save_path', "outputs",
    '--LRE',
    '--LRE_rank', 200