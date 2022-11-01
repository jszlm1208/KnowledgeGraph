data_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full"
model_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/OTE_model_output" #/OTE_wikikg90m_concat_d_240_g_12.00"
save_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/OTE_model_output/OTE_wikikg90m_concat_d_240_g_12.00"
cand_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/wikikg90m-v2/processed"

bash install_dgl.sh && python eval.py \
    --model_name OTE \
    --data_path $data_path \
    --dataset wikikg90m \
    --scale_type 2 \
    --ote_size 20 \
    --model_prefix OTE_wikikg90m_concat_d_240_g_12.00 \
    --neg_sample_size_eval 200 \
    --batch_size_eval  50 \
    --model_path $model_path \
    --encoder_model_name concat \
    --save_path $dave_path \
    --cand_path $cand_path