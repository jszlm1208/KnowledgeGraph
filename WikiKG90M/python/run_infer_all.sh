export DGLBACKEND=pytorch

export CUDA_VISIBLE_DEVICES=0,1,2,3

export data_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full"
#export model_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/OTE_model_output/OTE_wikikg90m_concat_d_240_g_12.00"
export model_prefix="OTE_wikikg90m_concat_d_240_g_12.00"
export model_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/OTE_model_output" #TransE_l2_wikikg90m_shallow_d_600_g_10.00"

infer(){

python infer.py --data_path $data_path --infer_valid --model_path $model_path --model_prefix $model_prefix &
sleep 600
ps -ef|grep infer.py | awk '{print $2}'| xargs kill -9

python infer.py --data_path $data_path --infer_test --model_path $model_path &
sleep 1200
ps -ef|grep infer.py | awk '{print $2}'| xargs kill -9

}

merge(){
    python merge_score.py $model_path 16 test
    python merge_score.py $model_path 16 valid
}

infer
merge
