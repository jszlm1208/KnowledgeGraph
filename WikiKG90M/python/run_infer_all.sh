export DGLBACKEND=pytorch

export CUDA_VISIBLE_DEVICES=0,1,2,3

export data_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full"
export model_path="/mnt/data/ogbl_wikikg2_recent/ogbl_wikikg2/smore_folder_full/saved_models/ote/OTE_wikikg90m_concat_d_240_g_12.00"

infer(){

python models/infer.py --data_path $data_path --infer_valid --model_path $model_path &
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

data_path=dataset


for line in model_output/*/*; do
    echo $line
    model_path=$line
    infer
    merge
done
