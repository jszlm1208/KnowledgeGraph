
export DGLBACKEND=pytorch
output_path=./model_output/ote40
mkdir -p $output_path

CUDA_VISIBLE_DEVICES=0,1 \
    dglke_train \
    --model_name OTE \
    --data_path dataset \
    --hidden_dim 200 --gamma 12 --ote_size 40 --lr 0.1 --regularization_coef 1e-9 \
    --valid -adv --mix_cpu_gpu --num_proc 2 --num_thread 8 \
    --gpu 0 1 \
    --lr_decay_rate 0.98 \
    --lr_decay_interval 10000 \
    --scale_type 2 \
    --max_step 6000000 \
    --async_update --force_sync_interval 10 \
    --eval_interval 50000 \
    --mlp_lr 0.00002 \
    --seed $RANDOM \
    --batch_size 1000 --neg_sample_size 1000 \
    --print_on_screen --encoder_model_name concat  --save_path ${output_path} 
