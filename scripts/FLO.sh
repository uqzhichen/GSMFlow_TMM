cd ..

python main.py --dataset FLO --gpu 1 --relation_nepoch 60 --relation_batchsize 64 --relation_wd 1e-6 --relation_lr 0.0003 \
  --aug_lr 20 --aug_wd 0 --aug_mt 0 --loops_adv 20 --entropy_weight 1 --gen_nepoch 1000 --lr 0.0001 --batchsize 256 --manualSeed 6289 \
  --weight_decay 1e-4 --step_size 30 --scheduler_gamma 0.8 --num_coupling_layers 3 --prototype 3 --pi 0.1 --dropout 0 \
   --evl_interval 300 --classifier_lr 0.005 --cls_batch 1000 --cls_step 30 --nSample 3000 --eval_start 2000 --disp_interval 200 --aug  --mse_weight 1

python main.py --dataset FLO --gpu 1 --relation_nepoch 60 --relation_batchsize 64 --relation_wd 1e-6 --relation_lr 0.0003 \
  --aug_lr 20 --aug_wd 0 --aug_mt 0 --loops_adv 20 --entropy_weight 1 --gen_nepoch 1000 --lr 0.0001 --batchsize 256 --manualSeed 3982 \
  --weight_decay 1e-4 --step_size 30 --scheduler_gamma 0.8 --num_coupling_layers 3 --prototype 3 --pi 0.1 --dropout 0 --zsl \
   --evl_interval 300 --classifier_lr 0.005 --cls_batch 1000 --cls_step 30 --nSample 3000 --eval_start 2000 --disp_interval 200 --aug  --mse_weight 1
