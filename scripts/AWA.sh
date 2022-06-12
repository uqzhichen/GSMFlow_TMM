cd ..

python main.py --dataset AWA2 --gpu 1 --relation_nepoch 50 --relation_batchsize 256 --relation_wd 3e-6 --relation_lr 0.0003 \
  --aug_lr 100 --aug_wd 0 --aug_mt 0 --loops_adv 30 --entropy_weight 20 --gen_nepoch 1000 --lr 0.0003 --batchsize 256 --manualSeed 7848 \
  --weight_decay 3e-6 --step_size 17 --scheduler_gamma 0.92 --num_coupling_layers 3 --prototype 10 --pi 0.15 --dropout 0 \
  --evl_interval 500 --classifier_lr 0.005 --cls_batch 1200 --cls_step 30 --nSample 50000 --eval_start 60000 \
  --disp_interval 200 --aug  --mse_weight 1

python main.py --dataset AWA2 --gpu 1 --relation_nepoch 50 --relation_batchsize 256 --relation_wd 3e-6 --relation_lr 0.0003 \
  --aug_lr 100 --aug_wd 0 --aug_mt 0 --loops_adv 30 --entropy_weight 20 --gen_nepoch 1000 --lr 0.0003 --batchsize 256 --manualSeed 4308 \
  --weight_decay 3e-6 --step_size 17 --scheduler_gamma 0.92 --num_coupling_layers 3 --prototype 10 --pi 0.15 --dropout 0 --zsl \
  --evl_interval 500 --classifier_lr 0.005 --cls_batch 1200 --cls_step 30 --nSample 50000 --eval_start 60000 \
  --disp_interval 200 --aug  --mse_weight 1
