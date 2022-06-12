cd ..

python main.py --dataset APY --gpu 0 --relation_nepoch 50 --relation_batchsize 200 --relation_wd 3e-6 --relation_lr 0.001 \
  --aug_lr 30 --aug_wd 0 --aug_mt 0 --loops_adv 20 --entropy_weight 10 --gen_nepoch 600 --lr 0.0003 --batchsize 200 --manualSeed 6152 \
  --weight_decay 1e-5 --step_size 20 --scheduler_gamma 0.55 --num_coupling_layers 5 --prototype 3 --pi 0.02 --dropout 0 \
   --evl_interval 300 --classifier_lr 0.001 --cls_batch 1200 --cls_step 20 --nSample 9000 --eval_start 5000 --disp_interval 200 --aug --mse_weight 1

python main.py --dataset APY --gpu 0 --relation_nepoch 50 --relation_batchsize 200 --relation_wd 3e-6 --relation_lr 0.001 \
  --aug_lr 30 --aug_wd 0 --aug_mt 0 --loops_adv 20 --entropy_weight 10 --gen_nepoch 600 --lr 0.0003 --batchsize 200 --manualSeed 6152 \
  --weight_decay 1e-5 --step_size 20 --scheduler_gamma 0.55 --num_coupling_layers 5 --prototype 3 --pi 0.02 --dropout 0  --zsl \
   --evl_interval 300 --classifier_lr 0.001 --cls_batch 1200 --cls_step 20 --nSample 9000 --eval_start 5000 --disp_interval 200 --aug --mse_weight 1
