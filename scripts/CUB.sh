cd ..

python main.py --dataset CUB --gpu 0 --relation_nepoch 100 --relation_batchsize 256 --relation_wd 1e-6 --relation_lr 0.0001 \
  --aug_lr 20 --aug_wd 0 --aug_mt 0 --loops_adv 20 --entropy_weight 10 --gen_nepoch 1500 --lr 0.0001 --batchsize 256 --manualSeed 6167 \
  --weight_decay 3e-5 --step_size 25 --scheduler_gamma 0.8 --num_coupling_layers 3 --prototype 1 --pi 0.05 --dropout 0 \
  --evl_interval 300 --classifier_lr 0.003 --cls_batch 600 --cls_step 50 --nSample 600 --eval_start 10000 --disp_interval 200 \
  --mse_weight 1 --aug #

python main.py --dataset CUB --gpu 0 --relation_nepoch 100 --relation_batchsize 256 --relation_wd 1e-6 --relation_lr 0.0001 \
  --aug_lr 20 --aug_wd 0 --aug_mt 0 --loops_adv 20 --entropy_weight 10 --gen_nepoch 1500 --lr 0.0001 --batchsize 256 --manualSeed 8264 \
  --weight_decay 3e-5 --step_size 25 --scheduler_gamma 0.8 --num_coupling_layers 3 --prototype 1 --pi 0.05 --dropout 0 --zsl \
  --evl_interval 300 --classifier_lr 0.003 --cls_batch 600 --cls_step 50 --nSample 600 --eval_start 10000 --disp_interval 200 \
  --mse_weight 1 --aug #
