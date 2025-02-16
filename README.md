# PoseCtrl

单卡：python train_pose_control.py

多卡：accelerate launch --num_processes 8 --multi_gpu --mixed_precision "fp16" tutorial_train.py 

然后测试训练的时候改一下 python train_pose_control.py --save_steps 10 先保存一下权重测试inference
