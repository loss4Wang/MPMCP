specify args for train
set training_args (TrainingArguments)
set seed
set work_dir (output_dir)
set deepspeed

# On a single GPU
xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
# On multiple GPUs
(DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
(SLURM) srun ${SRUN_ARGS} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --launcher slurm --deepspeed deepspeed_zero2

CUDA_VISIBLE_DEVICES=7 nohup xtuner train /localfast/xiaowang/LJPtask/train/internlm2/internlm2_chat_7b_sft_ljp_single.py --deepspeed deepspeed_zero2_offload > t2_d2_sft_ljp_single.log &

CUDA_VISIBLE_DEVICES=7 nohup xtuner train /localfast/xiaowang/LJPtask/train/internlm2/internlm2_chat_7b_sft_ljp_single.py > dp1_t2_d2_sft_ljp_single.log &

/localfast/xiaowang/LJP\ Task/train/internlm2/
/localfast/xiaowang/anaconda3/envs/LJP/lib/python3.11/site-packages/xtuner/configs/deepspeed

# T2D1
CUDA_VISIBLE_DEVICES=4,5 nohup xtuner train /localfast/xiaowang/LJPtask/train/internlm2/internlm2_chat_7b_sft_ljp_t2_d2.py --deepspeed deepspeed_zero2_offload > dp2offload_t2_d1_sft_ljp_single.log &

# T3D1
CUDA_VISIBLE_DEVICES=5 MASTER_PORT=29400 nohup xtuner train /localfast/xiaowang/LJPtask/train/internlm2/internlm2_chat_7b_sft_ljp_t3_d1.py --deepspeed deepspeed_zero2_offload > t3_d1_sft_ljp_single.log &

# T4D1
CUDA_VISIBLE_DEVICES=2 nohup xtuner train /localfast/xiaowang/LJPtask/train/internlm2/internlm2_chat_7b_sft_ljp_t4_d1.py --deepspeed deepspeed_zero2_offload > t4_d1_sft_ljp_multi.log &