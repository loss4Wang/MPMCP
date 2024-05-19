# Copyright (c) OpenMMLab. All rights reserved.
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.dataset import ConcatDataset, process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import (crime_kg_assitant_map_fn,
                                    law_reference_map_fn,
                                    template_map_fn_factory)
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path =  'internlm/internlm2-chat-7b-sft'

# Data
# download data from https://github.com/LiuHC0428/LAW-GPT
sft_data_folder = "/localfast/xiaowang/LJPtask/Data/SFT/"
train1_t2_sft_path = sft_data_folder+ 'train1_t2_sft.json'
prompt_template = PROMPT_TEMPLATE.internlm2_chat 
max_length = 32768 #! Q1: Can we set larger max_length? 32768, 2048
pack_to_max_length = False

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 16 #! 一个mini-batch需要多少个micro-batch计算的梯度累加
dataloader_num_workers = 0 
max_epochs = 3 #! default: 3, Q2: When should we stop training?
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Evaluate the generation performance during the training
system_prompt = "You are an AI assistant whose name is InternLM (书生·浦语).\n- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."
evaluation_freq = 500
SYSTEM = system_prompt #! Q3: How do we set the system template？
val1_t2_sft_q1 = "请你模拟法官依据下面事实和被告人预测被告的罪名（一个）。\n被告人：陈红花\n事实：2019年10月7日15时许，被告人陈红花来到被害人王某1位于莆田市涵江区的租住处，盗走卧室内一提包中的现金500元，后逃离现场。 2019年10月19日13时许，被告人陈红花来到被害人刘某位于莆田市涵江区的住处，盗走一楼杂物间内一白色提包中的现金600元，后逃离现场。 2019年10月20日8时许，被告人陈红花来到被害人凌某位于莆田市涵江区的住处，盗走二楼房间内一红色提包中的现金1040元，后逃离现场。 案发后，被告人陈红花于2019年10月22日被公安机关抓获归案。上述被盗现金已被公安机关追回并发还各被害人。 审理另查明，2020年3月16日，被告人陈红花向莆田市涵江区人民检察院签署认罪认罚具结书。 上述事实，被告人陈红花在开庭审理过程中亦无异议，且有受案登记表、立案决定书、到案经过、户籍证明、户籍信息、人员基本信息、中华人民共和国外国人居留许可、提取笔录、扣押清单、扣押决定书、发还清单，证人王某2、Ｇ某的证言，被害人王某1、凌某、刘某的陈述，辨认笔录、指认照片、现场勘验笔录、现场照片等证据证实，足以认定。 "
evaluation_inputs = [val1_t2_sft_q1] #! Q4: Don't need have eval metics or data?

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right') #! train的时候padding设置为right(如果模型使用absolute position embedding时用left会出错)，inference的时候padding设置为left

model = dict(
    type=SupervisedFinetune,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,)) # bfloat16, float16

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train1_t2_sft = dict(
    type=process_hf_dataset,
    dataset=dict(
        type=load_dataset,
        path='json',
        data_files=dict(train=train1_t2_sft_path)),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length
)

train_dataset = dict(
    type=ConcatDataset, datasets=[train1_t2_sft])

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))

#######################################################################xtuner list-cfg
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float32')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        T_max=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per epoch.
    checkpoint=dict(type=CheckpointHook, interval=1),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
