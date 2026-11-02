from trainer import Trainer
from utils.config_utils import get_conf_obj, read_yaml, clear_yaml, write_yaml
import pickle
import random
import torch
import numpy as np
import os

batch_config = 'configuration/BatchConfig.yaml'
conf_file_list = read_yaml(batch_config) # 读取批次训练配置文件

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


while conf_file_list and len(conf_file_list) > 0: # > 0 说明还有元素在列表中未弹出
    conf_file = conf_file_list[0]
    print(f'Start Process File: {conf_file}')
    args = get_conf_obj(conf_file)
    set_seed(args.seed) # 固定种子
    trainer = Trainer(args)

    trainer.train()
    trainer.test() # 直接加载权重
    conf_file_list.pop(0) # 进行train, 如果顺利的话弹出
    if len(conf_file_list) == 0:
        clear_yaml(batch_config) # 清空文件
    else:
        write_yaml(batch_config, conf_file_list)










