import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
# from utils.data_preparation import read_history_cycle_dataset
from datetime import datetime
from utils.base_trainer import BaseTrainer
from model.MPCDI.MPCDI import MPCDI as MPCDI

from thop import profile
from torch.optim import Adam
from dataloader.mp_dataloader import get_dataloader
from utils.losses import MixLoss

class Trainer(BaseTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args)

    def __generate_record_name__(self):
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 前缀名称
        record_name = f'{self.args.model_name}_{self.args.dataset}_{self.args.seq_len}_{self.args.missing_pattern}_{time_str}'
        return record_name

    def __init_optimizer__(self):
        optimizer = None # 初始化为None

        if list(self.model.parameters()):
            # optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
            optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-6)
            # optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-6)
        return optimizer

    def __init_scheduler__(self):
        scheduler = None
        if self.optimizer:
            # torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=opt.lr_decay_steps, gamma=opt.lr_scheduler_rate)
            # scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-5)  # 如果没用scheduler，直接返回None
            max_epoch = self.args.max_epoch
            p1 = int(0.5 * max_epoch)
            p2 = int(0.75 * max_epoch)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[p1, p2], gamma=0.1)
        return scheduler

    def __get_loss_func__(self): # 得到loss函数
        loss = MixLoss(self.args.alpha, self.args.beta)
        return loss



    def __build_model__(self):  # 构建模型
        def get_model_class(name):
            model_dict = {
                'MPCDI': MPCDI
            }
            return model_dict[name]

        model = get_model_class(self.args.model_name)(self.args).cuda()  # 根据参数得到model的实例对象，并将其转移到GPU上
        if self.args.init_param:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)
        return model

    def __get_loader__(self):  # 根据模型，要调用不同的数据记载方法
        train_loader = None
        vali_loader = None
        test_loader = None
        scaler = None

        if self.args.model_name in ['MPCDI']:
            train_loader, vali_loader, test_loader, scaler = get_dataloader(self.args.seq_file,
                                                                               self.args.batch_size,
                                                                               self.args.train_ratio,
                                                                               self.args.val_ratio,
                                                                               self.args.test_ratio,
                                                                               self.args.seq_len,  # 窗口长度
                                                                               self.args.train_missing_ratio,
                                                                               self.args.val_missing_ratio,
                                                                               self.args.test_missing_ratio,
                                                                               missing_pattern=self.args.missing_pattern,
                                                                               interval=self.args.seq_len, # 非重叠窗口
                                                                               normalizer=self.args.normalizer,
                                                                               column_wise=self.args.column_wise)


        return train_loader, vali_loader, test_loader, scaler

    def calculate_complexity(self, model, inputs):

        model.eval()

        # FLOPs + Params
        flops, params = profile(
            model,
            inputs=inputs,
            verbose=False
        )

        # 转换成 M
        params_m = params / 1e6
        flops_m = flops / 1e9

        # print("=" * 40)
        # print(f"Params : {params_m:.4f} M")
        # print(f"FLOPs  : {flops_m:.4f} G")
        # print("=" * 40)

        return params_m, flops_m

    def __process_model__(self, data, current_epoch, mode='train'):  # 这一步返回计算的loss

        if self.args.model_name in ['MPCDI']:

            if mode=='train':
                forward_noise, noise, target_mask = self.model(data, is_train=True)


                loss = self.model.caculate_loss(forward_noise, noise, target_mask)
                return loss

            elif mode=='val':
                loss_sum = 0
                for t in range(self.args.num_steps):  # calculate loss for all t
                    forward_noise, noise, target_mask = self.model(data, is_train=False, set_t=t)
                    loss = self.model.caculate_loss(forward_noise, noise, target_mask)
                    loss_sum += loss.item()
                return loss_sum / self.args.num_steps

            else:
                samples, X_Tilde, eval_mask = self.model.impute(data, self.args.n_samples)
                # 在这里进行还原
                samples = self.scaler.inverse_transform(samples)
                X_Tilde = self.scaler.inverse_transform(X_Tilde)

                return samples, X_Tilde, eval_mask



