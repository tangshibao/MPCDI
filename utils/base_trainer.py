import numpy as np
import torch
from torchinfo import summary
from utils.earlystopping import EarlyStopping
from utils.metrics import metric, missed_eval_np
from tqdm import tqdm
from utils.recorder import Recorder
import time
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
# 绘制一个曲线图
from utils.plotter import plot_lines
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast
class BaseTrainer():
    def __init__(self, args):

        self.args = args
        self.train_loader, self.vali_loader, self.test_loader, self.scaler = self.__get_loader__()  # 加载数据集
        self.model = self.__build_model__()  # 获取模型
        self.record_name = self.__generate_record_name__()  # 获取记录的名称
        self.weight_path = f'weight/{self.record_name}.pkl'
        self.loss_function = torch.nn.L1Loss()
        # self.loss_function = torch.nn.MSELoss()  # 损失函数
        self.optimizer = self.__init_optimizer__()  # 初始化优化器
        self.scheduler = self.__init_scheduler__()  # 初始化调节器
        self.loss = self.__get_loss_func__()
        self.es = EarlyStopping(self.weight_path, patience=self.args.patience)  # 早停器
        self.recorder = Recorder(file_dir='record', record_name=self.record_name)  # 文件记录

        # 细节数据
        self.epoch_train_loss = None # 每一轮的训练loss
        self.epoch_val_loss = None # 每一轮的验证loss
        self.current_epoch = None



    def __generate_record_name__(self):
        raise NotImplementedError

    def __init_scheduler__(self):
        raise NotImplementedError

    def __init_optimizer__(self):
        raise NotImplementedError

    def __build_model__(self):  # 构建模型
        raise NotImplementedError

    def __get_loader__(self):  # 根据模型，要调用不同的数据记载方法
        raise NotImplementedError

    def __get_loss_func__(self):
        raise NotImplementedError

    def __process_model__(self, data, current_epoch, mode='train'):  # 模型处理数据方法
        raise NotImplementedError

    def vali(self):
        self.model.eval()
        losses_vali = []
        with torch.no_grad():
            for data in self.vali_loader:
                loss_vali = self.__process_model__(data, self.current_epoch, 'val')
                # loss_vali = self.loss_function(outputs, targets)
                losses_vali.append(loss_vali)
        return np.average(losses_vali)
    def train(self):
        if not self.optimizer or not list(self.model.parameters()): # 如果优化器为None或者是模型没有参数，直接退出训练，说明这里是机器学习模型
            return
        if self.current_epoch is None:
            self.current_epoch = 0
        model_summary = summary(self.model, depth=self.args.summary_depth)
        avg_loss_vali = float('inf')
        # test_mse = float('inf')
        # if self.epoch_train_loss is None:
        self.epoch_train_loss = []
        # if self.epoch_val_loss is None:
        self.epoch_val_loss = []

        while (self.current_epoch < self.args.max_epoch and
               self.es.num_bad_epochs < self.es.patience):
            self.model.train()
            losses_train = []
            msg_bar = tqdm(self.train_loader, total=len(self.train_loader))  # 每一轮都重新封装

            # for batch_no, train_batch in enumerate(
            for data in msg_bar:
                self.model.zero_grad()
                self.optimizer.zero_grad()


                loss_train = self.__process_model__(data, self.current_epoch, 'train')
                # loss_train = self.loss_function(outputs, targets)
                # print(loss_train.item())

                losses_train.append(loss_train.item())

                loss_train.backward()
                if self.args.grad_clip: # 如果设置了裁剪才操作
                    clip_grad_norm_(self.model.parameters(), self.args.clip_max_value) # 梯度裁剪

                self.optimizer.step()
                if self.scheduler is not None:  # 如果传入了调度器才使用调度
                    self.scheduler.step()

                msg_bar.set_description("Epoch %s" % (self.current_epoch + 1))  # 设置前置描述
                if self.args.use_vali:
                    msg_bar.set_postfix_str('Train loss: {:.6f} | Last val loss: {:.6f}'.format(np.average(losses_train), avg_loss_vali))
                else:
                    msg_bar.set_postfix_str('Train loss: {:.6f}'.format(np.average(losses_train)))
            # avg_loss_train = np.average(losses_train)
            avg_loss_train = np.average(losses_train)

            if (self.current_epoch + 1) % self.args.vali_interval == 0 and self.current_epoch != 0 and self.args.use_vali:
                avg_loss_vali = self.vali()
                self.es.step(self.model, avg_loss_vali)  # 执行保存权重和步进

            # MAE, MAPE, RMSE, test_mse = self.test(target_stats)
            self.epoch_train_loss.append(float(avg_loss_train))
            self.epoch_val_loss.append(float(avg_loss_vali))

            if self.recorder.file_path is None:
                self.recorder.create_record_file()
                self.recorder.record_by_file(self.args.conf_file)  # 配置文件记录
                self.recorder.record(model_summary)  # 记录文件参数
            if self.current_epoch % self.args.vali_interval == 0 and self.args.use_vali:
                record_str = f'epoch: {self.current_epoch + 1}, train_loss: {avg_loss_train}, current valid_loss: {avg_loss_vali}'
            else:
                record_str = f'epoch: {self.current_epoch + 1}, train_loss: {avg_loss_train}'
            self.recorder.record(record_str)
            self.current_epoch += 1

        if not self.args.use_vali:
            torch.save(self.model.state_dict(), f'{self.weight_path}')  # 最终保存一下权重


        # 把训练过程数据存储起来
        epoch_train_loss = np.array(self.epoch_train_loss)
        epoch_val_loss = np.array(self.epoch_val_loss)
        np.savez(f"result/loss/{self.record_name}.npz", epoch_train_loss=epoch_train_loss,
                 epoch_val_loss=epoch_val_loss)


        train_data = {'data': epoch_train_loss, 'label': 'train_loss'}
        val_data = {'data': epoch_val_loss, 'label': 'val_loss'}
        plot_lines([train_data, val_data], title=self.record_name)  # 将标题名设置为记录文件名

    def test(self, weight_path=None):
        if weight_path is None:  # 没用指定加载哪个文件测试，则使用本次训练得到的权重
            weight_path = self.weight_path


        if list(self.model.parameters()):
            self.model.load_state_dict(torch.load(weight_path))  # 加载权重
        if self.recorder.file_path is None:
            self.recorder.create_record_file()

        self.model.eval()

        impute_data_list = []
        X_Tilde_list = []
        eval_mask_list = []

        # generated_samples_list = []

        with torch.no_grad():
            for data in self.test_loader:
                samples, X_Tilde, eval_mask = self.__process_model__(data, self.current_epoch,'test')

                impute_data_list.append(samples.to('cpu')) # 追加到列表中
                X_Tilde_list.append(X_Tilde.to('cpu'))
                eval_mask_list.append(eval_mask.to('cpu'))

            impute_data_list = torch.cat(impute_data_list, dim=0)
            X_Tilde_list = torch.cat(X_Tilde_list, dim=0)
            eval_mask_list = torch.cat(eval_mask_list, dim=0)# 1是插补，0是不插补
            eval_mask_list = 1 - eval_mask_list
            # mae, rmse, mape, mse, R2
            MAE_, RMSE_, MAPE_,  MSE_, R2_ = metric(impute_data_list, X_Tilde_list, eval_mask_list)
        result_str = f'MAE: {MAE_}, MAPE: {MAPE_}, MSE: {MSE_}, RMSE: {RMSE_}, R2: {R2_}'
        print(result_str)
        self.recorder.record(result_str)
        # 保存结果
        impute_data_list = np.array(impute_data_list)
        X_Tilde_list = np.array(X_Tilde_list)
        eval_mask_list = np.array(eval_mask_list)
        np.savez(f"result/output/{self.record_name}.npz", impute_data_list=impute_data_list, X_Tilde_list=X_Tilde_list, eval_mask_list=eval_mask_list)
        return MAE_, MAPE_, RMSE_, MSE_, R2_
