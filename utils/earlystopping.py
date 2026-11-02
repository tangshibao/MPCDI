import torch
import sys
class EarlyStopping(object):
    def __init__(self, weight_path,mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = sys.float_info.max
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        self.weight_path = weight_path
        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, model, metrics):
        if self.best is None:
            self.best = metrics
            torch.save(model.state_dict(), f'{self.weight_path}')
            return False

            # if torch.isnan(metrics):
        #     return True

        if self.is_better(metrics, self.best):  # 如果取得了最佳成绩，则将倒数置零，并更新最佳
            self.num_bad_epochs = 0
            self.best = metrics
            # 保存最佳权重
            torch.save(model.state_dict(), f'{self.weight_path}')

        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)
