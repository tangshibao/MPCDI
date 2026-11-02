import torch
import numpy as np
def metric(pred, true, mask):
    valid_elements = (1 - mask).sum()
    # MAE: 平均绝对误差
    mae = torch.sum(torch.abs(pred - true) * (1 - mask)) / (valid_elements + 1e-5)

    # MSE: 均方误差
    mse = torch.sum(torch.square(pred - true) * (1 - mask)) / (valid_elements + 1e-5)

    # RMSE: 均方根误差
    rmse = torch.sqrt(mse)  # 更高效的实现方式：直接对mse开方

    # MAPE: 平均绝对百分比误差
    mape = torch.sum(torch.abs((pred - true) * (1 - mask))) / (torch.sum(torch.abs(true * (1 - mask))) + 1e-5)

    # R² 决定系数
    # 计算真实值的均值（仅考虑有效位置）
    true_mean = torch.sum(true * (1 - mask)) / (valid_elements + 1e-5)
    # 计算总平方和（SST）
    sst = torch.sum(torch.square(true - true_mean) * (1 - mask))
    # 计算决定系数
    R2 = 1 - mse / (sst / (valid_elements + 1e-5))

    return mae, rmse, mape, mse, R2

def missed_eval_np(predict, true, mask):
    """
    predict: [samples, seq_len, feature_dim]
    true: [samples, seq_len, feature_dim]
    mask: [samples, seq_len, feature_dim]
    """
    predict, true = np.asarray(predict), np.asarray(true)
    mae = np.sum(np.absolute(predict - true) * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    mse = np.sum((predict - true) ** 2 * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    rmse = np.sqrt(np.sum((predict - true) ** 2 * (1 - mask)) / (np.sum(1 - mask) + 1e-5))
    mape = np.sum(np.absolute((predict - true) * (1 - mask))) / (np.sum(np.absolute(true * (1 - mask))) + 1e-5)
    R2 = 1 - mse / (np.sum((true - np.mean(true)) ** 2 * (1 - mask)) / (np.sum(1 - mask) + 1e-5))
    return mae, rmse, mape, mse, R2



