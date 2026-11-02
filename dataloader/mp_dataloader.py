import numpy as np
import pickle as pk

import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.mask_utils import sample_mask
from utils.data_utils import normalize_dataset

def get_dataloader(file_path,
                batch_size,
                train_ratio,
                val_ratio,
                test_ratio,
                seq_len, # 窗口长度
                train_missing_ratio,
                val_missing_ratio,
                test_missing_ratio,
                missing_pattern="point",
                interval=12,
                normalizer='std',
                column_wise=False):
    """
    :param file_path:
    :param batch_size:
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    :param seq_len: 窗口大小
    :param missing_ratio: 缺失比例
    :param missing_pattern: 人工缺失的模式
    :param interval: 重叠窗口采样，interval<seq_len, 非重叠窗口采样，interval = seq_len
    :return:
    """

    # 加载数据集，按照比例划分
    data_seq = np.load(file_path)['data']  # wd: (16992, 307, 3)
    data_seq = np.float32(data_seq)[:, :, 0] # 只加载交通流列

    # 按照比例划分
    # data_length = len(data_seq) # 得到数据条目数
    data_length = len(data_seq)
    split_line1 = int(data_length * train_ratio)  # 训练集0.6
    split_line2 = int(data_length * (train_ratio + val_ratio))  # 测试集和验证集各0.2
    split_line3 = int(data_length * (train_ratio + val_ratio + test_ratio))

    train_data, val_data, test_data = data_seq[:split_line1], data_seq[split_line1: split_line2], data_seq[split_line2: split_line3]

    scaler = normalize_dataset(train_data, normalizer, column_wise)

    # 这一段标准化正常吗
    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    # 使用滑窗来
    train_loader = generate_dataloader(train_data, seq_len, train_missing_ratio, missing_pattern, batch_size,
                                       step_interval=interval, shuffle=True, data_seed=9101110)
    val_loader = generate_dataloader(val_data, seq_len, val_missing_ratio, missing_pattern, batch_size,
                                       step_interval=interval, shuffle=False, data_seed=9101111)
    test_loader = generate_dataloader(test_data, seq_len, test_missing_ratio, missing_pattern, batch_size,
                                       step_interval=interval, shuffle=False, data_seed=9101112)

    return train_loader, val_loader, test_loader, scaler


def generate_dataloader(data, seq_len, missing_ratio, missing_pattern="point", batch_size=4, step_interval=1, shuffle=False, data_seed=0):
    rng = np.random.default_rng(data_seed)
    X_Tilde = data
    gt_mask = (~np.isnan(X_Tilde)).astype(np.float32)
    if missing_pattern == "block":
        # block missing
        indicating_mask = sample_mask(
            shape=data.shape,
            p=0.0015, # 设置为每一步1.5%概率
            p_noise=missing_ratio,
            min_seq=int(seq_len / 2),
            max_seq=int(seq_len * 2),
            rng=rng,  # 不设置种子
        )
    else:
        # point missing
        indicating_mask = sample_mask(
            shape=data.shape,
            p=0.0, # 直接设置为0
            p_noise=missing_ratio,
            min_seq=int(seq_len / 2),
            max_seq=int(seq_len * 2),
            rng=rng,
        )
    X = X_Tilde * (1 - indicating_mask)

    mask = gt_mask * (1 - indicating_mask)

    X = np.nan_to_num(X)
    X_Tilde = np.nan_to_num(X_Tilde)


    (input_X_list,
     input_mask_list,
     eval_mask,
     output_gt_list,
     output_gt_mask,
     hour_X_list,
     hour_mask_list,
     day_X_list,
     day_mask_list,
     week_X_list,
     week_mask_list,


     ) = [], [], [], [], [], [], [], [], [], [], []

    # 如果是Train的话，返回7个，否则返回5个
    idx = 0
    while idx < X.shape[0] - seq_len: # 减去一个seq_len，以留存一个
        indices = get_indices(X.shape[0], idx, seq_len)

        if not indices:
            idx = idx + step_interval # 直接加step_interval
            continue

        week_indices, day_indices, hour_indices = indices
        target_indices = [idx, idx + seq_len]

        # 通用方法
        input_X_list.append(X[target_indices[0]: target_indices[1]])
        input_mask_list.append(mask[target_indices[0]: target_indices[1]])
        eval_mask.append(indicating_mask[target_indices[0]: target_indices[1]])  # 手动遮蔽的掩码
        output_gt_list.append(X_Tilde[target_indices[0]: target_indices[1]])
        output_gt_mask.append(gt_mask[target_indices[0]: target_indices[1]])

        hour_X_list.append(X[hour_indices[0]: hour_indices[1]])
        hour_mask_list.append(mask[hour_indices[0]: hour_indices[1]])

        day_X_list.append(X[day_indices[0]: day_indices[1]])
        day_mask_list.append(mask[day_indices[0]: day_indices[1]])

        week_X_list.append(X[week_indices[0]: week_indices[1]])
        week_mask_list.append(mask[week_indices[0]: week_indices[1]])
        # 否则的话，在后面+step_interval
        idx = idx + step_interval


    X_tensor = torch.from_numpy(np.array(input_X_list)).float()
    mask_tensor = torch.from_numpy(np.array(input_mask_list)).float()
    eval_mask_tensor = torch.from_numpy(np.array(eval_mask)).float()
    X_Tilde_tensor = torch.from_numpy(np.array(output_gt_list)).float()
    output_gt_mask_tensor = torch.from_numpy(np.array(output_gt_mask)).float()

    hour_X_tensor = torch.from_numpy(np.array(hour_X_list)).float()  # 掩码
    hour_mask_tensor = torch.from_numpy(np.array(hour_mask_list)).float()

    day_X_tensor = torch.from_numpy(np.array(day_X_list)).float() # 掩码
    day_mask_tensor = torch.from_numpy(np.array(day_mask_list)).float()



    week_X_tensor = torch.from_numpy(np.array(week_X_list)).float()
    week_mask_tensor = torch.from_numpy(np.array(week_mask_list)).float()

    tensor_dataset = TensorDataset(X_tensor,
                                   mask_tensor,
                                   X_Tilde_tensor,
                                   output_gt_mask_tensor,
                                   eval_mask_tensor,
                                   hour_X_tensor,
                                   hour_mask_tensor,
                                   day_X_tensor,
                                   day_mask_tensor,
                                   week_X_tensor,
                                   week_mask_tensor
                                   )

    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True) # 如果是训练集就打乱，否则不打乱
    return dataloader



def search_data(sequence_length, label_start_idx, window_len, period_len):
    if period_len < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx > sequence_length:
        return None

    start_idx = label_start_idx - period_len

    end_idx = start_idx + window_len # end_idx 因该是 start_idx +
    if start_idx >= 0:
        return (start_idx, end_idx)
    else:
        return None

def get_indices(data_length, label_start_idx, seq_len):
    """
    :param data_length:
    :param num_of_weeks: 多少个周期
    :param num_of_days:
    :param num_of_hours:
    :param label_start_idx:
    :param num_for_predict:
    :param points_per_hour:
    :return:
    """
    week_indices = search_data(data_length, label_start_idx, seq_len, 7 * 24 * 12) # 一周前
    if not week_indices:
        return None

    day_indices = search_data(data_length, label_start_idx, seq_len, 24 * 12) # 一天前
    if not day_indices:
        return None

    hour_indices = search_data(data_length, label_start_idx, seq_len, seq_len) # 一小时前

    if not hour_indices:
        return None
    # week_sample = np.concatenate([data_sequence[i: j] for i, j in week_indices], axis=0)
    # day_sample = np.concatenate([data_sequence[i: j] for i, j in day_indices], axis=0)
    # hour_sample = np.concatenate([data_sequence[i: j] for i, j in hour_indices], axis=0)
    # target = data_sequence[label_start_idx: label_start_idx + num_for_predict]
    return week_indices, day_indices, hour_indices








