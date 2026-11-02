import numpy as np
import csv
import torch
from utils.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        # data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        # data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        # data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        # data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        # data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    # return data, scaler
    return scaler
def get_adjacent_matrix(distance_file, num_nodes, id_file, graph_type="connect"): # 加载路网数据
    A = np.zeros([int(num_nodes), int(num_nodes)])

    if id_file:
        with open(id_file, "r") as f_id:
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(f_id.read().strip().split("\n"))}

            with open(distance_file, "r") as f_d:
                f_d.readline()
                reader = csv.reader(f_d)
                for item in reader:
                    if len(item) != 3: # 这里避免空行影响
                        continue
                    i, j, distance = int(item[0]), int(item[1]), float(item[2])
                    if graph_type == "connect":
                        A[node_id_dict[i], node_id_dict[j]] = 1.
                        A[node_id_dict[j], node_id_dict[i]] = 1.
                    elif graph_type == "distance":
                        A[node_id_dict[i], node_id_dict[j]] = 1. / distance
                        A[node_id_dict[j], node_id_dict[i]] = 1. / distance
                    else:
                        raise ValueError("graph type is not correct (connect or distance)")
        return A

    with open(distance_file, "r") as f_d:
        f_d.readline()
        reader = csv.reader(f_d)
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])

            if graph_type == "connect":
                A[i, j], A[j, i] = 1., 1.
            elif graph_type == "distance":
                A[i, j] = 1. / distance
                A[j, i] = 1. / distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")
    return A


def multi_gran_data_generator_with_repeat(data, mg_list):
    """
    生成多粒度时间序列数据（带平均值重复扩展）

    :param data: 输入数据，形状为 [batch_size, seq_len] 或 [batch_size, seq_len, features_dim]
    :param mg_list: 多粒度列表，例如 [2, 4] 表示2倍和4倍下采样
    :return: 多粒度数据字典，键为粒度值，值为对应粒度的数据（已重复扩展回原长度）

    示例:
    >>> data = np.array([[1,2,3,4,5,6,7,8]])
    >>> mg_list = [2, 4]
    >>> multi_scale_data = multi_gran_data_generator_with_repeat(data, mg_list)
    >>> print(multi_scale_data[2])  # [[1.5,1.5,3.5,3.5,5.5,5.5,7.5,7.5]]
    >>> print(multi_scale_data[4])  # [[2.5,2.5,2.5,2.5,6.5,6.5,6.5,6.5]]
    """
    # 确保数据至少是2维的
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)

    multi_scale_data = {}

    for gran in mg_list:
        # 检查粒度是否有效
        if gran <= 0 or gran > data.shape[-2]:
            continue

        batch_size = data.shape[0]
        seq_len = data.shape[-2]
        features_dim = data.shape[-1] if len(data.shape) == 3 else 1

        # 初始化结果数组
        coarse_data = np.zeros_like(data, dtype=np.float32)

        for i in range(batch_size):
            # 处理每个样本
            sample = data[i]

            # 计算完整块的数量和剩余长度
            n_blocks = seq_len // gran
            remainder = seq_len % gran

            # 处理完整块部分
            if n_blocks > 0:
                # 重塑为 [n_blocks, gran, features_dim]
                reshaped = sample[:n_blocks * gran].reshape(n_blocks, gran, -1)
                # 计算块平均值 [n_blocks, features_dim]
                block_means = np.mean(reshaped, axis=1)
                # 重复扩展回原长度
                repeated_means = np.repeat(block_means, gran, axis=0)
                coarse_data[i, :n_blocks * gran] = repeated_means

            # 处理剩余部分
            if remainder > 0:
                remainder_part = sample[n_blocks * gran:]
                # 计算剩余部分的平均值
                remainder_mean = np.mean(remainder_part, axis=0, keepdims=True)
                # 重复扩展剩余部分
                repeated_remainder = np.repeat(remainder_mean, remainder, axis=0)
                coarse_data[i, n_blocks * gran:] = repeated_remainder

        # 将结果存入字典
        multi_scale_data[gran] = coarse_data.squeeze()

    # 添加原始数据作为最细粒度
    multi_scale_data[1] = data.squeeze()

    return multi_scale_data


def vectorized_multi_gran_generator(data, mg_list):
    """
    PyTorch版本的向量化多粒度数据生成器

    :param data: 输入数据，形状为 [batch_size, seq_len, features_dim] 或 [batch_size, seq_len]
    :param mg_list: 多粒度列表，例如 [1, 2, 4]
    :return: 多粒度数据字典，键为粒度值，值为对应粒度的数据
    """
    if len(data.shape) == 2:
        data = data.unsqueeze(-1)  # [batch_size, seq_len] -> [batch_size, seq_len, 1]

    batch_size, seq_len, features_dim = data.shape
    multi_scale_data = {}

    # 特殊处理粒度1（原始数据）
    if 1 in mg_list:
        multi_scale_data[1] = data.clone().permute(0, 2, 1)

    # 处理其他粒度
    for gran in sorted(mg_list):
        if gran <= 1 or gran > seq_len:
            continue

        # 计算完整块的数量和剩余长度
        n_blocks = seq_len // gran
        remainder = seq_len % gran

        # 处理完整块部分
        reshaped = data[:, :n_blocks * gran, :].reshape(batch_size, n_blocks, gran, features_dim)
        block_means = torch.mean(reshaped, dim=2, keepdim=True)
        repeated_means = block_means.repeat(1, 1, gran, 1)
        coarse_data = repeated_means.reshape(batch_size, n_blocks * gran, features_dim)

        # 处理剩余部分
        if remainder > 0:
            remainder_part = data[:, n_blocks * gran:, :]
            remainder_mean = torch.mean(remainder_part, dim=1, keepdim=True)
            repeated_remainder = remainder_mean.repeat(1, remainder, 1)
            coarse_data = torch.cat([coarse_data, repeated_remainder], dim=1)
        coarse_data = coarse_data.permute(0, 2, 1) # 转换维度，将Features转到第二维度
        multi_scale_data[gran] = coarse_data.squeeze(-1) if features_dim == 1 else coarse_data

    return multi_scale_data

if __name__ == '__main__':
    import numpy as np

    # 生成示例数据：32个样本，128时间步，10个特征
    data = np.random.rand(1, 12, 207)
    data = torch.tensor(data)
    mg_list = [1, 2, 4, 8, 16]

    # 生成多粒度数据
    multi_scale_data = vectorized_multi_gran_generator(data, mg_list)

    # 查看结果
    print("原始数据形状:", multi_scale_data[1].shape)  # (32, 128, 10)


    print("2倍下采样形状:", multi_scale_data[2].shape)  # (32, 64, 10)
    print(multi_scale_data[2][:,1])
    print("4倍下采样形状:", multi_scale_data[4].shape)  # (32, 32, 10)
    print(multi_scale_data[4][:, 1])
    # print("8倍下采样形状:", multi_scale_data[8].shape)  # (32, 16, 10)
    # print("16倍下采样形状:", multi_scale_data[16].shape)  # (32, 8, 10)