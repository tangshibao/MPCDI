import numpy as np
def sample_mask(
    shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None
):  # block missing, point missing。
    """
    :param shape:
    :param p: p用来控制段缺失数据
    :param p_noise: 用来控制点缺失
    :param max_seq:
    :param min_seq:
    :param rng:
    :return:
    因为点缺失是一种长度为1的段缺失，所以，在段缺失情况下，点缺失依然存在，但是点缺失情况下，段缺失不存在，即（p为0）
    """
    # 设置种子
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers

    # 通过rand(shape) < p 利用概率生成一个真假掩码矩阵
    mask = rand(shape) < p

    for col in range(mask.shape[1]): # 遍历每一列
        idxs = np.flatnonzero(mask[:, col]) # 当前列的非零元素索引
        if not len(idxs): # 如果没有非零列，即：该列全是0.则直接放弃此列
            continue
        fault_len = min_seq # 段缺失最小值，段（bolock）最小缺失1
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq)) # 随机一个段长度，从min-max之间
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs]) # 生成每一个段的掩码位置
        idxs = np.unique(idxs_ext) # 去除重复值
        idxs = np.clip(idxs, 0, shape[0] - 1) # 限制范围，这两个步骤其实没关系
        mask[idxs, col] = True # 标记缺失位置（注意，每一列的段缺失长度可能是不一样的）
    mask = mask | (rand(mask.shape) < p_noise) # 加入噪声 # TODO: 这里加入噪声有什么用？ 这里是用来模拟点缺失的
    return mask.astype("uint8")