import yaml
from utils.data_utils import get_adjacent_matrix
import torch
import os
def get_conf_obj(file_path):
    with open(file_path, "r") as f:
        conf = yaml.safe_load(f) # 加载出配置文件
        args = type('ConfClass', (), conf)()
        # 处理矩阵A
        if hasattr(args, "matrix_file"): # 如果对象中有矩阵的地址，则加载
            id_file = None
            if hasattr(args, "id_file"):
                id_file = args.id_file
            A = get_adjacent_matrix(args.matrix_file, args.node_num, id_file=id_file)
            A = torch.Tensor(A).cuda()
            setattr(args, "A", A) # 属性添加
        setattr(args, "conf_file", file_path) # 把配置文件也记录到args中
        return args

def get_filepath_by_dir(dir_path):
    conf_file_list = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            f = str(f)
            fp = os.path.join(root, f)
            conf_file_list.append(fp)
    return conf_file_list

def read_yaml(file_path):
    """
    根据文件名加载对应的yaml文件
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def write_yaml(file_path, obj_data):
    """
    传入一个数据，将其转换为yaml字符串并写入yaml文件
    """
    with open(file=file_path, mode="w") as f:
        yaml.dump(obj_data, f)


def add_property(file_path, key, value):
    data = read_yaml(file_path) # 读取文件内容
    if key not in data:
        data[key] = value
    write_yaml(file_path, data) # 重新写入文件

def delete_property(file_path, key):
    data = read_yaml(file_path)  # 读取文件内容
    if key in data:
        _ = data.pop(key) # 将键值对删除
    write_yaml(file_path, data)


def clear_yaml(file_path):
    with open(file_path, "w") as file:
        pass

if __name__ == '__main__':
    conf_file_list = get_filepath_by_dir('../configuration')
    for conf_path in conf_file_list:
        add_property(conf_path, 'use_vali', False)

