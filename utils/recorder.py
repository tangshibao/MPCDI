import time

from datetime import datetime
class Recorder(object):
    def __init__(self, file_dir=None, record_name=None):
        self.record_name = record_name
        self.file_dir = file_dir
        self.file_path = None
        # self.create_record_file()


    def create_record_file(self): # 创建记录文件
        if self.file_dir is None:
            self.file_path = f'{self.record_name}.txt' # 记录文件的位置
        else:
            self.file_path = f'{self.file_dir}/{self.record_name}.txt'

    def record(self, record_str):
        with open(self.file_path, "a", encoding='utf-8') as file:
            file.write(f'{record_str} \n')

    def record_by_file(self, source_file): # 根据源文件的数据，转移记录到记录文件中
        with open(source_file, "r") as f:  # 打开文件
            file_content = f.read()  # 读取文件

        self.record(file_content)
