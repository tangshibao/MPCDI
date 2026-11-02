# 绘图器，负责所有的图像绘制
import matplotlib.pyplot as plt
def plot_lines(line_datas, xlabel=None, title='loss_line', xlabel_name='epoch', ylabel_name='loss', save_path=None):
    """
    Args:
        line_datas: 元组，每一个元素是一个折线的字典

    Returns:

    """
    for i in range(len(line_datas)):
        if xlabel is None:
            plt.plot(line_datas[i]['data'], label=line_datas[i]['label'])
        else: # 如果写的有xlabel
            plt.plot(xlabel, line_datas[i]['data'], label=line_datas[i]['label'])
    plt.title(title)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    plt.legend()
    if save_path is None:
        plt.savefig(f'./result/img/{title}.png')
    # plt.show()


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5]
    data1 = [2, 4, 6, 8, 10]
    data1 = {'data': data1, 'label': 'y1'}
    data2 = [1, 2, 1, 2, 1]
    data2 = {'data': data2, 'label': 'y2'}
    line_datas = [data1, data2]
    plot_lines(line_datas, x)