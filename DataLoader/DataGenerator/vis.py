# 可视化轨迹数据
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

def main():
    parser = argparse.ArgumentParser(description='可视化轨迹文件 (CSV)')
    parser.add_argument('file', nargs='?', default='output/trajectory.txt', help='轨迹文件路径')
    args = parser.parse_args()
    # 读取数据
    # 读取 CSV 并去除列名空白
    df = pd.read_csv(args.file, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    # 创建3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 主目标轨迹
    ax.plot(df['x'], df['y'], df['z'], label='car', color='blue')
    # 子目标轨迹，可兼容 0、3、4 子目标
    # 寻找所有以 sx 开头的列，并提取索引
    sub_idxs = sorted(int(col[2:]) for col in df.columns if col.startswith('sx'))
    for idx in sub_idxs:
        x_col = f'sx{idx}'
        y_col = f'sy{idx}'
        z_col = f'sz{idx}'
        ax.plot(df[x_col], df[y_col], df[z_col], label=f'armor{idx}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('trace')
    ax.legend()
    # 添加滑块以查看单帧位置
    # 主目标初始散点
    main_scatter = ax.scatter([df['x'][0]], [df['y'][0]], [df['z'][0]], color='red', s=50)
    # 子目标初始散点列表
    sub_scatters = []
    for idx in sub_idxs:
        sc = ax.scatter([], [], [], color='green', s=30)
        sub_scatters.append(sc)
    # 滑块布局
    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    frame_slider = Slider(slider_ax, 'Frame', 0, len(df)-1, valinit=0, valfmt='%d')
    # 更新函数
    def update(val):
        i = int(val)
        # 更新主目标位置
        main_scatter._offsets3d = ([df['x'][i]], [df['y'][i]], [df['z'][i]])
        # 更新子目标位置
        for sc, idx in zip(sub_scatters, sub_idxs):
            x_i = df[f'sx{idx}'][i]; y_i = df[f'sy{idx}'][i]; z_i = df[f'sz{idx}'][i]
            sc._offsets3d = ([x_i], [y_i], [z_i])
        fig.canvas.draw_idle()
    frame_slider.on_changed(update)
    plt.show()

if __name__ == '__main__':
    main()
