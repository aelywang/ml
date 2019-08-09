import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SimpleLinearRegression(object):

    def __init__(self, theta, alpha, iterations):
        """
        目标函数：hΘ(x) = Θ0 + Θ1x1 + Θ2x2 + Θ3x3 + ... + Θnxn
        损失函数：j(Θ0,Θ1,...,Θn) = 1/(2m)*sum((hΘ(x) - y)^2)
        梯度下降：
                Θ0 := Θ0 - α * (1/m*sum(hΘ(x) - y))
                Θ1 := Θ1 - α * (1/m*sum(hΘ(x) - y)*x1)
                Θ2 := Θ2 - α * (1/m*sum(hΘ(x) - y)*x2)
                ...
        :param theta:参数
        :param alpha:学习率
        :param iterations:迭代步长
        """
        self.theta = theta
        self.alpha = alpha
        self.iterations = iterations
        # 每次迭代的损失值
        self.each_iteration_val = np.zeros((self.iterations, 1))

    @staticmethod
    def read_data(path):
        """
        读取csv数据
        :param path:路径
        :return: 数据
        """
        # 读取数据，逗号分隔
        csv_data = pd.read_csv(path, delimiter=',')
        # 数据特征缩放
        csv_data = (csv_data - csv_data.mean()) / csv_data.std()
        # 补充第一列常数项
        csv_data.insert(0, 'First', 1)
        return csv_data

    def cost_function(self, x, y, m):
        """
        损失函数：j(Θ0,Θ1) = 1/(2m)*sum((hΘ(x) - y)^2)
        :param x: x特征向量
        :param y: y结果向量
        :param m: 数量
        :return:
        """
        # 计算出损失总和 1/(2m)*sum((hΘ(x) - y)^2)
        loss = np.power(((x * self.theta.T) - y.T), 2)
        # 损失值
        cost = np.sum(loss) / (2 * m)
        return cost

    def gradient_descent(self, train_data):
        """
        梯度下降
        :param train_data:
        :return:
        """
        # 提取出前三列特征转换成矩阵，也就是二维ndarray，方便直接*矩阵乘法
        x = np.asmatrix(train_data.iloc[:, 0: -1].values)
        y = np.asmatrix(train_data.iloc[:, -1].values)
        # 行数
        m = x.shape[0]
        # 列数
        col = x.shape[1]
        # 初始化theta中间量
        temp = np.asmatrix(np.zeros(self.theta.shape))
        # 开始迭代
        for i in range(self.iterations):
            loss = (x * self.theta.T) - y.T
            # 更新每一个参数
            for j in range(col):
                # Θ0 := Θ0 - α * (1 / m * sum(hΘ(x) - y))
                # Θ1 := Θ1 - α * (1 / m * sum(hΘ(x) - y) * x1)
                # Θ2 := Θ2 - α * (1 / m * sum(hΘ(x) - y) * x2)
                term = np.multiply(loss, x[:, j])
                temp[0, j] = self.theta[0, j] - ((self.alpha / m) * np.sum(term))
            # 更新theta参数
            self.theta = temp
            # 记录每次损失值
            cost = self.cost_function(x, y, m)
            self.each_iteration_val[i, 0] = cost
            print(f'第{i}次损失值：{cost}，theta：{self.theta}')


if __name__ == '__main__':
    # 初始化theta，alpha学习率，迭代步长
    linear = SimpleLinearRegression(np.asmatrix(np.array([0, 0, 0])), 0.03, 1500)
    print(linear.theta)
    data = linear.read_data('./ex2.txt')
    # 执行梯度下降过程
    linear.gradient_descent(data)
    print(f'theta：{linear.theta}')
    # 绘图
    plt.plot(np.arange(linear.iterations), linear.each_iteration_val)
    plt.title('CostFunction')
    plt.show()
