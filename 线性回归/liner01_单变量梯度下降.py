import numpy as np
import matplotlib.pyplot as plt


class SimpleLinearRegression(object):

    def __init__(self, theta_0, theta_1, alpha, iterations):
        """
        目标函数：hΘ(x) = Θ0 + Θ1x
        损失函数：j(Θ0,Θ1) = 1/(2m)*sum((hΘ(x) - y)^2)
        梯度下降：
                Θ0 := Θ0 - α * (1/m*sum(hΘ(x) - y))
                Θ1 := Θ1 - α * (1/m*sum(hΘ(x) - y)*x1)
        :param theta_0:参数0
        :param theta_1: 参数1
        :param alpha:学习率
        :param iterations:迭代步长
        """
        self.theta_0 = theta_0
        self.theta_1 = theta_1
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
        # 读取数据，逗号分隔，float类型
        csv_data = np.loadtxt(path, dtype=float, delimiter=',')
        return csv_data

    def cost_function(self, x, y, m):
        """
        损失函数：j(Θ0,Θ1) = 1/(2m)*sum((hΘ(x) - y)^2)
        :param x: x特征向量
        :param y: y结果向量
        :param m: 样本数量
        :return:
        """
        # 假设函数计算出y值
        predict_y_val = self.theta_0 + self.theta_1 * x
        # 损失值
        cost = sum((predict_y_val - y) ** 2) / (2 * m)
        return cost

    def gradient_descent(self, train_data):
        """
        梯度下降
        :param train_data:
        :return:
        """
        x = train_data[:, 0]
        y = train_data[:, 1]
        m = train_data.shape[0]
        for i in range(0, self.iterations):
            # 假设函数
            hypothesis = self.theta_0 + self.theta_1 * x
            # 更新 Θ0 := Θ0 - α * (1/m*sum(hΘ(x) - y))
            temp_0 = self.theta_0 - self.alpha * ((1 / m) * sum(hypothesis - y))
            # 更新 Θ1 := Θ1 - α * (1/m*sum(hΘ(x) - y)*x1)
            temp_1 = self.theta_1 - self.alpha * (1 / m) * sum((hypothesis - y) * x)
            self.theta_0 = temp_0
            self.theta_1 = temp_1
            # 计算本次迭代损失函数
            cost = self.cost_function(x, y, m)
            # 保存便于画图显示
            self.each_iteration_val[i, 0] = cost
            print(f'第{i}次损失值：{cost}')


if __name__ == '__main__':
    # 初始化theta_0，theta_1，alpha学习率，迭代步长
    linear = SimpleLinearRegression(0, 0, 0.003, 5000)
    data = linear.read_data('./ex1.txt')
    # 执行梯度下降过程
    linear.gradient_descent(data)
    print(f'theta_0：{linear.theta_0}')
    print(f'theta_1：{linear.theta_1}')
    # 绘图
    plt.subplot(211)
    plt.scatter(data[:, 0], data[:, 1], color='g', s=20)
    plt.plot(data[:, 0], linear.theta_0 + linear.theta_1 * data[:, 0])
    plt.title('拟合曲线')
    plt.subplot(212)
    plt.plot(np.arange(linear.iterations), linear.each_iteration_val)
    plt.title('损失函数')
    plt.show()
