import numpy as np


class SimpleLinearRegression(object):
    def __init__(self):
        self.theta = None

    def normal_equation(self, train_data):
        """
        正规方程求解参数
        :param train_data:
        :return:
        """
        # 获取特征
        x = train_data[:, 0:-1]
        # 添加一列常数项
        x = np.c_[np.ones(x.shape[0]), x]
        # 样本标签
        y = train_data[:, -1]
        # 特征缩放
        x = (x - x.mean()) / x.std()
        # 正规方程求解 公式：(x^T * x)^-1 * x^T * y
        pseudo_inverse = np.dot(np.linalg.inv(np.dot(x.T, x)), x.T)
        self.theta = np.dot(pseudo_inverse, y)


if __name__ == '__main__':
    data = np.loadtxt('ex2.txt', delimiter=',', dtype=float)
    linear = SimpleLinearRegression()
    # 训练
    linear.normal_equation(data)
    # 参数
    print(linear.theta)
