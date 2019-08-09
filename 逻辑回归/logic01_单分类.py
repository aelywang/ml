import numpy as np


class LogicRegression(object):
    def __init__(self, theta, alpha):
        self.theta = theta
        self.alpha = alpha

    @staticmethod
    def sigmoid(z):
        """
        sigmoid函数
        :param z:
        :return:
        """
        return 1 / (1 + np.exp(-z))

    def cost_function(self, x, y, m):
        # 假设函数
        first = -y * np.log(self.sigmoid(np.dot(x,self.theta)))


        return np.mean(-y * np.log(self.sigmoid(np.dot(x, self.theta.T))))
