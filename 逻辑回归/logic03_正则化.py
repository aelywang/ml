import numpy as np


class LogisticRegression:
    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.beta = None
        self.gamma = gamma
        self.penalty = penalty
        self.fit_intercept = fit_intercept

    @staticmethod
    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        l_prev = np.inf
        self.beta = np.random.rand(X.shape[1])
        for _ in range(int(max_iter)):
            y_pred = self.sigmoid(np.dot(X, self.beta))
            loss = self._NLL(X, y, y_pred)
            if l_prev - loss < tol:
                return
            l_prev = loss
            self.beta -= lr * self._NLL_grad(X, y, y_pred)

    def _NLL(self, X, y, y_pred):
        """
        Penalized negative log likelihood of the targets under the current
        model.
            NLL = -1/N * (
                [sum_{i=0}^N y_i log(y_pred_i) + (1-y_i) log(1-y_pred_i)] -
                (gamma ||b||) / 2
            )
        """
        N, M = X.shape
        order = 2 if self.penalty == "l2" else 1
        nll = -np.log(y_pred[y == 1]).sum() - np.log(1 - y_pred[y == 0]).sum()
        penalty = 0.5 * self.gamma * np.linalg.norm(self.beta, ord=order) ** 2
        return (penalty + nll) / N

    def _NLL_grad(self, X, y, y_pred):
        """ Gradient of the penalized negative log likelihood wrt beta """
        N, M = X.shape
        p = self.penalty
        beta = self.beta
        gamma = self.gamma
        l1norm = lambda x: np.linalg.norm(x, 1)
        d_penalty = gamma * beta if p == "l2" else gamma * l1norm(beta) * np.sign(beta)
        return -(np.dot(y - y_pred, X) + d_penalty) / N

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return self.sigmoid(np.dot(X, self.beta))


# def gradientReg(theta, X, y, learningRate):
#
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)    
#     parameters = int(theta.ravel().shape[1])
#     grad = np.zeros(parameters)    
#     error = sigmoid(X * theta.T) - y    
#     for i in range(parameters):
#         term = np.multiply(error, X[:, i])     
#         if (i == 0):
#                 grad[i] = np.sum(term) / len(X)
#         else:
#             grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])    
#     return grad
