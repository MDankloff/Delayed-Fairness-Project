import numpy as np
import cvxpy as cp
from utils import sigmoid
from sklearn.linear_model import LogisticRegression



class LR:

    name = 'Logistic Regression'

    def __init__(self, l2_reg):
        self.model = LogisticRegression(C=1.0/l2_reg, max_iter=1000, random_state=2021)

    def train(self, s, X, y):
        Xs = np.c_[s, X]
        self.model.fit(Xs, y)

    def predict(self, s, X):
        Xs = np.c_[s, X]
        p = self.model.predict_proba(Xs)[:, 1]
        y = self.model.predict(Xs)
        return y, p

    @property
    def params(self):
        return np.r_[self.model.coef_[0], self.model.intercept_]

class CvxFairModel:
    name = 'Fair Model with Demographic Parity'
    def __init__(self, n_features, l2_reg, tao):
        self.l2_reg = l2_reg
        self.tao = tao
        self.w = cp.Variable(n_features)  # weights for [s, X]
        self.b = cp.Variable()            # intercept

    def add_features(self, s, X):
        Z = np.c_[s.astype(float), X.astype(float)]
        return Z

    def compute_loss(self, s, X, y):
        Z = self.add_features(s, X)
        n = Z.shape[0]
        h = Z @ self.w + self.b
        t1 = (1/n) * cp.sum(-cp.multiply(y, h) + cp.logistic(h))
        t2 = self.l2_reg * cp.sum_squares(self.w)
        return t1 + t2

    def compute_constraint(self, s, X):
        Z = self.add_features(s, X)
        h = Z @ self.w + self.b
        X_pos = Z[s == 1]; h_pos = X_pos @ self.w + self.b
        X_neg = Z[s == 0]; h_neg = X_neg @ self.w + self.b
        n_pos = max(1, len(X_pos)); n_neg = max(1, len(X_neg))
        c1 = (1/n_pos)*cp.sum(cp.logistic(h_pos)) + (1/n_neg)*cp.sum(cp.logistic(-h_neg))
        c2 = (1/n_pos)*cp.sum(cp.logistic(-h_pos)) + (1/n_neg)*cp.sum(cp.logistic(h_neg))
        return c1, c2

    def train(self, s, X, y):
        loss = self.compute_loss(s, X, y)
        c1, c2 = self.compute_constraint(s, X)
        prob = cp.Problem(cp.Minimize(loss), [c1 <= self.tao, c2 <= self.tao])
        installed = set(cp.installed_solvers())
        solver_attempts = [
            ("ECOS", {"max_iters": 10_000}),
            ("SCS", {"max_iters": 50_000, "eps": 1e-5}),
        ]

        last_err = None
        for solver_name, opts in solver_attempts:
            if solver_name not in installed:
                continue
            try:
                prob.solve(solver=getattr(cp, solver_name), warm_start=True, **opts)
                if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    last_err = None
                    break
            except cp.SolverError as e:
                last_err = e

        if last_err is not None:
            raise last_err

    def predict(self, s, X):
        Z = self.add_features(s, X)
        h = Z @ self.w.value + self.b.value
        yhat = (h >= 0).astype(float)
        p = 1/(1+np.exp(-h))
        return yhat, p

    @property
    def params(self):
        """Return parameters in the common convention used by generator.Agent.

        Format: [w_s, w_x1, ..., w_xd, b]
        """
        if self.w.value is None or self.b.value is None:
            return None
        return np.r_[self.w.value, float(self.b.value)]


# class CvxFairModel:

#     name = 'Fair Model with Demographic Parity'

#     def __init__(self, n_features, l2_reg, tao):
#         self.l2_reg = l2_reg
#         self.tao = tao
#         self.w = cp.Variable(n_features)

#     def add_intercept(self, s, X):
#         return np.c_[s, X, np.ones_like(s)]

#     def compute_loss(self, s, X, y):
#         X = self.add_intercept(s, X)
#         n = X.shape[0]

#         # compute log likelihood
#         t1 = 1.0/n * cp.sum(-1.0 * cp.multiply(y, X @ self.w) + cp.logistic(X @ self.w))
#         # add l2_reg
#         t2 = self.l2_reg * cp.norm(self.w[:-1]) ** 2
#         return t1 + t2

#     def compute_constraint(self, s, X):
#         X = self.add_intercept(s, X)
#         n = X.shape[0]
        
#         X_pos = X[s == 1]
#         X_neg = X[s == 0]

#         h_pos = X_pos @ self.w
#         h_neg = X_neg @ self.w
#         c1 = 1.0 / len(X_pos) * cp.sum(cp.logistic(h_pos)) + 1.0 / len(X_neg) * cp.sum(cp.logistic(-h_neg))
#         c2 = 1.0 / len(X_pos) * cp.sum(cp.logistic(-h_pos)) + 1.0 / len(X_neg) * cp.sum(cp.logistic(h_neg))
#         return c1, c2

#     def train(self, s, X, y):
#         loss = self.compute_loss(s, X, y)
#         c1, c2 = self.compute_constraint(s, X)
#         cons = [c2 <= self.tao, c1 <= self.tao]
#         obj = cp.Minimize(loss)
#         prob = cp.Problem(obj, cons)
#         prob.solve()
#         print(prob.status)

#     def predict(self, s, X):
#         X = self.add_intercept(s, X)
#         h = X @ self.w.value
#         pred_y = (h >= 0).astype(float)
#         p = sigmoid(h) 
#         return pred_y, p 

#     @property
#     def params(self):
#         return self.w.value


class EOFairModel:

    name = 'Fair Model with Equal Oppertunity'

    def __init__(self, n_features, l2_reg, tao):
        self.l2_reg = l2_reg
        self.tao = tao
        self.w = cp.Variable(n_features)

    def add_intercept(self, s, X):
        return np.c_[s, X, np.ones_like(s)]

    def compute_loss(self, s, X, y):
        X = self.add_intercept(s, X)
        n = X.shape[0]

        # compute log likelihood
        t1 = 1.0/n * cp.sum(-1.0 * cp.multiply(y, X @ self.w) + cp.logistic(X @ self.w))
        # add l2_reg
        t2 = self.l2_reg * cp.norm(self.w[:-1]) ** 2
        return t1 + t2

    def compute_constraint(self, s, X, y):
        X = self.add_intercept(s, X)
        n = X.shape[0]
        
        X_pos = X[(y == 1) == (s == 1)]
        X_neg = X[(y == 1) == (s == 0)]

        h_pos = X_pos @ self.w
        h_neg = X_neg @ self.w
        c1 = 1.0 / len(X_pos) * cp.sum(cp.logistic(h_pos)) + 1.0 / len(X_neg) * cp.sum(cp.logistic(-h_neg))
        c2 = 1.0 / len(X_pos) * cp.sum(cp.logistic(-h_pos)) + 1.0 / len(X_neg) * cp.sum(cp.logistic(h_neg))
        return c1, c2

    def train(self, s, X, y):
        loss = self.compute_loss(s, X, y)
        c1, c2 = self.compute_constraint(s, X, y)
        cons = [c2 <= self.tao, c1 <= self.tao]
        obj = cp.Minimize(loss)
        prob = cp.Problem(obj, cons)
        prob.solve()
        print(prob.status)

    def predict(self, s, X):
        X = self.add_intercept(s, X)
        h = X @ self.w.value
        pred_y = (h >= 0).astype(float)
        p = sigmoid(h) 
        return pred_y, p 

    @property
    def params(self):
        return self.w.value