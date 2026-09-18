import numpy as np
import cvxpy as cp
from utils import sigmoid
from sklearn.linear_model import LogisticRegression


def _train_fair_model(model, s, X, y, *, equal_opportunity):
    """Solve the original objective with one shared softplus epigraph per row.

    softplus(-h) = softplus(h) - h. Since the objective increases strictly
    in t, t >= softplus(h) is tight at the optimum. Reusing t avoids separate
    exponential-cone expansions of the loss and both fairness constraints.
    """
    model.w.value = None
    if not equal_opportunity:
        model.b.value = None
    model.training_diagnostics = []
    model.status_ = None
    s, X, y = np.asarray(s), np.asarray(X), np.asarray(y)
    if (s.ndim != 1 or y.ndim != 1 or X.ndim != 2 or len(y) == 0
            or len(s) != len(y) or len(X) != len(y)):
        raise ValueError("Expected aligned nonempty s (n,), X (n, d), y (n,).")
    if not np.isfinite(X).all() or not np.isin(s, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Features must be finite; s and y must be binary.")
    qualified = (y == 1) if equal_opportunity else np.ones(len(y), dtype=bool)
    pos, neg = (s == 1) & qualified, (s == 0) & qualified
    if not pos.any() or not neg.any():
        raise ValueError("Fairness requires observations in both groups (with y=1 for EO).")
    if not np.isfinite(model.tao) or model.tao < 2 * np.log(2):
        raise ValueError("tao must be finite and >= 2*log(2) for these softplus constraints.")
    if not np.isfinite(model.l2_reg) or model.l2_reg < 0:
        raise ValueError("l2_reg must be finite and nonnegative.")

    if equal_opportunity:
        h = model.add_intercept(s, X) @ model.w
        weights = model.w[:-1]
    else:
        h = model.add_features(s, X) @ model.w + model.b
        weights = model.w
    t = cp.Variable(len(y))
    loss = cp.sum(t - cp.multiply(y, h)) / len(y) + model.l2_reg * cp.sum_squares(weights)
    c1 = cp.sum(t[pos]) / pos.sum() + cp.sum(t[neg] - h[neg]) / neg.sum()
    c2 = cp.sum(t[pos] - h[pos]) / pos.sum() + cp.sum(t[neg]) / neg.sum()
    problem = cp.Problem(cp.Minimize(loss), [cp.logistic(h) <= t, c1 <= model.tao, c2 <= model.tao])
    installed = set(cp.installed_solvers())
    for solver, options in [
        ("CLARABEL", {"max_iter": 500}),
        ("ECOS", {"max_iters": 10_000}),
        ("SCS", {"max_iters": 50_000, "eps": 1e-6}),
    ]:
        if solver not in installed:
            continue
        record = {"solver": solver}
        model.training_diagnostics.append(record)
        try:
            problem.solve(solver=solver, warm_start=False, **options)
        except cp.SolverError as error:
            record.update(status="solver_error", error=str(error))
            continue
        record.update(status=problem.status, iterations=problem.solver_stats.num_iters,
                      solve_time=problem.solver_stats.solve_time)
        if h.value is None or not np.isfinite(h.value).all():
            continue
        # Check the original nonlinear constraints, not just their epigraphs.
        logits = h.value
        plus, minus = np.logaddexp(0, logits), np.logaddexp(0, -logits)
        values = [float(plus[pos].mean() + minus[neg].mean()),
                  float(minus[pos].mean() + plus[neg].mean())]
        objective = float(np.mean(plus - y * logits) + model.l2_reg * np.sum(weights.value ** 2))
        violation = max(0.0, max(values) - model.tao)
        record.update(loss=objective, constraint_values=values, max_constraint_violation=violation)
        if problem.status == cp.OPTIMAL and np.isfinite(objective) and violation <= 1e-6:
            model.status_ = problem.status
            print(f"[{model.name}] status: {problem.status}, loss: {objective:.4f}, "
                  f"solver: {solver}, constraint violation: {violation:.2e}")
            return

    model.w.value = None
    if not equal_opportunity:
        model.b.value = None
    raise cp.SolverError(f"{model.name}: no verified optimal solution; "
                         f"attempts: {model.training_diagnostics}")



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
        _train_fair_model(self, s, X, y, equal_opportunity=False)

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

    name = 'Fair Model with Equal Opportunity'

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

        # Equal Opportunity conditions on qualified applicants (y == 1) only.
        X_pos = X[(y == 1) & (s == 1)]
        X_neg = X[(y == 1) & (s == 0)]
        n_pos = max(1, len(X_pos))
        n_neg = max(1, len(X_neg))

        h_pos = X_pos @ self.w
        h_neg = X_neg @ self.w
        c1 = 1.0 / n_pos * cp.sum(cp.logistic(h_pos)) + 1.0 / n_neg * cp.sum(cp.logistic(-h_neg))
        c2 = 1.0 / n_pos * cp.sum(cp.logistic(-h_pos)) + 1.0 / n_neg * cp.sum(cp.logistic(h_neg))
        return c1, c2

    def train(self, s, X, y):
        _train_fair_model(self, s, X, y, equal_opportunity=True)

    def predict(self, s, X):
        X = self.add_intercept(s, X)
        h = X @ self.w.value
        pred_y = (h >= 0).astype(float)
        p = sigmoid(h) 
        return pred_y, p 

    @property
    def params(self):
        return self.w.value
