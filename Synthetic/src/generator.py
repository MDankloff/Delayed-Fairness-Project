import numpy as np
from utils import *



class Bank:
    
    name = 'Bank'
    params = np.array([2.5, 2, -1, -4.0])

    def __init__(self, params=None, seed=2021):
        self.seed = seed
        if params:
            self.params = params

    def predict(self, s, X):
        Xs = np.c_[s, X, np.ones(len(s))]
        p = sigmoid(Xs @ self.params / 3.)
        y = (p >= 0.5).astype(float)
        return y, p


class Agent:

    def __init__(self, n_samples, protect_ratio, eps, base, seed=2021):
        self.n_samples = n_samples
        self.protect_ratio = protect_ratio
        self.eps = eps
        self.base = base
        self.seed = seed

    def set_eps(self, eps):
        self.eps = eps

    def gen_init_profile(self):
        """
        Reference from https://github.com/mbilalzafar/fair-classification/blob/master/disparate_impact/synthetic_data_demo/generate_synthetic_data.py
        """
        np.random.seed(self.seed)

        def gen_gaussian(mean, cov, sen_label, sample_size):
            s = np.ones(sample_size, dtype=float) * sen_label
            X = np.random.multivariate_normal(mean=mean, cov=cov, size=sample_size)
            return s, X

        n_protects = int(self.protect_ratio * self.n_samples)

        # We will generate one gaussian cluster for each group
        mu0, sigma0 = [-2, -2], [[10, 1], [1, 5]]
        mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
        s0, X0 = gen_gaussian(mu0, sigma0, 0, n_protects) # protected group
        s1, X1 = gen_gaussian(mu1, sigma1, 1, self.n_samples - n_protects) #  non_protected groupx
        
        # join the posisitve and negative class clusters
        s = np.hstack((s0, s1))
        X = np.vstack((X0, X1))
        
        # shuffle the data
        perm = list(range(0, self.n_samples))
        np.random.shuffle(perm)
        s = s[perm]
        X = X[perm]

        return s, X

    def gen_next_profile(self, s, X, model):
        base = [[self.base[int(i)]] for i in s]
        _, prob = model.predict(s, X)
        sample_y = sampling(prob, coef=0.8)
        _, def_prob = Bank().predict(s, X)
        default = sampling(def_prob, values=[-1, 1.], coef=0.8)

        # X change
        change = self.eps * model.params[1:-1] * prob.reshape(-1, 1)   # test w/wo prob
        # Whether default
        default_change = change * default.reshape(-1, 1)
        # Whether getting the loan
        X_next = X + sample_y.reshape(-1, 1) * default_change + np.tile(base, 2)
        return X_next


def gen_multi_step_profiles(model, agent, steps, noise=(0.05, 0.1), seed=2021):
    np.random.seed(2021)
    noise_list = noise[0] + (noise[1] - noise[0]) * np.random.rand(steps)
    
    Xs, Ys = [], []

    s, init_X = agent.gen_init_profile()
    init_Y, prob = model.predict(s, init_X)
    init_Y = sampling(prob, coef=0.8)

    Xs.append(init_X)
    Ys.append(init_Y)

    for i in range(1, steps):
        next_X = agent.gen_next_profile(s, Xs[-1], model)
        next_Y, prob = model.predict(s, next_X)
        next_Y = sampling(prob, coef=0.8)

        Xs.append(next_X)
        Ys.append(next_Y)

    return s, Xs, Ys


def generate_y_from_bank(s, Xs, bank):
    Ys = []
    for X in Xs:
        y, _ = bank.predict(s, X)
        Ys.append(y)
    return Ys

# BAF workflow: empirical initialization and explicitly specified dynamics.
# The original Gaussian generator above remains available to synthetic experiments.
BAF_FEATURES = ('credit_risk_score', 'proposed_credit_limit')


def split_baf_data(df, test_size=0.2, seed=42):
    """Split source rows before fitting scaling or the repayment model."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    frame = df.copy()
    if 'S' not in frame:
        frame['S'] = (frame['customer_age'] >= 50).astype(int)
    raw = frame[list(BAF_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError('BAF features must be finite.')
    if not np.isin(frame[['S', 'fraud_bool']].to_numpy(), [0, 1]).all():
        raise ValueError('S and fraud_bool must be binary.')
    train_ids, test_ids = train_test_split(
        np.arange(len(frame)), test_size=test_size, random_state=seed,
        stratify=frame[['S', 'fraud_bool']],
    )
    scaler = StandardScaler().fit(raw[train_ids])

    def pool(ids):
        return dict(features=scaler.transform(raw[ids]), raw_features=raw[ids].copy(),
                    age_groups=frame['S'].to_numpy(dtype=int)[ids],
                    fraud_labels=frame['fraud_bool'].to_numpy(dtype=int)[ids],
                    row_ids=ids.copy(), scaler=scaler)

    return dict(train=pool(train_ids), test=pool(test_ids), scaler=scaler,
                train_df=frame.iloc[train_ids].copy())


class BAFBank:
    """Fixed non-fraud proxy model; probabilities have no temperature adjustment."""
    name = 'BAF repayment bank'

    def __init__(self, params):
        self.params = np.asarray(params, dtype=float)

    def predict(self, s, X):
        from scipy.special import expit
        p = expit(np.c_[s, X, np.ones(len(s))] @ self.params)
        return (p >= 0.5).astype(float), p


def fit_baf_bank(training_pool, C_values=(0.01, 0.1, 1.0, 10.0), cv=3, seed=42):
    """Select C by training-only CV log loss; fit scalers inside each fold.

    The returned coefficients expect the full-training-pool standardized X.
    BAF's non-fraud target is a repayment proxy, not observed loan repayment.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    s = training_pool['age_groups']
    raw = training_pool['raw_features']
    y = 1 - training_pool['fraud_labels']
    strata = 2 * np.asarray(s) + y
    if cv < 2 or np.bincount(strata, minlength=4).min() < cv:
        raise ValueError('Each age/label stratum must contain at least cv rows.')
    pipeline = Pipeline([
        ('features', ColumnTransformer([
            ('group', 'passthrough', [0]),
            ('scale', StandardScaler(), list(range(1, raw.shape[1] + 1))),
        ])),
        ('logistic', LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    folds = list(StratifiedKFold(cv, shuffle=True, random_state=seed).split(raw, strata))
    search = GridSearchCV(pipeline, {'logistic__C': list(C_values)},
                          scoring='neg_log_loss', cv=folds, n_jobs=1, error_score='raise')
    search.fit(np.c_[s, raw], y)
    logistic = search.best_estimator_.named_steps['logistic']
    bank = BAFBank(np.r_[logistic.coef_[0], logistic.intercept_[0]])
    bank.selected_C = float(search.best_params_['logistic__C'])
    bank.cv_log_loss = float(-search.best_score_)
    bank.cv_results = [dict(C=float(c), log_loss=float(-score)) for c, score in zip(
        search.cv_results_['param_logistic__C'], search.cv_results_['mean_test_score'])]
    bank.scaler = search.best_estimator_.named_steps['features'].named_transformers_['scale']
    np.testing.assert_allclose(bank.scaler.mean_, training_pool['scaler'].mean_)
    np.testing.assert_allclose(bank.scaler.scale_, training_pool['scaler'].scale_)
    return bank


class BAFPopulationMixin:
    """Empirical initialization compatible with the network simulator's Agent."""
    def __init__(self, s_pool, X_pool, n_samples, protect_ratio=0.5, eps=0.5,
                 base=(0.0, 0.0), seed=2021, row_ids=None, scaler=None, y_pool=None):
        super().__init__(n_samples=n_samples, protect_ratio=protect_ratio,
                         eps=eps, base=base, seed=seed)
        self._s_pool = np.asarray(s_pool)
        self._X_pool = np.asarray(X_pool, dtype=float)
        self._y_pool = None if y_pool is None else np.asarray(y_pool)
        self._row_ids = np.arange(len(s_pool)) if row_ids is None else np.asarray(row_ids)
        self.scaler = scaler
        if not 0 <= protect_ratio <= 1 or eps < 0 or not np.isfinite(eps):
            raise ValueError('Require protect_ratio in [0,1] and finite eps >= 0.')
        if len(self._X_pool) != len(s_pool) or len(self._row_ids) != len(s_pool):
            raise ValueError('Population arrays must be aligned.')
        if self._y_pool is not None and (self._y_pool.shape != self._s_pool.shape
                                        or not np.isin(self._y_pool, [0, 1]).all()):
            raise ValueError('Initial repayment labels must be aligned and binary.')

    def gen_init_profile(self):
        rng = np.random.default_rng(self.seed)
        n1 = int(round(self.n_samples * self.protect_ratio))
        selected = []
        for group, size in [(0, self.n_samples - n1), (1, n1)]:
            candidates = np.flatnonzero(self._s_pool == group)
            if size > len(candidates):
                raise ValueError(f'Not enough BAF rows in group {group} for sampling without replacement.')
            selected.append(rng.choice(candidates, size=size, replace=False))
        indices = np.concatenate(selected)
        rng.shuffle(indices)
        self.sampled_row_ids = self._row_ids[indices].copy()
        self.initial_repayment_labels = (None if self._y_pool is None
                                         else self._y_pool[indices].astype(int).copy())
        return self._s_pool[indices].astype(int).copy(), self._X_pool[indices].copy()


def baf_feature_update(s, X, Y, D, continue_mask, agent, decision_model, rng):
    """Taiwan-style full-vector transition using the current repayment label.

    Delta X = b_s + eps * theta_X * p_decision * (2Y-1) * 1{D=1}.
    Only continuing applicants change; no credit-limit regression is applied.
    """
    X_next = np.asarray(X, dtype=float).copy()
    continuing = np.asarray(continue_mask, dtype=bool)
    theta = np.asarray(decision_model.params[1:-1], dtype=float)
    if theta.shape != (X_next.shape[1],):
        raise ValueError('Policy feature coefficients must match the BAF feature dimension.')
    _, probability = decision_model.predict(s, X)
    probability = np.asarray(probability, dtype=float)
    if probability.shape != (len(X),) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Decision probabilities must be finite and in [0,1].')
    X_next[continuing] += np.asarray(agent.base)[np.asarray(s, dtype=int)[continuing], None]
    approved = continuing & (np.asarray(D) == 1)
    X_next[approved] += (agent.eps * theta[None, :] * probability[approved, None]
                         * (2 * np.asarray(Y)[approved, None] - 1))
    return X_next
