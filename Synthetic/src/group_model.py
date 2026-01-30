# lt_fair_model.py
# Long-term fairness model with pluggable short-term group-fairness metrics
# Requires: torch, numpy, scikit-learn (for the long-term path weights)

import numpy as np
from itertools import product
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from evaluation import *

torch.manual_seed(2021)

# ----------------------------
# utils
# ----------------------------
def to_tensor(x):
    if not torch.is_tensor(x):
        x = torch.FloatTensor(x)
    return x

def to_numpy(x):
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().numpy()
    return x

def softplus(x):
    # surrogate φ(z) = log(1+e^z)
    return torch.log1p(torch.exp(x))

def combine_features(s, X):
    s = to_tensor(s)
    X = to_tensor(X)
    return torch.cat((s.view(-1, 1), X), dim=1)

# ----------------------------
# Long-term path probability helpers (Eq. 1 style)
# ----------------------------
def compute_post_long_cond_probs(s: np.ndarray,
                                 Xs: List[np.ndarray],
                                 Ys: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    For i=1..t*-1 and group g∈{pos(s==1), neg(s==0)}, estimate ratios:
      r_g^i(y) ≈ P(Y^i=y | S,X^i,X^{i+1}) / P(Y^i=y | S,X^i)
    via two logistic models (numerator vs denominator).
    These ratios act as identification / importance weights in the long-term product.
    """
    probs = {}
    rng = np.random.RandomState(2021)
    for i in range(len(Xs) - 1):

        # group s==1 (we call 'pos')
        mask = (s == 1)
        XX_up = np.c_[s[mask], Xs[i][mask], Xs[i + 1][mask]]
        X_dn  = np.c_[s[mask], Xs[i][mask]]
        y     = Ys[i][mask]
        lr_up = LogisticRegression(random_state=2021, max_iter=200).fit(XX_up, y)
        lr_dn = LogisticRegression(random_state=2021, max_iter=200).fit(X_dn,  y)
        pu = lr_up.predict_proba(XX_up)  # [:,0],[ :,1]
        pd = lr_dn.predict_proba(X_dn)
        probs[f'pos(y{i+1}=0)'] = pu[:, 0] / (pd[:, 0] + 1e-8)
        probs[f'pos(y{i+1}=1)'] = pu[:, 1] / (pd[:, 1] + 1e-8)

        # group s==0 (we call 'neg')
        mask = (s == 0)
        XX_up = np.c_[s[mask], Xs[i][mask], Xs[i + 1][mask]]
        X_dn  = np.c_[s[mask], Xs[i][mask]]
        y     = Ys[i][mask]
        lr_up = LogisticRegression(random_state=2021, max_iter=200).fit(XX_up, y)
        lr_dn = LogisticRegression(random_state=2021, max_iter=200).fit(X_dn,  y)
        pu = lr_up.predict_proba(XX_up)
        pd = lr_dn.predict_proba(X_dn)
        probs[f'neg(y{i+1}=0)'] = pu[:, 0] / (pd[:, 0] + 1e-8)
        probs[f'neg(y{i+1}=1)'] = pu[:, 1] / (pd[:, 1] + 1e-8)
    return probs

# ----------------------------
# Model
# ----------------------------
class FairModel(nn.Module):
    """
    Long-term Fair Model with pluggable short-term fairness metrics.
    θ = (W,b) is linear; ŷ = σ(hθ(S,X)).
    Objective: l = l_u + λ_s * l_s(metric) + λ_l * l_l (path-specific long-term).
    """
    name = 'Long-term Fair Model (metric-pluggable)'

    def __init__(self, n_features: int, lr: float, l2_reg: float,
                 sf_reg: float, lf_reg: float):
        super().__init__()
        self.l2_reg = l2_reg
        self.sf_reg = sf_reg
        self.lf_reg = lf_reg

        self.linear = nn.Linear(n_features, 1)
        self.sigmoid = nn.Sigmoid()
        self.ce = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.old_linear_weight = None
        self.old_linear_bias = None

        torch.manual_seed(2021)
        nn.init.normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    # ---------- core forward ----------
    def forward(self, s, X):
        Xs = combine_features(s, X)         # concat(S, X)
        h = self.linear(Xs)                  # hθ(S,X)
        p = self.sigmoid(h)                  # σ(h)
        return h.squeeze(), p.squeeze()

    def predict(self, s, X):
        _, p = self.forward(s, X)            # probs
        yhat = torch.round(p)                # hard labels
        return to_numpy(yhat), to_numpy(p)

    @property
    def params(self):
        w = to_numpy(self.linear.weight)[0]
        b = to_numpy(self.linear.bias)
        return np.hstack([w, b])

    def save_params(self):
        self.old_linear_weight = self.linear.weight.detach().clone()
        self.old_linear_bias = self.linear.bias.detach().clone()

    # ---------- utility loss (CE + L2) ----------
    def compute_utility_loss(self, s, X, y):
        y = to_tensor(y).float()
        _, p = self.forward(s, X)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        loss = self.ce(p, y)
        l2 = torch.norm(self.linear.weight) ** 2
        return loss + self.l2_reg * l2

    # ---------- short-term fairness losses (metric=...) ----------
    def short_term_loss(self, s, X, y, metric: str) -> torch.Tensor:
        """
        Compute per-step group-fairness penalty using probabilities p=σ(hθ).
        s: {0,1}, 1=protected(s-), 0=unprotected(s+)
        y: {0,1} labels
        metric in:
          'dp','esp','eo','eodds','fpr','fnr','recall','ppv','npv','fdr','for'
        Returns a scalar penalty (squared disparity by default).
        """
        s_t = to_tensor(s).float()
        X_t = to_tensor(X).float()
        y_t = to_tensor(y).float()

        _, p = self.forward(s_t, X_t)            # prob of positive prediction

        mask_pos = (s_t == 1)  # group s-
        mask_neg = (s_t == 0)  # group s+

        eps = 1e-6

        def mean_safe(v, m):
            denom = torch.clamp(m.float().sum(), min=1.0)
            return (v[m].sum() / denom)

        if metric in ['dp', 'esp', 'demographic_parity', 'equal_selection_parity']:
            # DP: P(Ŷ=1|S=1) - P(Ŷ=1|S=0)
            g1 = mean_safe(p, mask_pos)
            g0 = mean_safe(p, mask_neg)
            gap = g1 - g0
            return gap.pow(2)

        if metric in ['eo', 'recall', 'tpr_parity', 'equal_opportunity']:
            # EO / Recall parity: P(Ŷ=1|Y=1,S=1) - P(Ŷ=1|Y=1,S=0)
            mask_y1 = (y_t == 1)
            g1 = mean_safe(p, mask_pos & mask_y1)
            g0 = mean_safe(p, mask_neg & mask_y1)
            gap = g1 - g0
            return gap.pow(2)

        if metric in ['fpr', 'false_positive_rate_parity']:
            # FPR parity: P(Ŷ=1|Y=0,S=1) - P(Ŷ=1|Y=0,S=0)
            mask_y0 = (y_t == 0)
            g1 = mean_safe(p, mask_pos & mask_y0)
            g0 = mean_safe(p, mask_neg & mask_y0)
            gap = g1 - g0
            return gap.pow(2)

        if metric in ['fnr', 'false_negative_rate_parity']:
            # FNR parity: P(Ŷ=0|Y=1,S=·) = 1 - P(Ŷ=1|Y=1,S=·)
            mask_y1 = (y_t == 1)
            g1 = 1.0 - mean_safe(p, mask_pos & mask_y1)
            g0 = 1.0 - mean_safe(p, mask_neg & mask_y1)
            gap = g1 - g0
            return gap.pow(2)

        if metric in ['eodds', 'equalized_odds']:
            # EOdds: TPR parity + FPR parity
            mask_y1 = (y_t == 1)
            mask_y0 = (y_t == 0)
            tpr1 = mean_safe(p, mask_pos & mask_y1)
            tpr0 = mean_safe(p, mask_neg & mask_y1)
            fpr1 = mean_safe(p, mask_pos & mask_y0)
            fpr0 = mean_safe(p, mask_neg & mask_y0)
            gap_tpr = tpr1 - tpr0
            gap_fpr = fpr1 - fpr0
            return gap_tpr.pow(2) + gap_fpr.pow(2)

        if metric in ['ppv', 'precision_parity', 'predictive_parity']:
            # PPV parity: P(Y=1 | Ŷ=1,S=·) ≈ E[Y*p]/E[p]
            num1 = (y_t * p)[mask_pos].sum()
            den1 = p[mask_pos].sum() + eps
            num0 = (y_t * p)[mask_neg].sum()
            den0 = p[mask_neg].sum() + eps
            ppv1 = num1 / den1
            ppv0 = num0 / den0
            gap = ppv1 - ppv0
            return gap.pow(2)

        if metric in ['npv', 'npv_parity']:
            # NPV parity: P(Y=0 | Ŷ=0,S=·) ≈ E[(1-Y)*(1-p)] / E[(1-p)]
            q = 1.0 - p
            num1 = ((1.0 - y_t) * q)[mask_pos].sum()
            den1 = q[mask_pos].sum() + eps
            num0 = ((1.0 - y_t) * q)[mask_neg].sum()
            den0 = q[mask_neg].sum() + eps
            npv1 = num1 / den1
            npv0 = num0 / den0
            gap = npv1 - npv0
            return gap.pow(2)

        if metric in ['fdr', 'false_discovery_rate_parity']:
            # FDR parity: P(Y=0 | Ŷ=1,S=·) = 1 - PPV
            num1 = ((1.0 - y_t) * p)[mask_pos].sum()
            den1 = p[mask_pos].sum() + eps
            num0 = ((1.0 - y_t) * p)[mask_neg].sum()
            den0 = p[mask_neg].sum() + eps
            fdr1 = num1 / den1
            fdr0 = num0 / den0
            gap = fdr1 - fdr0
            return gap.pow(2)

        if metric in ['for', 'false_omission_rate_parity']:
            # FOR parity: P(Y=1 | Ŷ=0,S=·) ≈ E[Y*(1-p)] / E[(1-p)]
            q = 1.0 - p
            num1 = (y_t * q)[mask_pos].sum()
            den1 = q[mask_pos].sum() + eps
            num0 = (y_t * q)[mask_neg].sum()
            den0 = q[mask_neg].sum() + eps
            for1 = num1 / den1
            for0 = num0 / den0
            gap = for1 - for0
            return gap.pow(2)

        raise ValueError(f"Unknown short-term fairness metric: {metric}")

    # ---------- long-term fairness loss (path-specific, horizon) ----------
    def long_term_loss(self, s: np.ndarray,
                       Xs: List[np.ndarray],
                       Ys: List[np.ndarray],
                       probs: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Surrogate version of path-specific long-term gap:
        E[φ(hθ) * product weights] (s+)  -  E[φ(-hθ) * product weights] (s−) - 1
        Returns a non-negative hinge via ReLU.
        """
        # Collect φ(±h) for intermediate steps per group
        outputs = {}
        for i in range(len(Xs) - 1):
            y_pos, _ = self.forward(s[s == 1], Xs[i][s == 1])
            y_neg, _ = self.forward(s[s == 0], Xs[i][s == 0])
            outputs[f'pos(y{i+1}=0)'] = softplus(-y_pos)
            outputs[f'pos(y{i+1}=1)'] = softplus( y_pos)
            outputs[f'neg(y{i+1}=0)'] = softplus(-y_neg)
            outputs[f'neg(y{i+1}=1)'] = softplus( y_neg)

        # Horizon: evaluate with path-specific S setting (block direct S->Y^T)
        y_pos_T, _ = self.forward(np.zeros_like(s[s == 1]), Xs[-1][s == 1])
        y_neg_T, _ = self.forward(np.zeros_like(s[s == 0]), Xs[-1][s == 0])

        # Sum over all sequences y^1..y^{T-1}
        indices = [[0, 1]] * (len(Xs) - 1)
        part1, part2 = 0.0, 0.0
        for idx in product(*indices):
            pos, neg = 1.0, 1.0
            for i in range(len(Xs) - 1):
                pos *= to_tensor(probs[f'pos(y{i+1}={idx[i]})']) * outputs[f'pos(y{i+1}={idx[i]})']
                neg *= to_tensor(probs[f'neg(y{i+1}={idx[i]})']) * outputs[f'neg(y{i+1}={idx[i]})']
            part1 += pos
            part2 += neg

        fair1 = torch.mean(softplus( y_pos_T) * part1)
        fair2 = torch.mean(softplus(-y_neg_T) * part2)
        # one-sided constraint as in the original code
        return torch.relu(fair1 + fair2 - 1.0)

    # ---------- training ----------
    def train_model(self,
                    s: np.ndarray,
                    OXs: List[np.ndarray], OYs: List[np.ndarray],
                    Xs:  List[np.ndarray],  Ys:  List[np.ndarray],
                    short_metric: str = 'eo',
                    epochs: int = 200,
                    tol: float = 1e-7,
                    verbose: bool = True):
        """
        Train with chosen short-term metric.
        OX/OY: original per-step data for utility term (pre-intervention).
        X/Y:   per-step current data for fairness terms (post-intervention sampling).
        """
        losses, u_losses, s_losses, l_losses = [], [], [], []

        gap = float('inf')
        prev_loss = float('inf')

        # Precompute long-term path weights once per outer RRM iteration
        long_probs = compute_post_long_cond_probs(s, Xs, Ys)

        while epochs > 0 or gap > tol:
            total_u, total_s = 0.0, 0.0

            # sum utility + short-term across time
            for OX, Oy, X, y in zip(OXs, OYs, Xs, Ys):
                # Ensure labels are {0,1}
                Oy_bin = (to_tensor(Oy).float() > 0.5).float()
                y_bin  = (to_tensor(y ).float() > 0.5).float()

                total_u += self.compute_utility_loss(s, OX, Oy_bin)
                total_s += self.short_term_loss(s, X, y_bin, metric=short_metric)

            # long-term fairness at horizon
            l_long = self.long_term_loss(s, Xs, Ys, long_probs)

            # final objective
            loss = total_u + self.sf_reg * total_s + self.lf_reg * l_long

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # book-keeping
            losses.append(loss.item())
            u_losses.append(total_u.item() if torch.is_tensor(total_u) else total_u)
            s_losses.append(total_s.item() if torch.is_tensor(total_s) else total_s)
            l_losses.append(l_long.item())

            gap = prev_loss - loss.item()
            prev_loss = loss.item()
            epochs -= 1

        self.save_params()
        if verbose:
            print(f"finished training: total_loss={losses[-1]:.4f}, "
                  f"utility={u_losses[-1]:.4f}, short={s_losses[-1]:.4f}, long={l_losses[-1]:.4f}")
        return losses, u_losses, s_losses, l_losses

# ----------------------------
# (Optional) simple evaluators
# ----------------------------
def compute_accuracy(s, X, y, model: FairModel):
    yhat, _ = model.predict(s, X)
    return (yhat == y).mean()

def compute_short_gap(s, X, y, model: FairModel, metric='eo'):
    # returns the *unsigned* empirical disparity for inspection (no gradients)
    s_t = to_tensor(s).float()
    X_t = to_tensor(X).float()
    y_t = to_tensor(y).float()
    _, p = model.forward(s_t, X_t)

    mask_pos = (s_t == 1)
    mask_neg = (s_t == 0)
    eps = 1e-6
    def mean_safe(v, m): 
        denom = torch.clamp(m.float().sum(), min=1.0)
        return (v[m].sum() / denom)

    if metric in ['dp', 'esp', 'demographic_parity', 'equal_selection_parity']:
        return abs(mean_safe(p, mask_pos) - mean_safe(p, mask_neg)).item()
    if metric in ['eo', 'recall', 'tpr_parity', 'equal_opportunity']:
        m = (y_t == 1)
        return abs(mean_safe(p, mask_pos & m) - mean_safe(p, mask_neg & m)).item()
    if metric in ['fpr', 'false_positive_rate_parity']:
        m = (y_t == 0)
        return abs(mean_safe(p, mask_pos & m) - mean_safe(p, mask_neg & m)).item()
    if metric in ['fnr', 'false_negative_rate_parity']:
        m = (y_t == 1)
        return abs((1-mean_safe(p, mask_pos & m)) - (1-mean_safe(p, mask_neg & m))).item()
    if metric in ['eodds', 'equalized_odds']:
        m1 = (y_t == 1); m0 = (y_t == 0)
        tpr_gap = abs(mean_safe(p, mask_pos & m1) - mean_safe(p, mask_neg & m1)).item()
        fpr_gap = abs(mean_safe(p, mask_pos & m0) - mean_safe(p, mask_neg & m0)).item()
        return tpr_gap + fpr_gap
    if metric in ['ppv', 'precision_parity', 'predictive_parity']:
        num1 = (y_t * p)[mask_pos].sum(); den1 = p[mask_pos].sum() + eps
        num0 = (y_t * p)[mask_neg].sum(); den0 = p[mask_neg].sum() + eps
        return abs((num1/den1 - num0/den0).item())
    if metric in ['npv', 'npv_parity']:
        q = 1-p
        num1 = ((1-y_t) * q)[mask_pos].sum(); den1 = q[mask_pos].sum() + eps
        num0 = ((1-y_t) * q)[mask_neg].sum(); den0 = q[mask_neg].sum() + eps
        return abs((num1/den1 - num0/den0).item())
    if metric in ['fdr', 'false_discovery_rate_parity']:
        num1 = ((1-y_t)*p)[mask_pos].sum(); den1 = p[mask_pos].sum() + eps
        num0 = ((1-y_t)*p)[mask_neg].sum(); den0 = p[mask_neg].sum() + eps
        return abs((num1/den1 - num0/den0).item())
    if metric in ['for', 'false_omission_rate_parity']:
        q = 1-p
        num1 = (y_t*q)[mask_pos].sum(); den1 = q[mask_pos].sum() + eps
        num0 = (y_t*q)[mask_neg].sum(); den0 = q[mask_neg].sum() + eps
        return abs((num1/den1 - num0/den0).item())
    raise ValueError(metric)

# ----------------------------
# Example usage (pseudo)
# ----------------------------
# from your data generator:
# s: (n,)
# OXs, OYs: lists of arrays per time (original/pre-intervention data for utility)
# Xs,  Ys : lists of arrays per time (current/post-intervention data for fairness)
#
# model = FairModel(n_features = 1 + Xs[0].shape[1],  # +1 for S
#                   lr=1e-2, l2_reg=1e-3,
#                   sf_reg=1.0, lf_reg=1.0)
# losses, u, s, l = model.train_model(s, OXs, OYs, Xs, Ys,
#                                     short_metric='equalized_odds',
#                                     epochs=300)
# gap = compute_short_gap(s, Xs[-1], Ys[-1], model, metric='equalized_odds')
# print('final short-term gap:', gap)
