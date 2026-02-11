import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.linear_model import LogisticRegression

eps = 1e-8

def compute_accuracy(s, X, y, model):
    pred_y, _ = model.predict(s, X)
    acc = sum(pred_y == y) / len(y)
    return acc


def compute_equal_opportunity(s, X, y, model): 
    X_pos = X[(y == 1) == (s == 1)]
    X_neg = X[(y == 1) == (s == 0)]
    s_pos = np.ones(len(X_pos))
    s_neg = np.zeros(len(X_neg))

    y_pos, _ = model.predict(s_pos, X_pos)
    y_neg, _ = model.predict(s_neg, X_neg)
    eo_fairness = y_pos.mean() - y_neg.mean()
    return eo_fairness
    

def compute_total_cond_fairness(s, X, model):
    s_pos, s_neg = s[s == 1], s[s == 0]
    X_pos, X_neg = X[s == 1], X[s == 0]

    y_pos, _ = model.predict(s_pos, X_pos)
    y_neg, _ = model.predict(s_neg, X_neg) 
    fairness = y_pos.mean() - y_neg.mean()
    return fairness


def compute_short_cond_fairness(s, X, model):
    X = X[s == 0]
    s = s[s == 0]

    s_pos = np.ones_like(s)
    s_neg = np.zeros_like(s)
        
    y_pos, _ = model.predict(s_pos, X)
    y_neg, _ = model.predict(s_neg, X)

    fairness = y_pos.mean() - y_neg.mean()
    return fairness


def compute_post_long_cond_probs(s, Xs, Ys):
    probs = {}
    
    for i in range(len(Xs)-1):
        
        XXs_comb = np.c_[s[s == 1], Xs[i][s == 1], Xs[i+1][s == 1]]
        Xs_comb = np.c_[s[s == 1], Xs[i][s == 1]]
        lr_up = LogisticRegression(max_iter=1000, random_state=2021).fit(XXs_comb, Ys[i][s == 1])
        lr_dn = LogisticRegression(max_iter=1000, random_state=2021).fit(Xs_comb, Ys[i][s == 1])
        probs_up = lr_up.predict_proba(XXs_comb)
        probs_dn = lr_dn.predict_proba(Xs_comb)
        probs[f'pos(y{i+1}=0)'] = probs_up[:, 0] / probs_dn[:, 0]
        probs[f'pos(y{i+1}=1)'] = probs_up[:, 1] / probs_dn[:, 1]


        XXs_comb = np.c_[s[s == 0], Xs[i][s == 0], Xs[i+1][s == 0]]
        Xs_comb = np.c_[s[s == 0], Xs[i][s == 0]]
        lr_up = LogisticRegression(max_iter=1000, random_state=2021).fit(XXs_comb, Ys[i][s == 0])
        lr_dn = LogisticRegression(max_iter=1000, random_state=2021).fit(Xs_comb, Ys[i][s == 0])
        probs_up = lr_up.predict_proba(XXs_comb)
        probs_dn = lr_dn.predict_proba(Xs_comb)
        probs[f'neg(y{i+1}=0)'] = probs_up[:, 0] / probs_dn[:, 0]
        probs[f'neg(y{i+1}=1)'] = probs_up[:, 1] / probs_dn[:, 1]
    return probs
    

def compute_post_long_cond_fairness(s, Xs, model, prob=None):
    outputs = {}
    for i in range(len(Xs)-1):
        y_pos, _ = model.predict(s[s == 1], Xs[i][s == 1])
        y_neg, _ = model.predict(s[s == 0], Xs[i][s == 0])
        outputs[f'pos(y{i+1}=0)'] = 1 - y_pos
        outputs[f'pos(y{i+1}=1)'] = y_pos
        outputs[f'neg(y{i+1}=0)'] = 1 - y_neg
        outputs[f'neg(y{i+1}=1)'] = y_neg

    y_pos, _ = model.predict(np.zeros_like(s[s == 1]), Xs[-1][s == 1])
    y_neg, _ = model.predict(np.zeros_like(s[s == 0]), Xs[-1][s == 0])

    indices = [[0, 1]] * (len(Xs) - 1)

    part1, part2 = None, None
    for idx in product(*indices):
        pos, neg = 1, 1
        for i in range(len(Xs)-1):
            pos *= (prob[f'pos(y{i+1}={idx[i]})'] * outputs[f'pos(y{i+1}={idx[i]})'])
            neg *= (prob[f'neg(y{i+1}={idx[i]})'] * outputs[f'neg(y{i+1}={idx[i]})'])
        if part1 is None:
            part1 = pos
        else:
            part1 += pos
        if part2 is None:
            part2 = neg
        else:
            part2 += neg

    if part1 is not None or part2 is not None:
        fairness = np.mean(y_pos * part1) - np.mean(y_neg * part2)
    else:
        fairness = np.mean(y_pos) - np.mean(y_neg)
    return fairness


def compute_statistics(s, Xs, Ys, model, OYs=None, As=None):

    retention = compute_retention_rate(Xs, As)
    ret_disparity = compute_retention_disparity(s, As) if As is not None else np.array([])

    for i, (X, y) in enumerate(zip(Xs, Ys)):
        print("-" * 30, f"Step {i + 1} - {model.name}", "-" * 30)

        if OYs is not None:
            acc = compute_accuracy(s, X, OYs[i], model)
        else:
            acc = compute_accuracy(s, X, y, model)
        print(f"Acc: {acc * 100:.1f}%")

        if retention.size:
            print(f"Retention: {retention[i] * 100:.1f}%")
        if ret_disparity.size:
            print(f"Retention Disparity: {ret_disparity[i]:.3f}")

        # op_fair = compute_equal_opportunity(s, X, y, model)
        # print(f"Equal Oppertunity: {abs(op_fair):.3f}")

        # cond_fair = compute_total_cond_fairness(s, X, model)
        # print(f"Total Fairness: {abs(cond_fair):.3f}")

        short_fair_cond = compute_short_cond_fairness(s, X, model)
        print(f"Short Fairness: {abs(short_fair_cond):.3f}")

        if i == 0:
            post_long_cond_fairness = compute_post_long_cond_fairness(s, Xs[:i+1], model)
        else:
            post_long_cond_prob = compute_post_long_cond_probs(s, Xs[:i+1], Ys[:i+1])
            post_long_cond_fairness = compute_post_long_cond_fairness(s, Xs[:i+1], model, post_long_cond_prob)

        print(f"Long fairness: {abs(post_long_cond_fairness):.3f}")
    print("\n")


def compute_retention_rate(Xs, As=None):
    """Compute applicant retention rate over time.

    Definition (per your description):
      retention[t] = (# applicants applying at step t) / (# applicants at step 1)

    Inputs:
      - Xs: list of feature matrices per step.
            If `As` is None, we treat `len(Xs[t])` (rows) as the number applying.
      - As: optional list of activity/apply indicators per step (0/1). If provided,
            we count applicants applying as `sum(As[t] == 1)`.

    Returns:
      - np.ndarray of shape (T,) with retention rates.
    """
    if Xs is None or len(Xs) == 0:
        return np.asarray([], dtype=float)

    T = len(Xs)

    if As is not None:
        if len(As) != T:
            raise ValueError("As must have the same length as Xs")
        denom = int(np.sum(np.asarray(As[0]).astype(int) == 1))
        numerators = [int(np.sum(np.asarray(a).astype(int) == 1)) for a in As]
    else:
        denom = int(np.asarray(Xs[0]).shape[0])
        numerators = [int(np.asarray(X).shape[0]) for X in Xs]

    if denom <= 0:
        return np.full(T, np.nan, dtype=float)

    return np.asarray([n / denom for n in numerators], dtype=float)


def compute_retention_disparity(s, As):
    """Compute |R_t^{S=0} - R_t^{S=1}| at each step.

    Returns:
        np.ndarray of shape (T,) with per-step retention disparity.
    """
    s = np.asarray(s).astype(int)
    T = len(As)
    denom0 = int((s == 0).sum())
    denom1 = int((s == 1).sum())
    if denom0 == 0 or denom1 == 0:
        return np.full(T, np.nan, dtype=float)

    disparity = np.empty(T, dtype=float)
    for t in range(T):
        A_t = np.asarray(As[t]).astype(int)
        r0 = (A_t[s == 0] == 1).sum() / denom0
        r1 = (A_t[s == 1] == 1).sum() / denom1
        disparity[t] = abs(r0 - r1)
    return disparity


def plot_simulation_summary(s, Xs, Ys, model, OYs=None, As=None):
    """Visualise training, dynamics, and evaluation across simulation steps.

    Row 1: Feature scatter plots (X0 vs X1) with decision boundary at each step.
    Row 2: Line plots of Acc, Retention (per-group), Short Fairness,
           Long Fairness, and Retention Disparity over steps.
    """
    s = np.asarray(s).astype(int)
    T = len(Xs)
    steps = np.arange(1, T + 1)

    # ── collect metrics ──
    accs, s_fairs, l_fairs = [], [], []
    retention = compute_retention_rate(Xs, As)
    ret_disp = compute_retention_disparity(s, As) if As is not None else np.zeros(T)

    # per-group retention
    denom0 = max(1, int((s == 0).sum()))
    denom1 = max(1, int((s == 1).sum()))
    ret0, ret1 = np.ones(T), np.ones(T)
    if As is not None:
        for t in range(T):
            A_t = np.asarray(As[t]).astype(int)
            ret0[t] = (A_t[s == 0] == 1).sum() / denom0
            ret1[t] = (A_t[s == 1] == 1).sum() / denom1

    for i in range(T):
        X_t = Xs[i]
        y_true = OYs[i] if OYs is not None else Ys[i]
        accs.append(compute_accuracy(s, X_t, y_true, model))
        s_fairs.append(abs(compute_short_cond_fairness(s, X_t, model)))
        if i == 0:
            l_fairs.append(abs(compute_post_long_cond_fairness(s, Xs[:i+1], model)))
        else:
            prob = compute_post_long_cond_probs(s, Xs[:i+1], Ys[:i+1])
            l_fairs.append(abs(compute_post_long_cond_fairness(s, Xs[:i+1], model, prob)))

    # ── decision boundary helper ──
    def _boundary_lines(model, x_range=(-10, 15)):
        params = model.params
        if params is None:
            return None, None
        x = np.linspace(*x_range, 200)
        if abs(params[2]) < 1e-12:
            return None, None
        s0 = np.zeros_like(x)
        s1 = np.ones_like(x)
        y0 = (-params[-1] - params[0] * s0 - params[1] * x) / params[2]
        y1 = (-params[-1] - params[0] * s1 - params[1] * x) / params[2]
        return x, (y0, y1)

    # ── figure ──
    fig = plt.figure(figsize=(4 * T + 1, 10))
    gs = fig.add_gridspec(2, T, hspace=0.35, wspace=0.35)

    # Row 1: scatter plots per step
    bx, blines = _boundary_lines(model)
    for t in range(T):
        ax = fig.add_subplot(gs[0, t])
        X_t = np.asarray(Xs[t])
        Y_t = np.asarray(OYs[t] if OYs is not None else Ys[t]).astype(int)
        if As is not None:
            A_t = np.asarray(As[t]).astype(int)

        for g, marker in [(0, 'x'), (1, 'o')]:
            mask_g = s == g
            for lbl, color in [(1, 'green'), (0, 'red')]:
                m = mask_g & (Y_t == lbl)
                if As is not None:
                    m = m & (A_t == 1)
                if m.sum() == 0:
                    continue
                fc = 'none' if g == 1 else color
                ax.scatter(X_t[m, 0], X_t[m, 1], c=color, marker=marker,
                           facecolors=fc, s=18, alpha=0.35, linewidths=1)

        if bx is not None and blines is not None:
            ax.plot(bx, blines[0], 'b--', linewidth=1, label='s=0')
            ax.plot(bx, blines[1], 'b-', linewidth=1, label='s=1')

        ax.set_xlim(-12, 16)
        ax.set_ylim(-12, 16)
        ax.set_title(f'Step {t+1}', fontsize=11)
        ax.set_xlabel('$X_0$', fontsize=10)
        if t == 0:
            ax.set_ylabel('$X_1$', fontsize=10)
        if t == T - 1:
            ax.legend(fontsize=8, loc='upper left')

    # Row 2: metrics over time
    metric_ax = fig.add_subplot(gs[1, :])
    metric_ax.plot(steps, accs, 'o-', label='Accuracy', color='tab:blue')
    metric_ax.plot(steps, retention, 's-', label='Retention', color='tab:green')
    metric_ax.plot(steps, ret0, '^--', label='Retention S=0', color='tab:orange')
    metric_ax.plot(steps, ret1, 'v--', label='Retention S=1', color='tab:purple')
    metric_ax.plot(steps, s_fairs, 'D-', label='Short Fairness', color='tab:red')
    metric_ax.plot(steps, l_fairs, 'P-', label='Long Fairness', color='tab:brown')
    metric_ax.plot(steps, ret_disp, 'X-', label='Retention Disparity', color='tab:pink')
    metric_ax.set_xlabel('Step', fontsize=11)
    metric_ax.set_ylabel('Value', fontsize=11)
    metric_ax.set_xticks(steps)
    metric_ax.set_title(f'{model.name} — Metrics over Time', fontsize=12)
    metric_ax.legend(fontsize=9, ncol=4, loc='upper right')
    metric_ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
