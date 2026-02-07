import numpy as np
import csv
from utils import *
from graph import gen_static_random_graph, gen_static_demographic_regular_graph, build_temporal_graph_features

class Bank:
    name = 'Bank'
    params = np.array([2.5, 2, -1, -4.0])

    def __init__(self, params=None, seed=2021):
        self.seed = seed
        if params is not None:
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
        s1, X1 = gen_gaussian(mu1, sigma1, 1, self.n_samples - n_protects) #  non_protected group
        
        # join the posisitve and negative class clusters
        s = np.hstack((s0, s1))
        X = np.vstack((X0, X1))
        
        # shuffle the data
        perm = list(range(0, self.n_samples))
        np.random.shuffle(perm)
        s = s[perm]
        X = X[perm]

        return s.astype(int), X

    def sample_repayment(
        self,
        s,
        X,
        # model,
        *,
        # decision_coef=0.8,
        repayment_coef=0.8,
    ):
        """Sample decision D_t and outcome Y_t.

        - Y_t in {0,1}: 1 = repay,        0 = default

        Notes:
          - `model.predict(s, X)` is interpreted as returning probability `p`, p = P(approve=1).
          - `Bank().predict(s, X)` returns P(repay=1).
        """
        s = np.asarray(s).astype(int)

        # # Decision: sample approval/denial.
        # _, p_dec = model.predict(s, X)
        # p_dec = np.asarray(p_dec, dtype=float)
        # approved = sampling(p_dec, values=[0.0, 1.0], coef=decision_coef).astype(int)

        # Outcome: sample repayment (Y=1 repay, Y=0 default). Meaningful only if approved.
        _, p_repay = Bank().predict(s, X)
        repaid = sampling(np.asarray(p_repay, dtype=float), values=[0.0, 1.0], coef=repayment_coef).astype(int)

        return repaid.astype(int)
        # return approved.astype(int), repaid.astype(int)

    def gen_next_profile(
        self,
        s,
        X,
        model,
        *,
        Dt,
        Yt,
        # decision_coef=0.8,
        repayment_coef=0.8,
        # return_latents=False,
        
    ):
        s = np.asarray(s).astype(int)
        base = np.array([[self.base[int(i)]] for i in s], dtype=float)
        # _, p_repay = Bank().predict(s, X)
        # Yt = sampling(np.asarray(p_repay, dtype=float), values=[0.0, 1.0], coef=repayment_coef).astype(int)

        # Feature dynamics: +eps on repay (Y=1), -eps on default (Y=0)
        theta = np.asarray(model.params[1:-1], dtype=float)
        change = self.eps * theta.reshape(1, -1)
        change = np.repeat(change, repeats=len(s), axis=0)
        y_sign = (2.0 * Yt - 1.0).reshape(-1, 1)  
        change = change * y_sign

        # Apply outcome-driven change only if approved (D=1)
        apply_change = (Dt == 1).astype(float).reshape(-1, 1)
        X_next = X + apply_change * change + np.tile(base, (1, X.shape[1]))
        
        _, p_repay_next = Bank().predict(s, X_next)
        Y_next = sampling(np.asarray(p_repay_next, dtype=float), values=[0.0, 1.0], coef=repayment_coef).astype(int)

        return X_next, Y_next.astype(int)
    
    @staticmethod
    def compute_observed_rejection_gap(
        adj,
        decisions,
        s,
        active=None,
        *,
        exclude_inactive_from_denominator=True,
    ):
        """Compute observed peer-outcome signal O_t based on demographic rejection rates.

        User-specified definition (per node i at time t):
        - The applicant observes the loan decisions of its connected agents.
        - Let r_g be the observed rejection rate among observed neighbors with S=g.
        - O_{i,t} = 1 iff r_{S_i} > r_{1-S_i}, else 0.

        Conventions:
        - decisions is {0,1} with 1 meaning approval (so rejection is 1-decisions).
        - If active is provided:
            - default: inactive/opted-out individuals are treated as "not rejected"
                in observations => decisions=1.
            - if exclude_inactive_from_denominator=True: inactive neighbors are excluded
                from observed-rate denominators.
        """
        adj = np.asarray(adj)
        decisions = np.asarray(decisions).astype(int)
        s = np.asarray(s).astype(int)
        if active is not None:
            active = np.asarray(active).astype(int)

        n = adj.shape[0]
        decisions_eff = decisions.copy()
        if active is not None and not exclude_inactive_from_denominator:
            decisions_eff[active == 0] = 1  # treated as not rejected

        O = np.zeros(n, dtype=int)
        for i in range(n):
            nbrs = np.flatnonzero(adj[i]).astype(int)
            if nbrs.size == 0:
                O[i] = 0
                continue

            if active is not None and exclude_inactive_from_denominator:
                nbrs = nbrs[active[nbrs] == 1]
                if nbrs.size == 0:
                    O[i] = 0
                    continue

            same = s[nbrs] == s[i]
            other = ~same
            same_n = int(same.sum())
            other_n = int(other.sum())
            if same_n == 0 or other_n == 0:
                O[i] = 0
                continue

            # rejection rate = mean(1 - approval)
            same_rate = float((1 - decisions_eff[nbrs][same]).mean())
            other_rate = float((1 - decisions_eff[nbrs][other]).mean())
            O[i] = int(same_rate > other_rate)

        return O


def run_simulation(
    decision_model,
    agent,
    steps,
    *,
    repayment_model=None,
    enforce_demographic_mixing=True,
    k=10,
    k_same=6,
    k_other=4,
    directed=False,
    graph_seed=2026,
    seed=2026,
    decision_coef=0.8,
    repayment_coef=0.8,
):
    """

    Timeline:
        - Reset provides (X_1, Y_1)
        - For each t:
                bank issues D_t
                agents observe O_t, compute U_t, choose A_{t+1}
                continuing agents transition to (X_{t+1}, Y_{t+1})

    Conventions:
        - D_t in {0,1}: 1=approve, 0=deny
        - Y_t in {0,1}: latent repayment ability/label at time t
        - A_t in {0,1}: 1=active/applying, 0=opted out
        - P_t in {0,1}: previous denial indicator (1 means denied at t-1 while active)

    Returns:
        s, adj, edges,
        Xs: [X_1..X_steps],
        Ys: [Y_1..Y_steps] (latent labels),
        Ds: [D_1..D_steps],
        Ps: [P_1..P_steps],
        Os: [O_1..O_steps],
        Us: [U_1..U_steps],
        As: [A_1..A_steps]
    """

    # Default repayment model is the Bank model from generator.py
    if repayment_model is None:
        from generator import Bank

        repayment_model = Bank()

    # Reset: generate population and initial state
    s, X = agent.gen_init_profile()
    s = np.asarray(s).astype(int)
    X = np.asarray(X, dtype=float)
    n = len(s)

    if enforce_demographic_mixing and directed:
        raise ValueError("enforce_demographic_mixing requires an undirected graph")

    if enforce_demographic_mixing:
        adj, edges = gen_static_demographic_regular_graph(
            s, k_same=int(k_same), k_other=int(k_other), seed=int(graph_seed)
        )
    else:
        adj, edges = gen_static_random_graph(
            n, k=int(k), seed=int(graph_seed), directed=bool(directed)
        )

    # Agents get (X_1, Y_1)
    _, p_y = repayment_model.predict(s, X)
    Y = sampling(np.asarray(p_y, dtype=float), values=[0.0, 1.0], coef=float(repayment_coef)).astype(int)

    active = np.ones(n, dtype=int)        # A_1
    prev_denied = np.zeros(n, dtype=int)  # P_1

    Xs = [X]
    Ys = [Y]
    Ds, Ps, Os, Us, As = [], [], [], [], []

    steps = int(steps)
    for t in range(steps):
        X_t = Xs[-1]
        Y_t = Ys[-1]
        A_t = active
        P_t = prev_denied

        # bank issues D_t
        _, p_dec = decision_model.predict(s, X_t)
        D_t = sampling(np.asarray(p_dec, dtype=float), values=[0.0, 1.0], coef=float(decision_coef)).astype(int)
        D_t = np.asarray(D_t, dtype=int)
        D_t[A_t == 0] = 0
        denied = ((A_t == 1) & (D_t == 0)).astype(int)

        # agents observe O_t
        O_t = agent.compute_observed_rejection_gap(
            adj,
            D_t,
            s,
            active=A_t,
            exclude_inactive_from_denominator=True,
        )

        # compute U_t and choose A_{t+1}
        repeated_denial_no_default = (denied == 1) & (P_t == 1)
        peer_spillover = (denied == 1) & (O_t == 1)
        U_t = ((repeated_denial_no_default | peer_spillover) & (A_t == 1)).astype(int)

        A_next = A_t.copy()
        A_next[(U_t == 1) & (A_t == 1)] = 0

        # transition (X_{t+1}, Y_{t+1}) for continuing agents
        continue_mask = (A_t == 1) & (A_next == 1)
        outcome_mask = continue_mask & (D_t == 1)

        X_next = np.asarray(X_t, dtype=float).copy()
        base = np.array([[agent.base[int(i)]] for i in s], dtype=float)
        base_tile = np.tile(base, (1, X_t.shape[1]))
        X_next[continue_mask] = X_t[continue_mask] + base_tile[continue_mask]

        theta = np.asarray(decision_model.params[1:-1], dtype=float)
        if theta.size != X_t.shape[1]:
            raise ValueError("decision_model.params[1:-1] must match feature dimension of X")

        repay_sign = (2.0 * Y_t - 1.0).reshape(-1, 1)
        delta = agent.eps * theta.reshape(1, -1)
        delta = np.repeat(delta, repeats=n, axis=0) * repay_sign
        X_next[outcome_mask] = X_next[outcome_mask] + delta[outcome_mask]

        Y_next = np.asarray(Y_t, dtype=int).copy()
        if continue_mask.any():
            _, p_y_next = repayment_model.predict(s, X_next)
            Y_sample = sampling(np.asarray(p_y_next, dtype=float), values=[0.0, 1.0], coef=float(repayment_coef)).astype(int)
            Y_next[continue_mask] = Y_sample[continue_mask]

        P_next = denied.copy()
        P_next[A_next == 0] = 0

        # record step t as D_t, O_t, U_t, A_t, P_t and then advance
        Ds.append(D_t)
        Os.append(np.asarray(O_t, dtype=int))
        Us.append(np.asarray(U_t, dtype=int))
        As.append(A_t.copy())
        Ps.append(P_t.copy())

        active = A_next
        prev_denied = P_next

        # Store X_{t+1}, Y_{t+1} unless we've already collected X_1..X_steps
        if t < steps - 1:
            Xs.append(X_next)
            Ys.append(Y_next)

    return s, np.asarray(adj), edges, Xs, Ys, Ds, Ps, Os, Us, As



def generate_y_from_bank(s, Xs, bank):
    """Generate Y_t from a bank model under unified convention (Y=1 repay, Y=0 default)."""
    Ys = []
    for X in Xs:
        y_repay, _ = bank.predict(s, X)
        Ys.append(np.asarray(y_repay).astype(int))
    return Ys



def save_agent_panel_csv(
    csv_path,
    *,
    s,
    Xs,
    adj,
    Ds,
    Ys,
    Ps=None,
    Us=None,
    As=None,
    Os=None,
    t0=0,
    neighbor_k=10,
):
    """Save the full generated agent panel into a CSV.

    Writes one row per agent i per timestamp t with columns:
      t,i,S,Y,P,U,A,D,O,X0,X1, D1..Dk

    Conventions:
      - D: 1=approve, 0=deny
      - Y: 1=repay,   0=default
    """
    s = np.asarray(s).astype(int)
    adj = np.asarray(adj)

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be a square (n,n) matrix")
    n = adj.shape[0]
    if s.shape[0] != n:
        raise ValueError("s must have length n")

    T = len(Xs)
    if not (len(Ds) == len(Ys) == T):
        raise ValueError("Xs, Ds, Ys must have the same length")

    def _maybe(series, default_value=0):
        if series is None:
            return [np.full(n, default_value, dtype=int) for _ in range(T)]
        if len(series) != T:
            raise ValueError("Optional series must have the same length as Xs")
        return [np.asarray(v).astype(int) for v in series]

    Ps = _maybe(Ps, 0)
    Us = _maybe(Us, 0)
    As = _maybe(As, 1)
    Os = _maybe(Os, 0)

    # Pre-compute a stable neighbor ordering per node
    neighbors = []
    for i in range(n):
        nbrs = np.flatnonzero(adj[i]).astype(int)
        nbrs = nbrs[nbrs != i]
        nbrs.sort()
        neighbors.append(nbrs)

    header = [
        "t",
        "i",
        "S",
        "Y",
        "P",
        "U",
        "A",
        "D",
        "O",
        "X0",
        "X1",
    ] + [f"D{j + 1}" for j in range(int(neighbor_k))]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for t in range(T):
            X_t = np.asarray(Xs[t], dtype=float)
            if X_t.shape[0] != n:
                raise ValueError("Each X_t must have shape (n, d)")
            if X_t.shape[1] < 2:
                raise ValueError("X_t must have at least 2 features to write X0,X1")

            D_t = np.asarray(Ds[t]).astype(int)
            Y_t = np.asarray(Ys[t]).astype(int)
            P_t = np.asarray(Ps[t]).astype(int)
            U_t = np.asarray(Us[t]).astype(int)
            A_t = np.asarray(As[t]).astype(int)
            O_t = np.asarray(Os[t]).astype(int)

            for i in range(n):
                nbrs = neighbors[i]
                nbr_decisions = [int(D_t[j]) for j in nbrs[: int(neighbor_k)]]
                if len(nbr_decisions) < int(neighbor_k):
                    nbr_decisions += [""] * (int(neighbor_k) - len(nbr_decisions))

                row = [
                    int(t0 + t),
                    int(i),
                    int(s[i]),
                    int(Y_t[i]),
                    int(P_t[i]),
                    int(U_t[i]),
                    int(A_t[i]),
                    int(D_t[i]),
                    int(O_t[i]),
                    float(X_t[i, 0]),
                    float(X_t[i, 1]),
                ] + nbr_decisions
                writer.writerow(row)

