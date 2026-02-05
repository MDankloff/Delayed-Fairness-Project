import numpy as np
import csv
from utils import *
from graph import gen_static_random_graph, gen_static_demographic_regular_graph, build_temporal_graph_features

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

    # def set_eps(self, eps):
    #     self.eps = eps

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

        return s.astype(int), X

    def sample_decision_and_repayment(self,s,X,model,*,decision_coef=0.8,repayment_coef=0.8,decision_positive_is_approval=True,):
        """Sample decision D_t and outcome Y_t.

        Unified convention ("negative" side is 1):
          - D_t in {0,1}: 1 = deny loan, 0 = approve loan
          - Y_t in {0,1}: 1 = default,  0 = repay

        Notes:
          - `model.predict(s, X)` is interpreted as returning a probability `p`.
            If `decision_positive_is_approval=True`, then p = P(approve=1).
            Otherwise p = P(deny=1).
          - `Bank().predict(s, X)` returns P(repay=1).
        """
        s = np.asarray(s).astype(int)

        # Decision: sample approval/denial.
        _, p_dec = model.predict(s, X)
        p_dec = np.asarray(p_dec, dtype=float)
        if decision_positive_is_approval:
            approve = sampling(p_dec, values=[0.0, 1.0], coef=decision_coef).astype(int)
            denied = (1 - approve).astype(int)
        else:
            denied = sampling(p_dec, values=[0.0, 1.0], coef=decision_coef).astype(int)

        # Outcome: Y=1 to mean default.
        _, p_repay = Bank().predict(s, X)
        defaulted = sampling(p_repay, values=[1.0, 0.0], coef=repayment_coef).astype(int)

        return denied.astype(int), defaulted.astype(int)

    def gen_next_profile(self,s,X,model,*,decision_coef=0.8,repayment_coef=0.8,decision_positive_is_approval=True,return_latents=False,):
        """Evolve features one step using (decision, outcome) dynamics.

        Unified convention:
          - D_t=1 deny (no loan), D_t=0 approve (loan granted)
          - Y_t=1 default,        Y_t=0 repay

        Feature update is applied only when a loan is approved (D_t=0).
        """
        s = np.asarray(s).astype(int)
        base = np.array([[self.base[int(i)]] for i in s], dtype=float)

        D01, Y01 = self.sample_decision_and_repayment(
            s,
            X,
            model,
            decision_coef=decision_coef,
            repayment_coef=repayment_coef,
            decision_positive_is_approval=decision_positive_is_approval,
        )

        # Feature dynamics: +eps on repay (Y=0), -eps on default (Y=1)
        theta = np.asarray(model.params[1:-1], dtype=float)
        change = self.eps * theta.reshape(1, -1)
        change = np.repeat(change, repeats=len(s), axis=0)
        y_sign = (1.0 - 2.0 * Y01).reshape(-1, 1)
        change = change * y_sign

        # Apply outcome-driven change only if approved (D=0)
        apply_change = (D01 == 0).astype(float).reshape(-1, 1)
        X_next = X + apply_change * change + np.tile(base, (1, X.shape[1]))

        if return_latents:
            return X_next, D01.astype(int), Y01.astype(int)
        return X_next


def gen_multi_step_process(model, agent, steps, *, decision_coef=0.8, repayment_coef=0.8, seed=2021):
    """Generate the full process (X_t, D_t, Y_t) over time.

    Conventions:
      - X_t has shape (n, d) and we return exactly `steps` snapshots: X_1..X_steps
            - D_t and Y_t are sampled at each snapshot t and both lie in {0, 1}

    Returns:
        s: (n,) sensitive attribute
        Xs: list of X_t, length = steps
        Ds: list of D_t in {0, 1}, length = steps
        Ys: list of Y_t in {0, 1}, length = steps
    """
    np.random.seed(seed)
    Xs, Ds, Ys = [], [], []

    s, X = agent.gen_init_profile()
    Xs.append(X)

    # For t = 1..steps-1, sample (D_t, Y_t) and update X_{t+1}
    for _ in range(steps - 1):
        X_next, d01, y01 = agent.gen_next_profile(
            s,
            Xs[-1],
            model,
            decision_coef=decision_coef,
            repayment_coef=repayment_coef,
            decision_positive_is_approval=True,
            return_latents=True,
        )
        Ds.append(d01)
        Ys.append(y01)
        Xs.append(X_next)

    # For the final snapshot t = steps, sample (D_t, Y_t) without advancing
    d_last, y_last = agent.sample_decision_and_repayment(
        s,
        Xs[-1],
        model,
        decision_coef=decision_coef,
        repayment_coef=repayment_coef,
        decision_positive_is_approval=True,
    )
    Ds.append(d_last)
    Ys.append(y_last)

    return s, Xs, Ds, Ys


def gen_temporal_graph(
    model,
    agent,
    steps,
    *,
    k=10,
    directed=False,
    include_sensitive_in_features=False,
    graph_seed=2021,
    seed=2021,
    enforce_demographic_mixing=True,
    k_same=6,
    k_other=4,
):
    """Generate a temporal graph \mathcal{G} = {G_t}_{t=1..T} with fixed topology.

    By default this matches your paper description:
      - undirected 10-regular graph
      - among each node's 10 neighbors: 6 from same group, 4 from other group
      - adjacency is static across time; only X_t evolves

    Returns:
        s: (n,) sensitive attribute per node (static)
        A: (n,n) adjacency matrix in {0,1} (static)
        E: undirected edge list
        X_ts: list of node feature matrices, one per timestamp
        D_ts: list of decisions in {0, 1}, one per timestamp (1=deny, 0=approve)
        Y_ts: list of outcomes in {0, 1}, one per timestamp (1=default, 0=repay)
    """
    if enforce_demographic_mixing and directed:
        raise ValueError("enforce_demographic_mixing requires an undirected graph")

    s, Xs, D_ts, Y_ts = gen_multi_step_process(
        model,
        agent,
        steps,
        decision_coef=0.8,
        repayment_coef=0.8,
        seed=seed,
    )

    if enforce_demographic_mixing:
        A, E = gen_static_demographic_regular_graph(s, k_same=k_same, k_other=k_other, seed=graph_seed)
    else:
        A, E = gen_static_random_graph(len(s), k=k, seed=graph_seed, directed=directed)

    X_ts = build_temporal_graph_features(s, Xs, include_sensitive=include_sensitive_in_features)
    return s, A, E, X_ts, D_ts, Y_ts


def generate_y_from_bank(s, Xs, bank):
    Ys = []
    for X in Xs:
        # bank.predict returns repayment indicator/probability; convert to default indicator
        y_repay, _ = bank.predict(s, X)
        y_default = (1 - np.asarray(y_repay).astype(int)).astype(int)
        Ys.append(y_default)
    return Ys


def _compute_observed_rejection_gap(
    adj,
    denied,
    s,
    active=None,
    *,
    exclude_inactive_from_denominator=False,
):
    """Compute observed peer-outcome signal O_t based on demographic rejection rates.

    User-specified definition (per node i at time t):
      - The applicant observes the loan decisions of its connected agents.
      - Let r_g be the observed rejection rate among observed neighbors with S=g.
      - O_{i,t} = 1 iff r_{S_i} > r_{1-S_i}, else 0.

    Conventions:
      - denied is {0,1} with 1 meaning rejection/denial.
            - If active is provided:
                    - default: inactive/opted-out individuals are treated as denied=0
                        (i.e., observed as not rejected).
                    - if exclude_inactive_from_denominator=True: inactive/opted-out neighbors
                        are excluded from the observed-rate denominators.
    """
    adj = np.asarray(adj)
    denied = np.asarray(denied).astype(int)
    s = np.asarray(s).astype(int)
    if active is not None:
        active = np.asarray(active).astype(int)

    n = adj.shape[0]
    denied_eff = denied.copy()
    if active is not None and not exclude_inactive_from_denominator:
        denied_eff[active == 0] = 0

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

        # If one side is unobserved, default to 0 (no evidence of disparity)
        if same_n == 0 or other_n == 0:
            O[i] = 0
            continue

        same_rate = float(denied_eff[nbrs][same].mean())
        other_rate = float(denied_eff[nbrs][other].mean())
        O[i] = int(same_rate > other_rate)

    return O


# def _compute_peer_spillover(adj, denied, active, *, threshold=0.6):
#     """Backward-compatible peer spillover based on neighbor denial rate.

#     This is the previous definition kept in case other code depends on it.
#     The opt-out generator now uses `_compute_observed_rejection_gap` instead.
#     """
#     adj = np.asarray(adj)
#     denied = np.asarray(denied).astype(int)
#     active = np.asarray(active).astype(int)

#     denied_eff = denied.copy()
#     denied_eff[active == 0] = 0

#     deg = adj.sum(axis=1)
#     deg_safe = np.where(deg == 0, 1, deg)
#     neighbor_denial_rate = (adj @ denied_eff) / deg_safe
#     return (neighbor_denial_rate >= threshold).astype(int)


def gen_temporal_graph_with_optout(
    model,
    agent,
    steps,
    *,
    graph_seed=2021,
    seed=2021,
    spillover_threshold=0.6,
    enforce_demographic_mixing=True,
    k_same=6,
    k_other=4,
    decision_coef=0.8,
    repayment_coef=0.8,
    decision_positive_is_approval=True,
    exclude_inactive_neighbors_from_O_denominator=False,
):
    """Generate a temporal graph with opt-out/attrition dynamics.

    Implements the rule table you provided with variables:
    - D_{i,t} in {0,1} (1=deny,    0=approve)
    - Y_{i,t} in {0,1} (1=default, 0=repay)
      - P_{i,t} in {0,1} (previous denial indicator)
    - O_{i,t} in {0,1} (observed peer outcome disparity signal)
      - U_{i,t} in {0,1} (perceived unfairness)
      - A_{i,t} in {0,1} (active; 0=opted out). Individuals remain in the graph.

    Also enforces: opted-out is treated as D_t=0 when constructing O_t.

    Returns:
        s: (n,)
        adj: (n,n) adjacency (static)
        edges: edge list
        Xs: list of X_t, length=steps
        Ds: list of D_t (denied), length=steps
        Ys: list of Y_t (fraud/default), length=steps
        Ps: list of P_t, length=steps
        Os: list of O_t, length=steps
        Us: list of U_t, length=steps
        As: list of A_t (active), length=steps
    """
    rng = np.random.default_rng(seed)

    # Initial profiles
    s, X = agent.gen_init_profile()
    n = len(s)

    # Static graph
    if enforce_demographic_mixing:
        adj, edges = gen_static_demographic_regular_graph(s, k_same=k_same, k_other=k_other, seed=graph_seed)
    else:
        adj, edges = gen_static_random_graph(n, k=10, seed=graph_seed, directed=False)

    # Time series containers
    Xs = [X]
    Ds, Ys, Ps, Os, Us, As = [], [], [], [], [], []

    active = np.ones(n, dtype=int)  # A_{t=1}
    prev_denied = np.zeros(n, dtype=int)  # P_{t=1}

    # Helper to sample approval probability from model.predict
    def _sample_approval_prob(s_now, X_now):
        _, p = model.predict(s_now, X_now)
        p = np.asarray(p, dtype=float)
        if decision_positive_is_approval:
            return p
        return 1.0 - p

    for _t in range(steps):
        X_t = Xs[-1]

        # Sample decisions/outcomes for active applicants
        p_approve = _sample_approval_prob(s, X_t)
        approve = sampling(p_approve, values=[0.0, 1.0], coef=decision_coef).astype(int)
        approve[active == 0] = 0
        denied = (1 - approve).astype(int)
        denied[active == 0] = 0  # opted-out treated as not denied

        # Bank().predict returns P(repay=1). We need Y=1 to mean default.
        _, p_repay = Bank().predict(s, X_t)
        defaulted = sampling(p_repay, values=[1.0, 0.0], coef=repayment_coef).astype(int)
        defaulted[active == 0] = 0

        P_t = prev_denied.copy()
        # Observed peer outcomes: compare observed rejection rate of my group vs other group
        # Note: spillover_threshold is kept for API compatibility but is unused under this definition.
        O_t = _compute_observed_rejection_gap(
            adj,
            denied,
            s,
            active=active,
            exclude_inactive_from_denominator=exclude_inactive_neighbors_from_O_denominator,
        )

        # Perceived unfairness U_t
        repeated_denial_no_default = (denied == 1) & (P_t == 1) & (defaulted == 0)
        peer_spillover = (denied == 1) & (O_t == 1)
        U_t = ((repeated_denial_no_default | peer_spillover) & (active == 1)).astype(int)

        # Attrition / opt-out: once opted out, remain inactive
        correct_default_detection = (denied == 1) & (defaulted == 1)
        opt_out_now = ((U_t == 1) | correct_default_detection) & (active == 1)
        active_next = active.copy()
        active_next[opt_out_now] = 0

        # Feature update (only for active applicants at time t)
        base = np.array([[agent.base[int(i)]] for i in s], dtype=float)
        theta = np.asarray(model.params[1:-1], dtype=float)

        # +eps if repaid (Y=0), -eps if defaulted (Y=1)
        repay_sign = (1.0 - 2.0 * defaulted).reshape(-1, 1)
        delta = agent.eps * theta.reshape(1, -1)
        delta = np.repeat(delta, repeats=n, axis=0) * repay_sign

        apply_change = ((active == 1) & (approve == 1)).astype(float).reshape(-1, 1)
        base_tile = np.tile(base, (1, X_t.shape[1]))
        X_next = X_t + (active.reshape(-1, 1) * base_tile) + apply_change * delta

        # Record and advance
        Ds.append(denied)
        Ys.append(defaulted)
        Ps.append(P_t)
        Os.append(O_t)
        Us.append(U_t)
        As.append(active)

        prev_denied = denied.copy()
        active = active_next
        Xs.append(X_next)

    # Xs is length steps+1; return X_1..X_steps to match other APIs
    return s, adj, edges, Xs[:steps], Ds, Ys, Ps, Os, Us, As


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

    where D1..Dk are the decisions of i's neighbors at time t.

    Args:
        csv_path: output file path
        s: (n,) sensitive attribute
        Xs: list of (n,2) feature matrices (length T)
        adj: (n,n) adjacency matrix (static)
        Ds, Ys: lists of (n,) arrays (length T)
        Ps, Us, As, Os: optional lists of (n,) arrays (length T)
        t0: starting index for t in the output file
        neighbor_k: number of neighbor decision entries to write (default 10)
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