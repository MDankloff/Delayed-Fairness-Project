import numpy as np
import pandas as pd
from utils import *  # expects: sigmoid, sampling, etc.

# ----------------------------
# Your classes (unchanged)
# ----------------------------

class Bank:
    name = "Bank"
    params = np.array([2.5, 2, -1, -4.0])

    def __init__(self, params=None, seed=2021):
        self.seed = seed
        if params is not None:
            self.params = np.asarray(params, dtype=float)

    def predict(self, s, X):
        Xs = np.c_[s, X, np.ones(len(s))]
        p = sigmoid(Xs @ self.params / 3.0)
        y = (p >= 0.5).astype(int)  # 1=DENY, 0=APPROVE
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
        np.random.seed(self.seed)

        def gen_gaussian(mean, cov, sen_label, sample_size):
            s = np.ones(sample_size, dtype=float) * sen_label
            X = np.random.multivariate_normal(mean=mean, cov=cov, size=sample_size)
            return s, X

        n_protects = int(self.protect_ratio * self.n_samples)

        mu0, sigma0 = [-2, -2], [[10, 1], [1, 5]]
        mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]

        s0, X0 = gen_gaussian(mu0, sigma0, 0, n_protects)
        s1, X1 = gen_gaussian(mu1, sigma1, 1, self.n_samples - n_protects)

        s = np.hstack((s0, s1))
        X = np.vstack((X0, X1))

        perm = np.arange(self.n_samples)
        np.random.shuffle(perm)
        s = s[perm]
        X = X[perm]
        return s.astype(int), X

    def gen_next_profile(self, s, X, model):
        base = np.array([[self.base[int(i)]] for i in s], dtype=float)

        _, prob = model.predict(s, X)
        sample_y = sampling(prob, coef=0.8)

        _, def_prob = Bank().predict(s, X)
        default = sampling(def_prob, values=[-1, 1.0], coef=0.8)

        change = self.eps * model.params[1:-1] * prob.reshape(-1, 1)
        default_change = change * default.reshape(-1, 1)
        X_next = X + sample_y.reshape(-1, 1) * default_change + np.tile(base, (1, X.shape[1]))
        return X_next


# ----------------------------
# Dynamics helpers
# ----------------------------

def _peer_signal_vector(D, A, alpha, n_peers, rng):
    n = len(D)
    O = np.zeros(n, dtype=int)

    observable = np.where(A == 1)[0]  # peers who applied

    for i in range(n):
        candidates = observable[observable != i]
        if candidates.size == 0:
            O[i] = 0
            continue

        k = min(n_peers, candidates.size)
        sampled = rng.choice(candidates, size=k, replace=False)
        frac_denied = float((D[sampled] == 1).mean()) if k > 0 else 0.0
        O[i] = int(frac_denied >= alpha)

    return O


def step_applicants(
    bank: Bank,
    dynamics: Agent,
    S, X, Y, P, U, A,
    alpha=0.6,
    n_peers=20,
    seed=2021,
):
    rng = np.random.default_rng(seed)

    n = len(S)
    S = np.asarray(S, dtype=int)
    Y = np.asarray(Y, dtype=int)  # fixed latent label
    P = np.asarray(P, dtype=int)
    U = np.asarray(U, dtype=int)
    A = np.asarray(A, dtype=int)
    X = np.asarray(X, dtype=float)

    # Active agents (still participating)
    active_idx = np.where(A == 1)[0]

    # 1) Bank decision only for active agents
    D = np.full(n, -1, dtype=int)
    if active_idx.size > 0:
        D_active, _ = bank.predict(S[active_idx], X[active_idx])
        D[active_idx] = D_active.astype(int)

    # 2) Peer signal based on active agents' decisions
    O = _peer_signal_vector(D=D, A=A, alpha=alpha, n_peers=n_peers, rng=rng)

    # 3) Perceived unfairness for active agents
    U_new = np.zeros(n, dtype=int)
    active = (A == 1) & (D != -1)

    U_new[active] = (((D[active] == 1) & (P[active] == 1)) |
                     ((D[active] == 1) & (O[active] == 1))).astype(int)

    # 4) Participation rule => A_next
    A_next = A.copy()  # opted-out stays opted-out

    approved = active & (D == 0)
    denied = active & (D == 1)

    A_next[approved] = 1
    A_next[denied & (U_new == 1)] = 0
    A_next[denied & (U_new == 0) & (Y == 1)] = 0
    A_next[denied & (U_new == 0) & (Y == 0)] = 1

    # 5) Update P for active agents
    P_next = P.copy()
    P_next[active] = D[active]

    # 6) Feature update ONLY for active agents (opted-out disappear from bank inputs)
    X_next = X.copy()
    if active_idx.size > 0:
        X_active_next = dynamics.gen_next_profile(S[active_idx], X[active_idx], bank)
        X_next[active_idx] = X_active_next

    # 7) Commit U for active agents
    U_next = U.copy()
    U_next[active] = U_new[active]

    return X_next, P_next, U_next, A_next, D, O


# ----------------------------
# Pretty printing: "agent vector" per step
# ----------------------------

def print_step_state(t, S, X, Y, P, U, A, D=None, O=None, max_agents=10, x_decimals=2):
    """
    Prints each agent's vector:
      <S, X, Y, P, U, A> plus (optional) observed <D, O> at step t.
    """
    n = len(S)
    m = min(n, max_agents)

    header = "i | S | Y | P | U | A"
    if D is not None:
        header += " | D"
    if O is not None:
        header += " | O"
    header += " | X"
    print(f"\n=== step {t} ===")
    print(header)
    print("-" * len(header))

    for i in range(m):
        x_str = np.array2string(X[i], precision=x_decimals, floatmode="fixed")
        row = f"{i:>2d} | {S[i]:>1d} | {Y[i]:>1d} | {P[i]:>1d} | {U[i]:>1d} | {A[i]:>1d}"
        if D is not None:
            dval = D[i]
            row += f" | {dval if dval != -1 else '.':>1}"
        if O is not None:
            row += f" | {O[i]:>1d}"
        row += f" | {x_str}"
        print(row)

    if n > m:
        print(f"... ({n-m} more agents not shown; set max_agents to print all)")


# ----------------------------
# Simulation: run rollout 
# ----------------------------

def simulate_gen(
    steps=5,
    n_samples=8,
    protect_ratio=0.5,
    eps=0.1,
    base=(0.0, 0.0),
    alpha=0.6,
    n_peers=5,
    seed=2021,
    print_all_agents=True,
    save_csv=True,
    csv_path="simulation_results.csv",
    return_df=False,
):
    # init models
    bank = Bank(seed=seed)
    dynamics = Agent(
        n_samples=n_samples,
        protect_ratio=protect_ratio,
        eps=eps,
        base=base,
        seed=seed,
    )

    # initial state
    np.random.seed(seed)
    S, X = dynamics.gen_init_profile()

    # Sample fixed Y once (latent intent)
    _, prob0 = bank.predict(S, X)
    Y = sampling(prob0, coef=0.8).astype(int)

    n = len(S)
    P = np.zeros(n, dtype=int)
    U = np.zeros(n, dtype=int)
    A = np.ones(n, dtype=int)  # everyone applies at t=0

    max_agents = n if print_all_agents else 10

    # Print initial state (before any decisions)
    print_step_state(
        t=0,
        S=S, X=X, Y=Y, P=P, U=U, A=A,
        D=None, O=None,
        max_agents=max_agents,
    )

    # Keep the original rollout outputs for downstream code (e.g., PF-syn.ipynb)
    Xs = [X.copy()]
    Ys = [Y.copy()]

    # Collect data for DataFrame
    data_records = []
    
    # Store initial state for active agents (A=1 at t=0)
    active_at_t0 = np.where(A == 1)[0]
    for i in active_at_t0:
        x_flat = list(X[i])  # flatten features
        data_records.append({
            't': 0,
            'i': i,
            'S': int(S[i]),
            'Y': int(Y[i]),
            'P': int(P[i]),
            'U': int(U[i]),
            'A': int(A[i]),
            'D': None,
            'O': None,
            **{f'X{j}': float(x_flat[j]) for j in range(len(x_flat))}
        })

    # Run steps; at each step t we produce D_t, O_t and update to next state
    for t in range(steps):
        X, P, U, A, D, O = step_applicants(
            bank=bank,
            dynamics=dynamics,
            S=S, X=X, Y=Y, P=P, U=U, A=A,
            alpha=alpha,
            n_peers=n_peers,
            seed=seed + t + 1,
        )

        # Update Y: Y_{t+1} = D_t
        Y = D.copy()

        Xs.append(X.copy())
        Ys.append(Y.copy())

        # Print state AFTER processing step t (i.e., state at t+1, with D_t and O_t shown)
        print_step_state(
            t=t + 1,
            S=S, X=X, Y=Y, P=P, U=U, A=A,
            D=D, O=O,
            max_agents=max_agents,
        )
        
        # Store data for active agents only (A_{t+1} == 1)
        # If A_t = 0, don't store data from t+1 onwards
        active_agents = np.where(A == 1)[0]
        for i in active_agents:
            x_flat = list(X[i])
            data_records.append({
                't': t + 1,
                'i': i,
                'S': int(S[i]),
                'Y': int(Y[i]),
                'P': int(P[i]),
                'U': int(U[i]),
                'A': int(A[i]),
                'D': int(D[i]) if D[i] != -1 else None,
                'O': int(O[i]),
                **{f'X{j}': float(x_flat[j]) for j in range(len(x_flat))}
            })

    # Create DataFrame
    df = pd.DataFrame(data_records)
    
    # Save to CSV
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"\nData saved to {csv_path} ({len(df)} records)")

    if return_df:
        return S, Xs, Ys, df

    return S, Xs, Ys


# ----------------------------
# Example run
# ----------------------------
if __name__ == "__main__":
    _ = simulate_gen(
        steps=5,
        n_samples=3,
        protect_ratio=0.4,
        eps=0.15,
        base=(0.05, 0.00),   # base shift per group
        alpha=0.6,
        n_peers=3,
        seed=2021,
        print_all_agents=True,  # prints all agents each step
    )
