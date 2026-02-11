import numpy as np

def gen_static_random_graph(n_nodes, k=10, *, seed=2021, directed=False):
    """Generate a static random graph with per-node degree ~k.

    Each node randomly selects `k` distinct other nodes.
    - If `directed=False`, the adjacency is symmetrized (undirected), so some
      nodes may end up with degree > k due to incoming selections.
    - Self-loops are not created.

    Returns:
        A: (n_nodes, n_nodes) adjacency matrix in {0,1}
        E: edge list; for undirected graphs edges are unique pairs (i, j) with i<j
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if k < 0:
        raise ValueError("k must be non-negative")

    k = int(min(k, n_nodes - 1))
    rng = np.random.default_rng(seed)

    A = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    candidates = np.arange(n_nodes)
    for i in range(n_nodes):
        others = np.delete(candidates, i)
        if k == 0:
            continue
        nbrs = rng.choice(others, size=k, replace=False)
        A[i, nbrs] = 1

    if not directed:
        A = np.maximum(A, A.T)

    np.fill_diagonal(A, 0)

    if directed:
        src, dst = np.where(A == 1)
        E = list(zip(src.tolist(), dst.tolist()))
    else:
        src, dst = np.where(np.triu(A, 1) == 1)
        E = list(zip(src.tolist(), dst.tolist()))

    return A, E


def gen_static_demographic_regular_graph(s, *, k_same=6, k_other=4, seed=2021, max_tries=2000):
    """Generate an undirected graph where every node has fixed demographic mixing.

    For each node v:
      - it has exactly `k_same` neighbors with the same sensitive attribute S
      - it has exactly `k_other` neighbors with the other demographic group

    This implies the overall graph is (k_same + k_other)-regular and undirected.

    Notes:
      - Requires exactly two demographic groups encoded as {0,1}.
      - For exact regular cross-group degree on both sides, group sizes must match.

    Returns:
        A: (n,n) adjacency matrix in {0,1}
        E: undirected edge list as (i,j) with i<j
    """
    s = np.asarray(s).astype(int)
    if s.ndim != 1:
        raise ValueError("s must be a 1D array")

    n = len(s)
    if n == 0:
        raise ValueError("s must be non-empty")

    k_same = int(k_same)
    k_other = int(k_other)
    if k_same < 0 or k_other < 0:
        raise ValueError("k_same and k_other must be non-negative")

    idx0 = np.where(s == 0)[0]
    idx1 = np.where(s == 1)[0]
    if idx0.size + idx1.size != n:
        raise ValueError("s must only contain 0/1")

    # Feasibility checks
    if k_same >= idx0.size or k_same >= idx1.size:
        raise ValueError("k_same must be < group size for both groups")
    if k_other > idx0.size or k_other > idx1.size:
        raise ValueError("k_other must be <= opposite group size")
    if (idx0.size * k_same) % 2 != 0 or (idx1.size * k_same) % 2 != 0:
        raise ValueError("group_size * k_same must be even for simple undirected graphs")
    if idx0.size != idx1.size and k_other != 0:
        raise ValueError("Exact regular cross-group degree requires equal group sizes")

    rng = np.random.default_rng(seed)

    def _random_simple_regular_edges(nodes, degree, *, local_seed):
        """Simple d-regular graph on the given node IDs."""
        if degree == 0:
            return set()
        nodes = np.asarray(nodes)
        try:
            import networkx as nx

            G = nx.random_regular_graph(degree, nodes.size, seed=int(local_seed))
            edges = set()
            for u, v in G.edges():
                a = int(nodes[u])
                b = int(nodes[v])
                edges.add((a, b) if a < b else (b, a))
            return edges
        except Exception:
            # Fallback: configuration model with rejection (may be slower)
            for _ in range(max_tries):
                stubs = np.repeat(nodes, degree)
                rng.shuffle(stubs)
                edges = set()
                ok = True
                for u, v in stubs.reshape(-1, 2):
                    if u == v:
                        ok = False
                        break
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) in edges:
                        ok = False
                        break
                    edges.add((a, b))
                if ok and len(edges) == (len(stubs) // 2):
                    return edges
            raise RuntimeError("Failed to generate a simple regular graph; increase max_tries")

    def _random_simple_regular_bipartite_edges(left_nodes, right_nodes, degree):
        if degree == 0:
            return set()
        left_nodes = np.asarray(left_nodes)
        right_nodes = np.asarray(right_nodes)
        if left_nodes.size * degree != right_nodes.size * degree:
            raise ValueError("Total bipartite stubs must match")
        for _ in range(max_tries):
            left_stubs = np.repeat(left_nodes, degree)
            right_stubs = np.repeat(right_nodes, degree)
            rng.shuffle(left_stubs)
            rng.shuffle(right_stubs)
            edges = set()
            ok = True
            for u, v in zip(left_stubs, right_stubs):
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in edges:
                    ok = False
                    break
                edges.add((a, b))
            if ok and len(edges) == len(left_stubs):
                return edges
        raise RuntimeError("Failed to generate a simple regular bipartite graph; increase max_tries")

    intra0 = _random_simple_regular_edges(idx0, k_same, local_seed=seed + 17)
    intra1 = _random_simple_regular_edges(idx1, k_same, local_seed=seed + 29)
    cross = _random_simple_regular_bipartite_edges(idx0, idx1, k_other)

    E = set()
    E |= intra0
    E |= intra1
    E |= cross

    A = np.zeros((n, n), dtype=np.int8)
    for u, v in E:
        A[u, v] = 1
        A[v, u] = 1
    np.fill_diagonal(A, 0)

    expected_degree = k_same + k_other
    deg = A.sum(axis=1)
    if not np.all(deg == expected_degree):
        raise RuntimeError("Generated graph does not satisfy exact degree constraints")

    # Return a stable list order
    E_list = sorted(E)
    return A, E_list


def build_temporal_graph_features(s, Xs, *, include_sensitive=False):
    """Build a sequence of node feature matrices X_t from profiles.

    Args:
        s: (n,) sensitive attribute per node
        Xs: list of (n, d) feature matrices per timestamp
        include_sensitive: if True, prepend s as the first feature column
    """
    if not include_sensitive:
        return list(Xs)
    s_col = np.asarray(s).reshape(-1, 1)
    return [np.concatenate([s_col, X], axis=1) for X in Xs]



