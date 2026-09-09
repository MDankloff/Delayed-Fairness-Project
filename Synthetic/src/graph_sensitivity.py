""" Graph sensitivity analysis of survivorship bias 

Original 'graph.py' builds one kind of network: every person gets the same fixed number of same-group and other-group connections. 
two topologies: 
- gen_static_random_graph: erdos-renyi-style random graph (uniform mixing) 
- gen_static_demographic_regular_graph: an extact k_same/k_other regular graph (every node gets the same k_same ingroup and k_other outgroup edges)

this module adds:
- graph_er_graph: Erdos-Renyi (1959) / configuration model control condition (no demographic awareness). Connections are placed at random, with no regard for group membership.
Used as a no-strucutre baseline.
- gen_ws_block_graph: Watts-Strogatz (1998) small-world graph (demographic aware): a ring lattice within each group (rewired with probability p) plus a random matching across groups.
Applicants are mostly connected to their immediate cicle, with a few random long-distance connections mixed in, similar to real social networks.
- gen_demographic_regular_graph_robust: more reliable version of the original network model used throughout this project.
- graph_diagnostics: reports basic properties of any network produced (average nr of connections, how clustered the network is, how much people mix across groups), results can then be explained in terms of network structure rather than
just the input settings. 

Every function returns the network as an adjacency matrix (adj), in the same format used for ealier synthethic LTF pop project
"""

import numpy as np
import networkx as nx

def _A_E_from_nx(G, n):
    A = np.zeros((n, n), dtype=np.int8)
    E = set()
    for u, v in G.edges():
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        E.add((a, b))
        A[a, b] = 1
        A[b, a] = 1
    return A, sorted(E)

def gen_er_graph(n_nodes, k=10, *, seed=2026):
    """Erdos-Renyi-style random network: connections are placed without any regard for group membership (no demographic awareness). 
    All applicants get the same nr of connections (k), matching the other network models for fair comparison, but connections themselves are random.
    Used as a no group-based bias baseline
    """
    seed = int(seed)
    k = int(min(k, n_nodes - 1))
    if k <= 0:
        G = nx.empty_graph(n_nodes)
    elif (n_nodes * k) % 2 == 0:
        G = nx.random_regular_graph(k, n_nodes, seed=seed)
    else:
        G = nx.gnm_random_graph(n_nodes, n_nodes * k // 2, seed=seed)
    return _A_E_from_nx(G, n_nodes)

def gen_ws_block_graph(s, *, k_same=8, k_other=2, rewire_p=0.1, seed=2021):
    """Watts-Strogatz small-world network, adapted to group membership.
 
    Within each group: applicants are arranged in a circle and connected to their nearest neighbors on that circle, then a few of those
    connections are randomly rewired to someone further away (controlled by rewire_p: 0 = no rewiring at all, tightly clustered circles; 1 = fully rewired, effectively random). 
    Produces the "small-world" pattern seen in real social networks: mostly local connections, with a few long-distance shortcuts.
    Across groups: a fixed number of random connections (k_other), same as the main network model used elsewhere in this project.
 
    Varying rewire_p lets us test whether how clustered/local a network is (separately from its size or how much it mixes groups) affects the results.
    """
    s = np.asarray(s).astype(int)
    n = len(s)
    idx0 = np.where(s == 0)[0]
    idx1 = np.where(s == 1)[0]
    rng = np.random.default_rng(int(seed))
 
    def _ws_subgraph(nodes, k, p, local_seed):
        nodes = np.asarray(nodes)
        m = nodes.size
        if k <= 0 or m < 3:
            return set()
        k_eff = max(2, k - (k % 2))  # watts_strogatz_graph needs even k
        k_eff = min(k_eff, m - 1 if (m - 1) % 2 == 0 else m - 2)
        G = nx.watts_strogatz_graph(m, k_eff, p, seed=int(local_seed))
        edges = set()
        for u, v in G.edges():
            a, b = int(nodes[u]), int(nodes[v])
            edges.add((a, b) if a < b else (b, a))
        return edges
 
    def _bipartite_regular(left, right, degree, max_tries=500):
        left = np.asarray(left)
        right = np.asarray(right)
        if degree <= 0:
            return set()
        for _ in range(max_tries):
            l_stubs = np.repeat(left, degree)
            r_stubs = np.repeat(right, degree)
            rng.shuffle(l_stubs)
            rng.shuffle(r_stubs)
            edges = set()
            ok = True
            for u, v in zip(l_stubs, r_stubs):
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in edges:
                    ok = False
                    break
                edges.add((a, b))
            if ok:
                return edges
        raise RuntimeError("Failed to build cross-group regular bipartite graph")
 
    edges = set()
    edges |= _ws_subgraph(idx0, k_same, rewire_p, seed + 1)
    edges |= _ws_subgraph(idx1, k_same, rewire_p, seed + 2)
    edges |= _bipartite_regular(idx0, idx1, k_other)
 
    A = np.zeros((n, n), dtype=np.int8)
    for a, b in edges:
        A[a, b] = 1
        A[b, a] = 1
    return A, sorted(edges)

def _circulant_bipartite_regular(left, right, degree, rng):
    """Builds connections between two equal-sized groups so that all applicants get exactly the same number of cross-group connections (`degree`)
    Used internally by gen_demographic_regular_graph_robust.
    """
    left = np.asarray(left)
    right = np.asarray(right)
    m = left.size
    assert right.size == m and 0 <= degree <= m
 
    offsets = rng.choice(np.arange(m), size=degree, replace=False)
    G = nx.Graph()
    G.add_nodes_from([("L", i) for i in range(m)])
    G.add_nodes_from([("R", i) for i in range(m)])
    for i in range(m):
        for off in offsets:
            j = int((i + off) % m)
            G.add_edge(("L", i), ("R", j))
 
    # Randomize while preserving the bipartite degree sequence.
    n_edges = G.number_of_edges()
    try:
        nx.double_edge_swap(G, nswap=max(1, n_edges * 3), max_tries=n_edges * 50,
                             seed=int(rng.integers(0, 2**31 - 1)))
    except nx.NetworkXAlgorithmError:
        pass  # ran out of swap attempts near the end; graph is still valid & regular
 
    edges = set()
    for u, v in G.edges():
        a = int(left[u[1]]) if u[0] == "L" else int(right[u[1]])
        b = int(left[v[1]]) if v[0] == "L" else int(right[v[1]])
        edges.add((a, b) if a < b else (b, a))
    return edges
 
 
def gen_demographic_regular_graph_robust(s, *, k_same=8, k_other=2, seed=2021):
    """Same network model used throughout this project, the original sometimes failed to build networks when applicants had a high number of connections. 
    Requires the two groups to be the same size.
    """
    s = np.asarray(s).astype(int)
    n = len(s)
    idx0 = np.where(s == 0)[0]
    idx1 = np.where(s == 1)[0]
    if idx0.size != idx1.size and k_other != 0:
        raise ValueError("Exact regular cross-group degree requires equal group sizes")
 
    rng = np.random.default_rng(int(seed))
 
    def _intra(nodes, degree, local_seed):
        nodes = np.asarray(nodes)
        if degree <= 0:
            return set()
        G = nx.random_regular_graph(degree, nodes.size, seed=int(local_seed))
        edges = set()
        for u, v in G.edges():
            a, b = int(nodes[u]), int(nodes[v])
            edges.add((a, b) if a < b else (b, a))
        return edges
 
    edges = set()
    edges |= _intra(idx0, k_same, seed + 1)
    edges |= _intra(idx1, k_same, seed + 2)
    if k_other > 0:
        edges |= _circulant_bipartite_regular(idx0, idx1, k_other, rng)
 
    A = np.zeros((n, n), dtype=np.int8)
    for a, b in edges:
        A[a, b] = 1
        A[b, a] = 1
    np.fill_diagonal(A, 0)
    return A, sorted(edges)
 
 
def graph_diagnostics(A, s=None):
    """Basic properties of a network to describe results in terms of network structure.
    Returns: nr of applicants and connections, average number of connections per applicant, how much that varies between applicants, how clustered the network is, 
    what fraction of applicants are "reachable" from one another, and average "distance" between two applicants. 
    
    If group membership (s) is given, also reports how much applicants mix across groups versus staying within their
    own group (negative = applicants mostly connect within their own group).
    """
    A = np.asarray(A)
    n = A.shape[0]
    G = nx.from_numpy_array(A)
    deg = np.array([d for _, d in G.degree()], dtype=float)
    out = {
        "n": n,
        "m": int(G.number_of_edges()),
        "mean_degree": float(deg.mean()) if n else float("nan"),
        "degree_cv": float(deg.std() / deg.mean()) if deg.mean() > 0 else 0.0,
        "clustering": float(nx.average_clustering(G)) if n else float("nan"),
    }
    if G.number_of_edges() == 0:
        out["largest_cc_frac"] = 0.0
        out["avg_shortest_path"] = float("nan")
    else:
        comps = list(nx.connected_components(G))
        giant = max(comps, key=len)
        out["largest_cc_frac"] = len(giant) / n
        if len(giant) <= 2000:
            Gs = G.subgraph(giant)
            out["avg_shortest_path"] = float(nx.average_shortest_path_length(Gs))
        else:
            out["avg_shortest_path"] = float("nan")  # too slow, skip
 
    if s is not None:
        s = np.asarray(s).astype(int)
        same = 0
        diff = 0
        for u, v in G.edges():
            if s[u] == s[v]:
                same += 1
            else:
                diff += 1
        total = same + diff
        out["ei_homophily_index"] = float((diff - same) / total) if total else float("nan")
        out["frac_cross_group_edges"] = float(diff / total) if total else float("nan")
    return out

