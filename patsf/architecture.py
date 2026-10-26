"""DOWNSTREAM MEASURES OF AN INVENTION'S ARCHITECTURE
Generates the following measures of an invention's architecture, based on its decomposition into components and functionalities:

  components                 parts (every component but C0)
  functionalities            functionalities
  links                      pairs of parts joined by a functionality
  depth                      levels of the part-of tree below C0 (a part directly under C0 is 1)
  branching                  mean number of direct parts of each component that has parts
  link_density               links / possible pairs of parts
  unlinked_share             share of parts in no link
  disconnected_groups        groups of linked parts with no link between groups (connected
                             components of the link network; unlinked parts are not counted)
  modularity                 Newman modularity of the clusters found by greedy modularity
                             maximization (networkx) on the linked parts (higher: fewer links
                             between clusters)
  degree_centralization      Freeman degree centralization (1: one hub linked to every other part)
  hub                        the part in the most links
  functionalities_per_component  functionalities performed by parts / parts

Measures that need something to count (pairs, links) are empty when there is nothing to count.
"""

from __future__ import annotations

import networkx as nx

from .compare import ROOT, links

MEASURES = ["components", "functionalities", "links", "depth", "branching", "link_density", "unlinked_share",
            "disconnected_groups", "modularity", "degree_centralization", "hub", "functionalities_per_component"]


def measures(rep: dict) -> dict:
    parts = [c for c in rep.get("components", []) if c["id"] != ROOT]
    ids = {c["id"] for c in parts}
    funcs = rep.get("functionalities", [])
    pairs = [tuple(sorted(p)) for p in links(rep) if p <= ids]
    n = len(parts)

    # hierarchy
    parent = {c["id"]: c.get("parent") for c in rep.get("components", [])}
    def level(x, seen=()):
        p = parent.get(x)
        return 1 if not p or p == ROOT or p not in parent or x in seen else 1 + level(p, seen + (x,))
    children: dict[str, int] = {}
    for c in rep.get("components", []):
        if c.get("parent"):
            children[c["parent"]] = children.get(c["parent"], 0) + 1

    # the link network
    g = nx.Graph()
    g.add_nodes_from(ids)
    g.add_edges_from(pairs)
    linked = g.subgraph([x for x in g if g.degree(x) > 0])
    if linked.number_of_edges():
        clusters = nx.community.greedy_modularity_communities(linked)
        disconnected_groups = nx.number_connected_components(linked)
        modularity = round(nx.community.modularity(linked, clusters), 3)
    else:
        disconnected_groups, modularity = None, None
    deg = dict(g.degree())
    if n >= 3 and pairs:
        top = max(deg.values())
        centralization = round(sum(top - d for d in deg.values()) / ((n - 1) * (n - 2)), 3)
    else:
        centralization = None
    names = {c["id"]: c["name"] for c in parts}
    hub = max(sorted(deg), key=lambda x: deg[x]) if pairs else None

    possible = n * (n - 1) / 2
    return {
        "components": n,
        "functionalities": len(funcs),
        "links": len(pairs),
        "depth": max((level(c["id"]) for c in parts), default=0),
        "branching": round(sum(children.values()) / len(children), 3) if children else None,
        "link_density": round(len(pairs) / possible, 3) if possible else None,
        "unlinked_share": round(sum(deg[x] == 0 for x in ids) / n, 3) if n else None,
        "disconnected_groups": disconnected_groups,
        "modularity": modularity,
        "degree_centralization": centralization,
        "hub": names[hub] if hub else None,
        "functionalities_per_component": round(sum(f["component"] in ids for f in funcs) / n, 3) if n else None,
    }
