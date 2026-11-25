import networkx as nx


def node_name(graph: nx.DiGraph, node: str) -> str:
    return graph.nodes[node]["name"]


def describe_edge(graph: nx.DiGraph, source: str, target: str) -> str:
    kind = graph.edges[source, target]["type"]
    return f"{node_name(graph, source)} -{kind}-> {node_name(graph, target)}"
