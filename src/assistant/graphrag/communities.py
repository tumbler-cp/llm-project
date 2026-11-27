"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.graphrag.facts import describe_edge, node_name
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

SEED = 42
MAX_WORDS = 150
MAX_LINES = 60
SYSTEM = (
    "Ты пишешь сводку фрагмента графа знаний компании «ПолярТех». "
    "Опиши, какие сущности в него входят, как они связаны и что встречается чаще всего. "
    f"Пиши только по данным ниже, числа переноси без изменений. Не больше {MAX_WORDS} слов."
)


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    parts = nx.community.louvain_communities(graph.to_undirected(), weight="weight", seed=SEED)
    return [Community(index, nodes) for index, nodes in enumerate(sorted(sorted(part) for part in parts))]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    edges = sorted(graph.subgraph(community.nodes).edges(data="weight"), key=lambda edge: -edge[2])
    totals = Counter()
    for source, target, weight in edges:
        totals[graph.edges[source, target]["type"], target] += weight
    entities = ", ".join(node_name(graph, node) for node in community.nodes[:MAX_LINES])
    facts = "\n".join(
        f"{describe_edge(graph, source, target)} (вес {weight})" for source, target, weight in edges[:MAX_LINES]
    )
    counts = "\n".join(
        f"{node_name(graph, target)}: {kind} всего {weight}" for (kind, target), weight in totals.most_common(MAX_LINES)
    )
    prompt = f"Сущности: {entities}\n\nСвязи с весами:\n{facts}\n\nИтоги по весам:\n{counts}"
    reply = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    return " ".join(reply.content.split()[:MAX_WORDS])
