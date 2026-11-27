"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult, Relation


def _add_source(sources: list[str], chunk_id: str) -> None:
    if chunk_id not in sources:
        sources.append(chunk_id)


def _add_node(graph: nx.DiGraph, entity: Entity, chunk_id: str) -> None:
    if entity.node_id not in graph:
        graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[])
    _add_source(graph.nodes[entity.node_id]["sources"], chunk_id)


def _add_edge(graph: nx.DiGraph, relation: Relation, chunk_id: str) -> None:
    source, target = relation.subject.node_id, relation.object.node_id
    for entity in (relation.subject, relation.object):
        _add_node(graph, entity, chunk_id)
    if not graph.has_edge(source, target):
        graph.add_edge(source, target, type=relation.type, sources=[])
    edge = graph.edges[source, target]
    _add_source(edge["sources"], chunk_id)
    edge["weight"] = len(edge["sources"])


def _resolve_replacement_conflicts(graph: nx.DiGraph) -> int:
    pairs = [
        (old, new)
        for old, new, kind in graph.edges(data="type")
        if kind == "REPLACED_BY" and old < new and graph.has_edge(new, old)
    ]
    for old, new in pairs:
        forward, backward = graph.edges[old, new]["weight"], graph.edges[new, old]["weight"]
        if forward <= backward:
            graph.remove_edge(old, new)
        if backward <= forward:
            graph.remove_edge(new, old)
    return len(pairs)


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph(conflicts=0)
    for result in results:
        for entity in result.entities:
            _add_node(graph, entity, result.chunk_id)
        for relation in result.relations:
            if relation.matches_schema() and relation.subject != relation.object:
                _add_edge(graph, relation, result.chunk_id)
    graph.graph["conflicts"] = _resolve_replacement_conflicts(graph)
    return graph
