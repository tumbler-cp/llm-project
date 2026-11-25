"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

from itertools import combinations, pairwise

import networkx as nx

from assistant.graphrag.facts import describe_edge
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Answer, KnowledgeBase

NOT_FOUND = "в базе знаний не найдено"
PATH_LENGTH = 3
MAX_FACTS_CHARS = 8000
LOCAL_SYSTEM = (
    "Ты ассистент сервисной службы компании «ПолярТех». Отвечай на вопрос только по пронумерованным фактам ниже. "
    "Факт «A -СВЯЗЬ-> B» означает, что связь направлена от A к B. "
    "Коды моделей, деталей, ошибок и регламентов пиши точно как в фактах. "
    f"Если ответа в фактах нет, напиши: «{NOT_FOUND}»."
)


def find_anchors(question: str, kb: KnowledgeBase) -> list[str] | None:
    codes = [f"{code_type(code)}:{code}" for code in find_codes(question)]
    if any(node not in kb.graph for node in codes):
        return None
    names = [f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary)]
    return codes + [node for node in names if node in kb.graph]


def path_edges(graph: nx.DiGraph, anchors: list[str]) -> set[tuple[str, str]]:
    undirected = graph.to_undirected(as_view=True)
    edges = set()
    for source, target in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, source, target, cutoff=PATH_LENGTH):
            edges.update((a, b) if graph.has_edge(a, b) else (b, a) for a, b in pairwise(path))
    return edges


def number_facts(facts: list[str]) -> str:
    lines, size = [], 0
    for index, fact in enumerate(facts, 1):
        line = f"{index}. {fact}"
        size += len(line) + 1
        if size > MAX_FACTS_CHARS:
            break
        lines.append(line)
    return "\n".join(lines)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    anchors = find_anchors(question, kb)
    edges = path_edges(kb.graph, anchors) if anchors else set()
    if not edges:
        return Answer(NOT_FOUND, refused=True)
    facts = [describe_edge(kb.graph, source, target) for source, target in sorted(edges)]
    prompt = f"Факты:\n{number_facts(facts)}\n\nВопрос: {question}"
    reply = llm.chat([{"role": "system", "content": LOCAL_SYSTEM}, {"role": "user", "content": prompt}])
    return Answer(reply.content, facts=facts, refused=NOT_FOUND in reply.content.lower())


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    # TODO: частичные ответы по сводкам, объединение
    raise NotImplementedError("Реализуйте global_search")
