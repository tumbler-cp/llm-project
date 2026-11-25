"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

RELATION_MEANINGS = {
    "BELONGS_TO": "модель входит в линейку",
    "HAS_PART": "в модели установлена деталь",
    "IS_A": "деталь является узлом этого вида",
    "INDICATES": "код ошибки указывает на отказ узла",
    "FIXED_BY": "ошибка устраняется по регламенту",
    "REPLACED_BY": "старая деталь снята с производства и заменена новой: старая -> новая",
    "SUPPLIED_BY": "деталь поставляет поставщик",
    "FAILED": "в отчёте о выезде у модели зафиксирован отказ узла",
}

EXAMPLES = [
    (
        "# Руководство по эксплуатации: моноблок BZ-990\n\n"
        "Моноблок BZ-990 входит в линейку «Бриз».\n\n"
        "- компрессор – деталь P-9010, поставщик «КомпрессорМаш»\n\n"
        "- E-91 – перегрев компрессора, отказ узла «компрессор»",
        {
            "entities": [
                {"type": "Model", "name": "BZ-990"},
                {"type": "ProductLine", "name": "Бриз"},
                {"type": "Part", "name": "P-9010"},
                {"type": "Component", "name": "компрессор"},
                {"type": "Supplier", "name": "КомпрессорМаш"},
                {"type": "ErrorCode", "name": "E-91"},
            ],
            "relations": [
                ["Model", "BZ-990", "BELONGS_TO", "ProductLine", "Бриз"],
                ["Model", "BZ-990", "HAS_PART", "Part", "P-9010"],
                ["Part", "P-9010", "IS_A", "Component", "компрессор"],
                ["Part", "P-9010", "SUPPLIED_BY", "Supplier", "КомпрессорМаш"],
                ["ErrorCode", "E-91", "INDICATES", "Component", "компрессор"],
            ],
        },
    ),
    (
        "# Сервисный бюллетень SB-2099-01\n\n"
        "Деталь P-9020 (датчик температуры) снята с производства, вместо неё ставится P-9020B "
        "от «ТермоПрибор». Ошибка E-92 устраняется по регламенту SP-199.",
        {
            "entities": [
                {"type": "Part", "name": "P-9020"},
                {"type": "Part", "name": "P-9020B"},
                {"type": "Component", "name": "датчик температуры"},
                {"type": "Supplier", "name": "ТермоПрибор"},
                {"type": "ErrorCode", "name": "E-92"},
                {"type": "Procedure", "name": "SP-199"},
            ],
            "relations": [
                ["Part", "P-9020", "REPLACED_BY", "Part", "P-9020B"],
                ["Part", "P-9020", "IS_A", "Component", "датчик температуры"],
                ["Part", "P-9020B", "IS_A", "Component", "датчик температуры"],
                ["Part", "P-9020B", "SUPPLIED_BY", "Supplier", "ТермоПрибор"],
                ["ErrorCode", "E-92", "FIXED_BY", "Procedure", "SP-199"],
            ],
        },
    ),
    (
        "# Отчёт о выезде R-2099-001\n\nОборудование: витрина ar 990. На панели код Е-93.\n\n"
        "Диагностика показала отказ узла: вентилятор испарителя.",
        {
            "entities": [
                {"type": "Model", "name": "AR-990"},
                {"type": "ErrorCode", "name": "E-93"},
                {"type": "Component", "name": "вентилятор испарителя"},
            ],
            "relations": [
                ["Model", "AR-990", "FAILED", "Component", "вентилятор испарителя"],
                ["ErrorCode", "E-93", "INDICATES", "Component", "вентилятор испарителя"],
            ],
        },
    ),
]


def _system_prompt(vocab: dict[str, list[str]]) -> str:
    relations = "\n".join(
        f"- {kind}: {subject} -> {obj} ({RELATION_MEANINGS[kind]})"
        for kind, (subject, obj) in RELATION_SIGNATURES.items()
    )
    names = "\n".join(f"- {kind}: {', '.join(items)}" for kind, items in vocab.items())
    return (
        "Ты извлекаешь граф знаний из документа сервисной службы холодильного оборудования.\n"
        f"Типы сущностей: {', '.join(ENTITY_TYPES)}.\n"
        "Коды пиши канонически: модели AR-210, детали P-1042 и P-1042B, ошибки E-17, регламенты SP-104.\n"
        f"Названия линеек, узлов и поставщиков бери только из словаря:\n{names}\n"
        f"Связи и их направление, начало -> конец:\n{relations}\n"
        "Извлекай только то, что прямо сказано в тексте. Не придумывай коды. "
        "Номера бюллетеней и отчётов не являются сущностями. Если связей нет, верни пустые списки.\n"
        "Ответ – JSON с полями entities и relations."
    )


def _example_answer(answer: dict) -> str:
    relations = [
        {"subject_type": st, "subject": s, "type": kind, "object_type": ot, "object": o}
        for st, s, kind, ot, o in answer["relations"]
    ]
    return json.dumps({"entities": answer["entities"], "relations": relations}, ensure_ascii=False)


def _messages(chunk: Chunk, vocab: dict[str, list[str]]) -> list[dict]:
    messages = [{"role": "system", "content": _system_prompt(vocab)}]
    for text, answer in EXAMPLES:
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": _example_answer(answer)})
    messages.append({"role": "user", "content": chunk.text})
    return messages


def _entity(entity_type: str, raw: str, vocab: dict[str, list[str]]) -> Entity | None:
    name = canonical_name(entity_type, raw, vocab)
    return Entity(entity_type, name) if name else None


def _result(chunk: Chunk, data: dict, vocab: dict[str, list[str]]) -> ExtractionResult:
    entities = {}
    for item in data["entities"]:
        entity = _entity(item["type"], item["name"], vocab)
        if entity:
            entities[entity.node_id] = entity
    relations = []
    for item in data["relations"]:
        subject = _entity(item["subject_type"], item["subject"], vocab)
        obj = _entity(item["object_type"], item["object"], vocab)
        if subject and obj:
            relations.append(Relation(subject, item["type"], obj))
    return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(entities.values()), relations)


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    vocab = vocabulary()
    response = llm.chat(_messages(chunk, vocab), json_schema=EXTRACTION_SCHEMA)
    return _result(chunk, json.loads(response.content), vocab)
