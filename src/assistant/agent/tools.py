"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

QUESTION = {"question": {"type": "string", "description": "Вопрос на русском языке"}}
PART_ID = {"part_id": {"type": "string", "description": "Номер детали, например P-1042B"}}
WAREHOUSE = {"warehouse": {"type": "string", "enum": WAREHOUSES, "description": "Склад"}}
QTY = {"qty": {"type": "integer", "minimum": 1, "description": "Количество, которое назвал пользователь"}}
MODEL = {"model": {"type": "string", "description": "Модель оборудования, например AR-210"}}


def _schema(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(properties)},
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _schema(
        "search_knowledge",
        "Поиск по графу знаний: узел по коду ошибки, деталь модели, замена снятой детали, регламент ремонта",
        QUESTION,
    ),
    _schema("summarize_topic", "Обзорный ответ по всему корпусу: частые отказы, общие тенденции", QUESTION),
    _schema("check_stock", "Остаток детали на складе по данным ERP", PART_ID | WAREHOUSE),
    _schema("get_price", "Цена детали в рублях по данным ERP", PART_ID),
    _schema(
        "create_service_order",
        "Заказ детали на склад для ремонта модели. Только по просьбе пользователя",
        PART_ID | WAREHOUSE | QTY | MODEL,
    ),
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def search_knowledge(question: str) -> dict:
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        return erp.get_stock(part_id, warehouse)

    def get_price(part_id: str) -> dict:
        return erp.get_price(part_id)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        return erp.create_order(part_id, warehouse, qty, model)

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
