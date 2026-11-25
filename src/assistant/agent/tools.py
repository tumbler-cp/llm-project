"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
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
        part = _part(part_id)
        error = _part_error(part_id, part) or _warehouse_error(warehouse)
        return {"error": error} if error else _ask_erp(erp.get_stock, part, warehouse)

    def get_price(part_id: str) -> dict:
        part = _part(part_id)
        error = _part_error(part_id, part)
        return {"error": error} if error else _ask_erp(erp.get_price, part)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        part = _part(part_id)
        error = _part_error(part_id, part) or _warehouse_error(warehouse) or _qty_error(qty)
        return {"error": error} if error else _ask_erp(erp.create_order, part, warehouse, qty, model)

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }


def _part(part_id: object) -> str | None:
    code = normalize_code(str(part_id))
    return code if code and code_type(code) == "Part" else None


def _part_error(part_id: object, part: str | None) -> str | None:
    return None if part else f"неверный номер детали {part_id!r}: нужен вид P-1042 или P-1042B"


def _warehouse_error(warehouse: object) -> str | None:
    return None if warehouse in WAREHOUSES else f"неизвестный склад {warehouse!r}, допустимые: {', '.join(WAREHOUSES)}"


def _qty_error(qty: object) -> str | None:
    valid = isinstance(qty, int) and not isinstance(qty, bool) and qty > 0
    return None if valid else f"количество должно быть целым числом больше нуля, получено {qty!r}"


def _ask_erp(request: Callable[..., dict], *args) -> dict:
    try:
        return request(*args)
    except ErpNotFound as exc:
        return {"error": f"данных нет: {exc}"}
    except ErpRejected as exc:
        return {"error": f"ERP отклонила запрос: {exc}"}
