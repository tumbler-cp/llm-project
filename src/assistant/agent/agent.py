"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable
from contextlib import suppress

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes, normalize_code
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

SYSTEM_PROMPT = (
    "Ты помощник диспетчера сервисной службы «ПолярТех». Отвечай кратко по-русски.\n"
    "Чтобы узнать узел по коду ошибки, деталь модели и её замену, вызови search_knowledge. "
    "Для обзорных вопросов о частых отказах вызови summarize_topic. "
    "Остатки и цены бери только из check_stock и get_price.\n"
    "Номера деталей бери только из вопроса и из результатов инструментов. "
    "Если деталь заменена, называй последнюю деталь цепочки замен.\n"
    "Заказ create_service_order оформляй, только если пользователь просит заказ и назвал количество.\n"
    "Коды, цены и остатки пиши так же, как в результатах инструментов, числа без пробелов.\n"
    "Если данных нет, ответь: «В базе знаний не найдено»."
)
REFUSAL = "в базе знаний не найдено"
ERP_DOWN = "Данные ERP получить не удалось."
ERP_ATTEMPTS = 3
ORDER_TOOL = "create_service_order"
NUMBER = re.compile(r"(?<![\w-])\d+(?![\w-])")
REQUIRED = {schema["function"]["name"]: schema["function"]["parameters"]["required"] for schema in TOOL_SCHEMAS}


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    return _AgentRun(question, llm, tools, max_steps, confirm).run()


class _AgentRun:
    def __init__(
        self, question: str, llm: LLMClient, tools: dict[str, Callable[..., dict]], max_steps: int, confirm: bool
    ) -> None:
        self.question = question
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.confirm = confirm
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        self.steps: list[ToolStep] = []
        self.done: dict[str, dict] = {}
        self.erp_down = False

    def run(self) -> AgentResult:
        while True:
            response = self.llm.chat(self.messages, tools=TOOL_SCHEMAS)
            if not response.tool_calls:
                return self._finish(response.content)
            self.messages.append(response.as_message())
            for call in response.tool_calls:
                if len(self.steps) >= self.max_steps:
                    return self._interrupted()
                held = self._hold_order(call)
                if held:
                    return held
                self.messages.append(tool_message(call, self._execute(call)))

    def _hold_order(self, call: ToolCall) -> AgentResult | None:
        if call.name != ORDER_TOOL or self._rejection(call):
            return None
        args = call.arguments
        if args["qty"] not in _numbers(self.question):
            self.steps.append(ToolStep(call.name, args, error="количество не названо пользователем"))
            question = f"Сколько штук {args['part_id']} заказать? В вопросе количество не названо."
            return AgentResult(question, self.steps, self._sources())
        if self.confirm:
            return None
        answer = (
            f"Готов оформить заказ: {args['part_id']}, {args['qty']} шт., склад {args['warehouse']}, "
            f"модель {args['model']}. Подтвердите заказ."
        )
        pending = {"tool": ORDER_TOOL, "args": args}
        return AgentResult(answer, self.steps, self._sources(), needs_confirmation=True, pending_action=pending)

    def _execute(self, call: ToolCall) -> dict:
        error = self._rejection(call)
        if error:
            self.steps.append(ToolStep(call.name, call.arguments, error=error))
            return {"error": error}
        key = json.dumps([call.name, call.arguments], ensure_ascii=False, sort_keys=True)
        if key not in self.done:
            try:
                self.done[key] = self._call_with_retries(call)
            except ErpUnavailable as exc:
                self.erp_down = True
                error = f"{ERP_DOWN} {exc}"
                self.steps.append(ToolStep(call.name, call.arguments, error=error))
                return {"error": error}
        result = self.done[key]
        self.steps.append(ToolStep(call.name, call.arguments, result=result, error=result.get("error")))
        return result

    def _call_with_retries(self, call: ToolCall) -> dict:
        for _ in range(ERP_ATTEMPTS - 1):
            with suppress(ErpUnavailable):
                return self.tools[call.name](**call.arguments)
        return self.tools[call.name](**call.arguments)

    def _rejection(self, call: ToolCall) -> str | None:
        if call.name not in self.tools or call.name not in REQUIRED:
            return f"неизвестный инструмент {call.name!r}, доступны: {', '.join(REQUIRED)}"
        required = REQUIRED[call.name]
        if set(call.arguments) != set(required):
            return f"нечитаемые аргументы {call.arguments}, нужны: {', '.join(required)}"
        part_id = call.arguments.get("part_id")
        if part_id is not None and normalize_code(str(part_id)) not in find_codes(self._evidence()):
            return (
                f"номер детали {part_id!r} не встречался в вопросе и результатах инструментов. "
                "Вызови search_knowledge, чтобы узнать номер детали"
            )
        return None

    def _evidence(self) -> str:
        results = [json.dumps(step.result, ensure_ascii=False) for step in self.steps if step.result]
        return "\n".join([self.question, *results])

    def _finish(self, answer: str) -> AgentResult:
        if self.erp_down and ERP_DOWN.lower() not in answer.lower():
            answer = f"{answer} {ERP_DOWN}".strip()
        return AgentResult(answer, self.steps, self._sources(), refused=REFUSAL in answer.lower())

    def _interrupted(self) -> AgentResult:
        answer = f"Работа прервана: исчерпан лимит в {self.max_steps} вызовов инструментов."
        learned = [
            f"{step.tool}: {json.dumps(step.result, ensure_ascii=False)}"
            for step in self.steps
            if step.result and not step.error
        ]
        if learned:
            answer += " Удалось узнать:\n" + "\n".join(learned)
        return AgentResult(answer, self.steps, self._sources())

    def _sources(self) -> list[str]:
        sources = [source for step in self.steps if step.result for source in step.result.get("sources", [])]
        return list(dict.fromkeys(sources))


def _numbers(text: str) -> set[int]:
    return {int(number) for number in NUMBER.findall(text)}
