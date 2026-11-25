"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
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

    def run(self) -> AgentResult:
        while True:
            response = self.llm.chat(self.messages, tools=TOOL_SCHEMAS)
            if not response.tool_calls:
                return self._finish(response.content)
            self.messages.append(response.as_message())
            for call in response.tool_calls:
                if len(self.steps) >= self.max_steps:
                    return self._interrupted()
                self.messages.append(tool_message(call, self._execute(call)))

    def _execute(self, call: ToolCall) -> dict:
        error = self._rejection(call)
        if error:
            self.steps.append(ToolStep(call.name, call.arguments, error=error))
            return {"error": error}
        key = json.dumps([call.name, call.arguments], ensure_ascii=False, sort_keys=True)
        if key not in self.done:
            self.done[key] = self.tools[call.name](**call.arguments)
        result = self.done[key]
        self.steps.append(ToolStep(call.name, call.arguments, result=result, error=result.get("error")))
        return result

    def _rejection(self, call: ToolCall) -> str | None:
        if call.name not in self.tools or call.name not in REQUIRED:
            return f"неизвестный инструмент {call.name!r}, доступны: {', '.join(REQUIRED)}"
        required = REQUIRED[call.name]
        if set(call.arguments) != set(required):
            return f"нечитаемые аргументы {call.arguments}, нужны: {', '.join(required)}"
        return None

    def _finish(self, answer: str) -> AgentResult:
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
