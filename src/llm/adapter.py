from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
import asyncio
from dataclasses import dataclass
from typing import Callable, Type, TypeVar

import requests
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from src.config import ModelConfig
from src.planner.schema import DatasetAssessment, StepDecision, WorkflowPlan, WorkflowStep

T = TypeVar("T")

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


def _is_transient_llm_error(exc: BaseException) -> bool:
    """True for likely network/TLS/proxy failures worth retrying."""
    if httpx is not None:
        if isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
                httpx.RemoteProtocolError,
            ),
        ):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            return code in (429, 502, 503, 504)
    if isinstance(exc, ConnectionError):
        return True
    msg = str(exc).lower()
    if any(
        s in msg
        for s in (
            "ssl",
            "unexpected_eof",
            "eof occurred",
            "connection reset",
            "connection aborted",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "bad gateway",
            "gateway timeout",
        )
    ):
        return True
    # OpenAI SDK errors (optional dependency path)
    et = type(exc).__name__
    if et in ("APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"):
        return True
    return False


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _strip_markdown_json_fences(text: str) -> str:
    """Remove ``` / ```json wrappers so model_validate_json sees raw JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _message_content_to_text(msg: object) -> str | None:
    """Best-effort extract assistant text from LangChain AIMessage (or similar)."""
    if msg is None:
        return None
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and str(block.get("type", "")).lower() == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else None
    if content is not None:
        return str(content)
    return None


def _try_recover_structured_from_raw(raw_msg: object, schema: Type[BaseModel]) -> BaseModel | None:
    """If LangChain structured parsing failed (e.g. ```json fences), parse manually."""
    text = _message_content_to_text(raw_msg)
    if not text or not text.strip():
        return None
    cleaned = _strip_markdown_json_fences(text)
    try:
        return schema.model_validate_json(cleaned)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            return schema.model_validate_json(m.group(0))
        except Exception:
            return None
    return None


_MOCK_PLAN = WorkflowPlan(
    workflow_name="mock_workflow",
    rationale="Mock planner fallback for local testing.",
    dataset_assessment=DatasetAssessment(
        detected_missingness=True,
        detected_batch_risk=True,
        detected_compositionality=True,
        recommended_input_matrix="abundance",
    ),
    decision_audit=[
        {
            "aspect": "missingness",
            "observed_evidence": ["missing_fraction_observed_above_zero"],
            "judgment": "missingness_requires_imputation",
            "chosen_action": "impute_missing_values",
            "rejected_actions": ["skip_imputation"],
            "confidence": 0.9,
        },
        {
            "aspect": "compositionality",
            "observed_evidence": ["rows_sum_near_constant_like_composition"],
            "judgment": "clr_needed_before_de",
            "chosen_action": "clr_transform",
            "rejected_actions": ["de_on_raw_abundance"],
            "confidence": 0.95,
        },
        {
            "aspect": "batch",
            "observed_evidence": ["batch_labels_available", "n_batches_gt_one"],
            "judgment": "diagnose_then_correct_batch",
            "chosen_action": "check_batch_effect_then_run_combat",
            "rejected_actions": ["ignore_batch", "correct_without_check"],
            "confidence": 0.75,
        },
    ],
    steps=[
        WorkflowStep(tool="inspect_dataset", purpose="summarize matrix"),
        WorkflowStep(tool="impute_missing_values", params={"strategy": "median"}, purpose="impute"),
        WorkflowStep(tool="clr_transform", params={"pseudocount": 1e-6}, purpose="clr"),
        WorkflowStep(tool="check_batch_effect", purpose="batch qc"),
        WorkflowStep(tool="run_combat", params={"use_bio_covariate": True}, purpose="batch correction"),
        WorkflowStep(tool="run_differential_expression", params={"alpha": 0.05}, purpose="DE"),
        WorkflowStep(tool="run_pca", purpose="qc"),
    ],
    final_analysis_target="differential_expression",
    stop_conditions=[],
)

_MOCK_PLAN_JSON = (
    '{"workflow_name":"mock_workflow","rationale":"mock",'
    '"dataset_assessment":{"detected_missingness":true,"detected_batch_risk":true,'
    '"detected_compositionality":true,"recommended_input_matrix":"abundance"},'
    '"decision_audit":['
    '{"aspect":"missingness","observed_evidence":["missing_fraction_observed_above_zero"],"judgment":"missingness_requires_imputation","chosen_action":"impute_missing_values","rejected_actions":["skip_imputation"],"confidence":0.9},'
    '{"aspect":"compositionality","observed_evidence":["rows_sum_near_constant_like_composition"],"judgment":"clr_needed_before_de","chosen_action":"clr_transform","rejected_actions":["de_on_raw_abundance"],"confidence":0.95},'
    '{"aspect":"batch","observed_evidence":["batch_labels_available","n_batches_gt_one"],"judgment":"diagnose_then_correct_batch","chosen_action":"check_batch_effect_then_run_combat","rejected_actions":["ignore_batch","correct_without_check"],"confidence":0.75}'
    '],'
    '"final_analysis_target":"differential_expression","stop_conditions":[],'
    '"steps":['
    '{"tool":"inspect_dataset","purpose":"qc"},'
    '{"tool":"impute_missing_values","params":{"strategy":"median"}},'
    '{"tool":"clr_transform"},{"tool":"check_batch_effect"},'
    '{"tool":"run_combat"},{"tool":"run_differential_expression"},{"tool":"run_pca"}]}'
)


def _mock_step_decision(user_prompt: str) -> StepDecision:
    executed_tools: list[str] = []
    m = re.search(r'"executed_tools"\s*:\s*(\[[^\]]*\])', user_prompt)
    if m:
        try:
            executed_tools = list(json.loads(m.group(1)))
        except Exception:
            executed_tools = []
    sequence = [
        WorkflowStep(tool="inspect_dataset", purpose="summarize matrix"),
        WorkflowStep(tool="impute_missing_values", params={"strategy": "median"}, purpose="impute"),
        WorkflowStep(tool="clr_transform", params={"pseudocount": 1e-6}, purpose="clr"),
        WorkflowStep(tool="check_batch_effect", purpose="batch qc"),
        WorkflowStep(tool="run_combat", params={"use_bio_covariate": True}, purpose="batch correction"),
        WorkflowStep(tool="run_differential_expression", params={"alpha": 0.05}, purpose="DE"),
        WorkflowStep(tool="run_pca", purpose="qc"),
    ]
    if len(executed_tools) >= len(sequence):
        return StepDecision(done=True, rationale="Mock iterative planner finished.", stop_reason="sequence_complete")
    return StepDecision(
        done=False,
        next_step=sequence[len(executed_tools)].model_copy(),
        rationale="Mock iterative planner step.",
        updated_final_analysis_target="differential_expression",
    )


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMCallResult:
    content: object
    usage: LLMUsage


class LLMAdapter:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def supports_tool_calling(self) -> bool:
        return self.cfg.supports_tool_calling

    def _invoke_with_outer_retry(self, fn: Callable[[], T]) -> T:
        attempts = _env_int("GLYCOAGENT_LLM_OUTER_ATTEMPTS", 4)
        backoff = _env_float("GLYCOAGENT_LLM_RETRY_BACKOFF_SEC", 1.5)
        last: BaseException | None = None
        for i in range(attempts):
            try:
                return fn()
            except BaseException as e:
                last = e
                if i >= attempts - 1 or not _is_transient_llm_error(e):
                    raise
                time.sleep(min(60.0, backoff * (2**i)))
        assert last is not None
        raise last

    def invoke_structured(self, system_prompt: str, user_prompt: str, schema: Type[BaseModel]) -> BaseModel:
        return self.invoke_structured_with_usage(system_prompt, user_prompt, schema).content

    def invoke_structured_with_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[BaseModel],
    ) -> LLMCallResult:
        if self.cfg.provider == "mock":
            if schema is WorkflowPlan:
                return LLMCallResult(content=_MOCK_PLAN.model_copy(), usage=LLMUsage())
            if schema is StepDecision:
                return LLMCallResult(content=_mock_step_decision(user_prompt), usage=LLMUsage())
            raise ValueError("mock provider only supports WorkflowPlan schema in this demo")

        if self.cfg.provider == "stream_server_http":
            # Use direct HTTP POST, then manually parse JSON into schema
            augmented_user = (
                user_prompt
                + "\n\nYou MUST respond with ONLY valid JSON matching the required schema. "
                "No markdown fences, no extra text."
            )
            text, usage = self._invoke_with_outer_retry(
                lambda: self._post_llm(system_prompt, augmented_user),
            )
            cleaned = _strip_markdown_json_fences(text)
            parsed = schema.model_validate_json(cleaned)
            return LLMCallResult(content=parsed, usage=usage)

        if self.cfg.provider not in {"openai_compatible", "local_openai_http"}:
            raise ValueError(f"unsupported provider: {self.cfg.provider}")

        llm = self._build_llm()
        s_llm = llm.with_structured_output(schema, include_raw=True)
        result = self._invoke_with_outer_retry(
            lambda: s_llm.invoke([("system", system_prompt), ("user", user_prompt)]),
        )
        if not isinstance(result, dict):
            return LLMCallResult(content=result, usage=self._extract_usage(result))
        raw_msg = result.get("raw")
        parsing_error = result.get("parsing_error")
        parsed = result.get("parsed")
        if parsing_error is not None or parsed is None:
            recovered = _try_recover_structured_from_raw(raw_msg, schema)
            if recovered is not None:
                logger.info(
                    "Structured output recovered after fence strip / manual JSON parse (schema=%s)",
                    getattr(schema, "__name__", str(schema)),
                )
                return LLMCallResult(content=recovered, usage=self._extract_usage(raw_msg))
        if parsing_error is not None:
            raise parsing_error
        return LLMCallResult(content=parsed, usage=self._extract_usage(raw_msg))

    def invoke_text(self, system_prompt: str, user_prompt: str) -> str:
        return str(self.invoke_text_with_usage(system_prompt, user_prompt).content)

    def invoke_text_with_usage(self, system_prompt: str, user_prompt: str) -> LLMCallResult:
        if self.cfg.provider == "mock":
            return LLMCallResult(content=_MOCK_PLAN_JSON, usage=LLMUsage())

        if self.cfg.provider == "stream_server_http":
            text, usage = self._invoke_with_outer_retry(
                lambda: self._post_llm(system_prompt, user_prompt),
            )
            return LLMCallResult(content=text, usage=usage)

        if self.cfg.provider not in {"openai_compatible", "local_openai_http"}:
            raise ValueError(f"unsupported provider: {self.cfg.provider}")
        llm = self._build_llm()
        msg = self._invoke_with_outer_retry(
            lambda: llm.invoke([("system", system_prompt), ("user", user_prompt)]),
        )
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return LLMCallResult(content=content, usage=self._extract_usage(msg))

    def _post_llm(self, system_prompt: str, user_prompt: str) -> tuple[str, LLMUsage]:
        """Call the stream-server HTTP API directly via requests.post."""
        ss_url = self.cfg.base_url or os.getenv(self.cfg.base_url_env, "")
        wsid = self.cfg.ss_wsid or os.getenv("GLYCODA_STREAM_WSID", "")
        auth_token = self.cfg.ss_auth_token or os.getenv(self.cfg.api_key_env, "")
        if not ss_url or not auth_token:
            raise RuntimeError(
                "stream_server_http requires base URL and token through the configured environment variables"
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token}",
            "Wsid": wsid,
        }
        json_data = {
            "model": self.cfg.model,
            "query_id": "glycoagent_" + str(uuid.uuid4()),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.cfg.temperature,
            "top_p": 0.8,
            "top_k": 40,
            "repetition_penalty": 1,
            "chat_template_kwargs": {
                "enable_thinking": self.cfg.enable_thinking,
                "thinking_budget": 4096 if self.cfg.enable_thinking else 0,
            },
            "stream": False,
            "openai_infer": True,
        }

        timeout = _env_float("GLYCOAGENT_LLM_TIMEOUT_SEC", 600.0)
        logger.info(
            "[stream-server] POST %s  model=%s  enable_thinking=%s",
            ss_url, self.cfg.model, self.cfg.enable_thinking,
        )

        resp = requests.post(ss_url, headers=headers, json=json_data, stream=True, timeout=timeout)
        resp.raise_for_status()
        resp_json = resp.json()
        logger.debug("[stream-server] response: %s", resp_json)

        # Defensive: server may return malformed JSON on transient errors
        try:
            result_text = resp_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            err_msg = resp_json.get("error", resp_json.get("message", str(resp_json)[:500]))
            logger.warning(
                "[stream-server] malformed response (missing 'choices'), "
                "likely transient – will retry. error=%s",
                err_msg,
            )
            raise ConnectionError(
                f"stream-server returned unexpected response (no 'choices'): {err_msg}"
            ) from exc

        # Extract usage if available
        usage_data = resp_json.get("usage", {})
        usage = LLMUsage(
            input_tokens=int(usage_data.get("prompt_tokens", 0)),
            output_tokens=int(usage_data.get("completion_tokens", 0)),
            total_tokens=int(usage_data.get("total_tokens", 0)),
        )
        logger.info("[stream-server] done  tokens=%d", usage.total_tokens)
        return result_text, usage

    def _build_llm(self) -> ChatOpenAI:
        api_key = os.getenv(self.cfg.api_key_env, "")
        if not api_key and not self.cfg.allow_missing_api_key:
            raise RuntimeError(
                f"missing API key env: {self.cfg.api_key_env}. "
                "Set it in the shell, put it in repo-root `.env`, or set `api_key` in `configs/api.yaml` "
                "(non-empty). If the variable exists but is empty, ensure `.env` is loaded (override) or unset it."
            )
        if not api_key:
            # Local OpenAI-compatible servers (vLLM / LM Studio / Ollama OpenAI API)
            # often do not enforce API key; pass a harmless placeholder.
            api_key = "EMPTY"
        base_url = self.cfg.base_url or os.getenv(self.cfg.base_url_env, "")
        timeout = _env_float("GLYCOAGENT_LLM_TIMEOUT_SEC", 600.0)
        max_retries = _env_int("GLYCOAGENT_LLM_HTTP_MAX_RETRIES", 2)
        kwargs = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if self.cfg.enable_thinking is not None:
            budget = self.cfg.thinking_budget
            if budget is None:
                budget = 4096 if self.cfg.enable_thinking else 0
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": bool(self.cfg.enable_thinking),
                    "thinking_budget": int(budget),
                }
            }
        return ChatOpenAI(
            **kwargs,
        )

    @staticmethod
    def _extract_usage(msg: object) -> LLMUsage:
        usage = getattr(msg, "usage_metadata", None) or {}
        response_meta = getattr(msg, "response_metadata", None) or {}
        token_usage = response_meta.get("token_usage", {}) if isinstance(response_meta, dict) else {}

        def _to_int(value: object) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        input_tokens = _to_int(
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or token_usage.get("prompt_tokens")
            or token_usage.get("input_tokens")
        )
        output_tokens = _to_int(
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or token_usage.get("completion_tokens")
            or token_usage.get("output_tokens")
        )
        total_tokens = _to_int(
            usage.get("total_tokens")
            or token_usage.get("total_tokens")
            or (input_tokens + output_tokens)
        )
        return LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    # ------------------------------------------------------------------
    # Async methods: use asyncio for non-blocking IO-bound LLM calls
    # ------------------------------------------------------------------

    async def _ainvoke_with_outer_retry(self, fn: Callable[[], T]) -> T:
        """Run a sync LLM call in a thread pool with retry logic (async version)."""
        attempts = _env_int("GLYCOAGENT_LLM_OUTER_ATTEMPTS", 4)
        backoff = _env_float("GLYCOAGENT_LLM_RETRY_BACKOFF_SEC", 1.5)
        last: BaseException | None = None
        loop = asyncio.get_event_loop()
        for i in range(attempts):
            try:
                return await loop.run_in_executor(None, fn)
            except BaseException as e:
                last = e
                if i >= attempts - 1 or not _is_transient_llm_error(e):
                    raise
                await asyncio.sleep(min(60.0, backoff * (2**i)))
        assert last is not None
        raise last

    async def _apost_llm(self, system_prompt: str, user_prompt: str) -> tuple[str, LLMUsage]:
        """Async version of _post_llm using aiohttp for non-blocking HTTP."""
        try:
            import aiohttp
        except ImportError:
            # Fallback: run sync version in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._post_llm, system_prompt, user_prompt)

        ss_url = self.cfg.base_url or os.getenv(self.cfg.base_url_env, "")
        wsid = self.cfg.ss_wsid or os.getenv("GLYCODA_STREAM_WSID", "")
        auth_token = self.cfg.ss_auth_token or os.getenv(self.cfg.api_key_env, "")
        if not ss_url or not auth_token:
            raise RuntimeError(
                "stream_server_http requires base URL and token through the configured environment variables"
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token}",
            "Wsid": wsid,
        }
        json_data = {
            "model": self.cfg.model,
            "query_id": "glycoagent_" + str(uuid.uuid4()),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.cfg.temperature,
            "top_p": 0.8,
            "top_k": 40,
            "repetition_penalty": 1,
            "chat_template_kwargs": {
                "enable_thinking": self.cfg.enable_thinking,
                "thinking_budget": 4096 if self.cfg.enable_thinking else 0,
            },
            "stream": False,
            "openai_infer": True,
        }

        timeout_sec = _env_float("GLYCOAGENT_LLM_TIMEOUT_SEC", 600.0)
        logger.info(
            "[stream-server-async] POST %s  model=%s  enable_thinking=%s",
            ss_url, self.cfg.model, self.cfg.enable_thinking,
        )

        timeout = aiohttp.ClientTimeout(total=timeout_sec)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(ss_url, headers=headers, json=json_data) as resp:
                resp.raise_for_status()
                resp_json = await resp.json()

        logger.debug("[stream-server-async] response: %s", resp_json)

        try:
            result_text = resp_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            err_msg = resp_json.get("error", resp_json.get("message", str(resp_json)[:500]))
            logger.warning(
                "[stream-server-async] malformed response (missing 'choices'), "
                "likely transient – will retry. error=%s",
                err_msg,
            )
            raise ConnectionError(
                f"stream-server returned unexpected response (no 'choices'): {err_msg}"
            ) from exc

        usage_data = resp_json.get("usage", {})
        usage = LLMUsage(
            input_tokens=int(usage_data.get("prompt_tokens", 0)),
            output_tokens=int(usage_data.get("completion_tokens", 0)),
            total_tokens=int(usage_data.get("total_tokens", 0)),
        )
        logger.info("[stream-server-async] done  tokens=%d", usage.total_tokens)
        return result_text, usage

    async def ainvoke_structured(self, system_prompt: str, user_prompt: str, schema: Type[BaseModel]) -> BaseModel:
        """Async version of invoke_structured."""
        return (await self.ainvoke_structured_with_usage(system_prompt, user_prompt, schema)).content

    async def ainvoke_structured_with_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[BaseModel],
    ) -> LLMCallResult:
        """Async version of invoke_structured_with_usage."""
        if self.cfg.provider == "mock":
            if schema is WorkflowPlan:
                return LLMCallResult(content=_MOCK_PLAN.model_copy(), usage=LLMUsage())
            if schema is StepDecision:
                return LLMCallResult(content=_mock_step_decision(user_prompt), usage=LLMUsage())
            raise ValueError("mock provider only supports WorkflowPlan schema in this demo")

        if self.cfg.provider == "stream_server_http":
            augmented_user = (
                user_prompt
                + "\n\nYou MUST respond with ONLY valid JSON matching the required schema. "
                "No markdown fences, no extra text."
            )
            attempts = _env_int("GLYCOAGENT_LLM_OUTER_ATTEMPTS", 4)
            backoff = _env_float("GLYCOAGENT_LLM_RETRY_BACKOFF_SEC", 1.5)
            last_exc: BaseException | None = None
            for i in range(attempts):
                try:
                    text, usage = await self._apost_llm(system_prompt, augmented_user)
                    break
                except BaseException as e:
                    last_exc = e
                    if i >= attempts - 1 or not _is_transient_llm_error(e):
                        raise
                    await asyncio.sleep(min(60.0, backoff * (2**i)))
            else:
                assert last_exc is not None
                raise last_exc
            cleaned = _strip_markdown_json_fences(text)
            parsed = schema.model_validate_json(cleaned)
            return LLMCallResult(content=parsed, usage=usage)

        if self.cfg.provider not in {"openai_compatible", "local_openai_http"}:
            raise ValueError(f"unsupported provider: {self.cfg.provider}")

        # For OpenAI-compatible providers, run sync call in thread pool
        llm = self._build_llm()
        s_llm = llm.with_structured_output(schema, include_raw=True)
        result = await self._ainvoke_with_outer_retry(
            lambda: s_llm.invoke([("system", system_prompt), ("user", user_prompt)]),
        )
        if not isinstance(result, dict):
            return LLMCallResult(content=result, usage=self._extract_usage(result))
        raw_msg = result.get("raw")
        parsing_error = result.get("parsing_error")
        parsed = result.get("parsed")
        if parsing_error is not None or parsed is None:
            recovered = _try_recover_structured_from_raw(raw_msg, schema)
            if recovered is not None:
                logger.info(
                    "Structured output recovered after fence strip / manual JSON parse (schema=%s)",
                    getattr(schema, "__name__", str(schema)),
                )
                return LLMCallResult(content=recovered, usage=self._extract_usage(raw_msg))
        if parsing_error is not None:
            raise parsing_error
        return LLMCallResult(content=parsed, usage=self._extract_usage(raw_msg))

    async def ainvoke_text(self, system_prompt: str, user_prompt: str) -> str:
        """Async version of invoke_text."""
        return str((await self.ainvoke_text_with_usage(system_prompt, user_prompt)).content)

    async def ainvoke_text_with_usage(self, system_prompt: str, user_prompt: str) -> LLMCallResult:
        """Async version of invoke_text_with_usage."""
        if self.cfg.provider == "mock":
            return LLMCallResult(content=_MOCK_PLAN_JSON, usage=LLMUsage())

        if self.cfg.provider == "stream_server_http":
            attempts = _env_int("GLYCOAGENT_LLM_OUTER_ATTEMPTS", 4)
            backoff = _env_float("GLYCOAGENT_LLM_RETRY_BACKOFF_SEC", 1.5)
            last_exc: BaseException | None = None
            for i in range(attempts):
                try:
                    text, usage = await self._apost_llm(system_prompt, user_prompt)
                    break
                except BaseException as e:
                    last_exc = e
                    if i >= attempts - 1 or not _is_transient_llm_error(e):
                        raise
                    await asyncio.sleep(min(60.0, backoff * (2**i)))
            else:
                assert last_exc is not None
                raise last_exc
            return LLMCallResult(content=text, usage=usage)

        if self.cfg.provider not in {"openai_compatible", "local_openai_http"}:
            raise ValueError(f"unsupported provider: {self.cfg.provider}")

        llm = self._build_llm()
        msg = await self._ainvoke_with_outer_retry(
            lambda: llm.invoke([("system", system_prompt), ("user", user_prompt)]),
        )
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return LLMCallResult(content=content, usage=self._extract_usage(msg))
