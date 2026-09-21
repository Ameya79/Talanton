"""Tests for talanton.integrations SDK wrappers."""

import pytest

from talanton.tracker import TalantonTracker


@pytest.fixture
def tmp_db(tmp_path):
    """Creates a temporary database path for test isolation."""
    return tmp_path / "test_integrations.db"


@pytest.fixture
def tracker(tmp_db):
    """Creates a TalantonTracker with temp database."""
    return TalantonTracker(db_path=tmp_db)


# ─── OpenAI Wrapper Tests ────────────────────────────────────────────────


class TestOpenAIWrapper:

    def test_auto_tracks_completion(self, tracker):
        """Verify OpenAI wrapper auto-tracks a chat completion call."""
        from talanton.integrations.openai_wrapper import TalantonOpenAI

        class MockUsage:
            prompt_tokens = 150
            completion_tokens = 75

        class MockResponse:
            usage = MockUsage()
            model = "gpt-4o"
            choices = [{"message": {"content": "Hello!"}}]

        class MockCompletions:
            def create(self, **kwargs):
                return MockResponse()

        class MockChat:
            completions = MockCompletions()

        class MockOpenAI:
            chat = MockChat()

        client = TalantonOpenAI(MockOpenAI(), tracker=tracker)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response.usage.prompt_tokens == 150
        calls = tracker.get_calls()
        assert len(calls) == 1
        assert calls[0]["model"] == "gpt-4o"
        assert calls[0]["input_tokens"] == 150
        assert calls[0]["output_tokens"] == 75

    def test_guardrail_blocks_call(self, tracker):
        """Verify guardrail blocks when hard limit is exceeded."""
        from talanton.guardrails import BudgetExceededError, BudgetGuardrail, raise_exception
        from talanton.integrations.openai_wrapper import TalantonOpenAI

        guardrail = BudgetGuardrail(
            tracker,
            soft_limit=0.0001,
            hard_limit=0.0002,
            period="day",
            on_hard_limit=raise_exception,
        )

        class MockCompletions:
            def create(self, **kwargs):
                raise AssertionError("Should not reach API call")

        class MockChat:
            completions = MockCompletions()

        class MockOpenAI:
            chat = MockChat()

        client = TalantonOpenAI(MockOpenAI(), tracker=tracker, guardrail=guardrail)

        with pytest.raises(BudgetExceededError):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "This should be blocked"}],
                max_tokens=10000,
            )

    def test_proxy_other_attributes(self, tracker):
        """Verify non-chat attributes are proxied to original client."""
        from talanton.integrations.openai_wrapper import TalantonOpenAI

        class MockOpenAI:
            api_key = "test-key-123"

            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        pass

        client = TalantonOpenAI(MockOpenAI(), tracker=tracker)
        assert client.api_key == "test-key-123"


# ─── Anthropic Wrapper Tests ─────────────────────────────────────────────


class TestAnthropicWrapper:

    def test_auto_tracks_message(self, tracker):
        """Verify Anthropic wrapper auto-tracks a messages.create() call."""
        from talanton.integrations.anthropic_wrapper import TalantonAnthropic

        class MockUsage:
            input_tokens = 200
            output_tokens = 100

        class MockResponse:
            usage = MockUsage()
            model = "claude-sonnet-4.5"
            content = [{"text": "Hello!"}]

        class MockMessages:
            def create(self, **kwargs):
                return MockResponse()

        class MockAnthropic:
            messages = MockMessages()

        client = TalantonAnthropic(MockAnthropic(), tracker=tracker)
        response = client.messages.create(
            model="claude-sonnet-4.5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response.usage.input_tokens == 200
        calls = tracker.get_calls()
        assert len(calls) == 1
        assert calls[0]["model"] == "claude-sonnet-4.5"
        assert calls[0]["input_tokens"] == 200


# ─── LangChain Callback Tests ────────────────────────────────────────────


class TestLangChainCallback:

    def test_on_llm_end_records_call(self, tracker):
        """Verify LangChain callback records call on on_llm_end."""
        from talanton.integrations.langchain_callback import TalantonCallbackHandler

        handler = TalantonCallbackHandler(tracker=tracker)

        # Simulate on_llm_start
        from uuid import uuid4
        run_id = uuid4()
        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            prompts=["Hello!"],
            run_id=run_id,
        )

        # Simulate on_llm_end with LLMResult-like object
        class MockLLMResult:
            llm_output = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}}

        handler.on_llm_end(MockLLMResult(), run_id=run_id)

        calls = tracker.get_calls()
        assert len(calls) == 1
        assert calls[0]["model"] == "gpt-4o"
        assert calls[0]["input_tokens"] == 100
        assert calls[0]["output_tokens"] == 50

    def test_on_llm_error_cleans_up(self, tracker):
        """Verify on_llm_error cleans up pending run without recording."""
        from talanton.integrations.langchain_callback import TalantonCallbackHandler
        from uuid import uuid4

        handler = TalantonCallbackHandler(tracker=tracker)
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"kwargs": {"model": "gpt-4o"}},
            prompts=["Hello"],
            run_id=run_id,
        )
        handler.on_llm_error(RuntimeError("API error"), run_id=run_id)

        assert str(run_id) not in handler._pending_runs
        assert len(tracker.get_calls()) == 0


# ─── LiteLLM Callback Tests ──────────────────────────────────────────────


class TestLiteLLMCallback:

    def test_log_success_records_call(self, tracker):
        """Verify LiteLLM callback records call on log_success_event."""
        from talanton.integrations.litellm_callback import TalantonLiteLLMCallback

        callback = TalantonLiteLLMCallback(tracker=tracker)

        class MockUsage:
            prompt_tokens = 250
            completion_tokens = 125

        class MockResponse:
            usage = MockUsage()

        callback.log_success_event(
            kwargs={"model": "gpt-4o"},
            response_obj=MockResponse(),
            start_time=None,
            end_time=None,
        )

        calls = tracker.get_calls()
        assert len(calls) == 1
        assert calls[0]["model"] == "gpt-4o"
        assert calls[0]["input_tokens"] == 250
        assert calls[0]["output_tokens"] == 125

    def test_log_failure_does_not_record(self, tracker):
        """Verify LiteLLM failure callback does not record a call."""
        from talanton.integrations.litellm_callback import TalantonLiteLLMCallback

        callback = TalantonLiteLLMCallback(tracker=tracker)
        callback.log_failure_event(
            kwargs={"model": "gpt-4o"},
            response_obj={"error": "timeout"},
            start_time=None,
            end_time=None,
        )

        calls = tracker.get_calls()
        assert len(calls) == 0
