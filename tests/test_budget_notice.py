"""The [budget] line on tool results (TransitTask.budget_notice in transitbench_env/task.py): the request
interceptor alone on verifiers' own types, then through verifiers' RolloutSession, whose contract checks and tool gate
it must pass.

Needs verifiers (the env extra), like tests/test_env.py.
"""

import asyncio
import time
import tomllib
from pathlib import Path

import pytest

vf = pytest.importorskip("verifiers.v1", reason="needs verifiers (the env extra)")

from verifiers.v1 import graph  # noqa: E402
from verifiers.v1.clients import ModelContext  # noqa: E402
from verifiers.v1.harnesses.utils.compaction import (  # noqa: E402
    TOOL_OUTPUT_MAX_BYTES,
    bound_tool_message,
    truncate_tool_output,
)
from verifiers.v1.session import RolloutLimits, RolloutSession, hook_boundary  # noqa: E402
from verifiers.v1.trace import ModelCall, TraceTask  # noqa: E402
from verifiers.v1.utils.decorators import discover_decorated  # noqa: E402

from transitbench_env import box, task  # noqa: E402
from transitbench_mini import MiniTask, mini_data  # noqa: E402
from transitbench_survey import SurveyConfig, SurveyTaskset  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EVAL_CONFIGS = sorted(CONFIGS.glob("*/*.toml"))


def config_limits(path: Path) -> dict[str, int]:
    """An eval TOML's agent limits, the ones verifiers enforces."""
    (source,) = tomllib.loads(path.read_text())["source"]
    return {k: source["env"]["agent"][k] for k in ("max_turns", "max_total_tokens")}


# The example configs' agent limits (every survey and mini config has the same as its example, test below).
LIMITS = {"survey": config_limits(CONFIGS / "examples" / "survey.toml"),
          "mini": config_limits(CONFIGS / "examples" / "mini.toml")}
PRESET_CONFIGS = [p for p in EVAL_CONFIGS if p.stem.split("-")[0] in LIMITS]
TAIL = ". The session ends at the first limit, and solution/ is graded as it is then."
BASH = vf.Tool(name="bash", description="run a command", parameters={"type": "object"})
SYSTEM, USER = vf.SystemMessage(content="be careful"), vf.UserMessage(content="solve it")


def make_task(preset: str):
    if preset == "survey":
        return list(SurveyTaskset(SurveyConfig(id="transitbench-survey")))[0]
    return MiniTask(mini_data("train", "train-00000000", "pool", "image"))


def make_trace(t, **config) -> vf.Trace:
    return vf.Trace(task=TraceTask(type=type(t).__name__, data=t.data, key=t.key, hash=t.hash), state=vf.State(),
                    agent=vf.AgentInfo(config=vf.AgentConfig(runtime=vf.DockerConfig(), **config), name="test"))


def calls(*ids: str) -> vf.AssistantMessage:
    return vf.AssistantMessage(content="", tool_calls=[
        vf.ToolCall(id=i, name="bash", arguments='{"command": "ls"}') for i in ids])


def result(call_id: str, content="out") -> vf.ToolMessage:
    return vf.ToolMessage(tool_call_id=call_id, content=content, name="bash")


def commit(trace: vf.Trace, messages: list, reply: vf.AssistantMessage, prompt: int, completion: int) -> int:
    """One model turn recorded as the interception server records it: the turn committed, then its call."""
    usage = vf.Usage(prompt_tokens=prompt, completion_tokens=completion)
    node = graph.prepare_turn(trace, messages, [BASH]).commit(vf.Response(
        id="r", created=0, model="m", message=reply, finish_reason="tool_calls", usage=usage))
    trace.calls.append(ModelCall(node=node, usage=usage))
    return node


def two_turns(trace: vf.Trace) -> list:
    """Two committed turns; the second proposes two parallel calls. Returns the harness's next request."""
    first = [SYSTEM, USER]
    commit(trace, first, calls("c1"), prompt=1_000, completion=200)
    second = [*first, calls("c1"), result("c1")]
    commit(trace, second, calls("c2", "c3"), prompt=1_500, completion=300)
    return [*second, calls("c2", "c3"), result("c2", "first"), result("c3", "second")]


# ---- the line --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["survey", "mini"])
def test_line_shows_the_traces_numbers_against_the_agent_limits(preset):
    t = make_task(preset)
    trace = make_trace(t, **LIMITS[preset])
    two_turns(trace)
    # verifiers' own counts: two replies; new input 1,000 + 300 (the second prompt beyond the first call's 1,200)
    # plus 500 generated.
    assert (trace.num_turns, trace.num_total_tokens) == (2, 1_800)
    now = time.time()
    trace.timing.agent.start = now - 1_000
    turns, tokens = LIMITS[preset]["max_turns"], LIMITS[preset]["max_total_tokens"]
    left = "3 h 43 min of 4 h" if preset == "survey" else "43 min of 1 h"
    assert task.budget_line(t.data, trace, now=now) == (
        f"[budget] 2 of {turns} turns used; 1,800 of {tokens:,} tokens used; {left} left{TAIL}")


@pytest.mark.parametrize("path", PRESET_CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_eval_config_sets_the_limits_task_md_states(path):
    """verifiers enforces only the agent config's limits, and TASK.md states the task data's: they must agree."""
    preset = path.stem.split("-")[0]
    t = make_task(preset)
    task_md = {"max_turns": t.data.max_turns, "max_total_tokens": t.data.max_tokens}
    assert config_limits(path) == LIMITS[preset] == task_md


def test_limits_are_the_agent_configs_and_time_falls_back_to_the_task():
    t = make_task("survey")
    trace = make_trace(t, max_turns=30, max_total_tokens=100_000, timeout={"rollout": 7_200})
    trace.timing.agent.start = 1_000.0
    assert task.budget_line(t.data, trace, now=1_000.0) == (
        f"[budget] 0 of 30 turns used; 0 of 100,000 tokens used; 2 h 00 min of 2 h left{TAIL}")
    # no limit in the config is no limit in verifiers (RolloutLimits): the line states none and never says "last"
    trace = make_trace(t)
    trace.timing.agent.start = 1_000.0
    assert task.budget_line(t.data, trace, now=1_000.0 + 14_400 - 59) == (
        f"[budget] 0 turns used; 0 tokens used; 0 min of 4 h left{TAIL}")
    assert task.budget_line(t.data, trace, now=1_000.0 + 20_000).endswith(f"; 0 min of 4 h left{TAIL}")
    # one turn under the task data's cap, which nothing enforces here: no "last turn" note
    t = make_task("mini")
    trace = make_trace(t, max_total_tokens=LIMITS["mini"]["max_total_tokens"])
    messages = [SYSTEM, USER]
    for k in range(1, t.data.max_turns):
        commit(trace, messages, calls(f"c{k}"), prompt=100 * k, completion=10)
        messages = [*messages, calls(f"c{k}"), result(f"c{k}")]
    assert trace.num_turns == t.data.max_turns - 1
    assert task.budget_line(t.data, trace) == (
        f"[budget] {t.data.max_turns - 1} turns used; {trace.num_total_tokens:,} of 32,768 tokens used{TAIL}")


def test_time_left_is_left_out_before_the_agent_stage_starts():
    t = make_task("mini")
    trace = make_trace(t, **LIMITS["mini"])
    assert trace.timing.agent.start == 0
    assert task.budget_line(t.data, trace) == f"[budget] 0 of 12 turns used; 0 of 32,768 tokens used{TAIL}"


@pytest.mark.parametrize("preset", ["survey", "mini"])
def test_the_line_one_turn_under_the_limit_says_the_next_reply_is_the_last(preset):
    """verifiers refuses the call at num_turns == max_turns before the hooks run, so max_turns - 1 is the highest count
    the agent sees, and its reply to it is its last."""
    t = make_task(preset)
    turns = LIMITS[preset]["max_turns"]
    trace = make_trace(t, **LIMITS[preset])
    messages = [SYSTEM, USER]
    for k in range(1, turns):
        assert "last turn" not in task.budget_line(t.data, trace)
        commit(trace, messages, calls(f"c{k}"), prompt=100 * k, completion=10)
        messages = [*messages, calls(f"c{k}"), result(f"c{k}")]
    assert trace.num_turns == turns - 1
    line = task.budget_line(t.data, trace)
    assert line == (f"[budget] {turns - 1} of {turns} turns used; {trace.num_total_tokens:,} of "
                    f"{LIMITS[preset]['max_total_tokens']:,} tokens used. Your next reply is your last turn: its "
                    f"commands run, but you will not see their output{TAIL}")
    assert "\n" not in line


@pytest.mark.parametrize("preset", ["survey", "mini"])
def test_task_md_says_tool_results_end_with_the_line(preset):
    prompt = make_task(preset).data.prompt
    assert prompt == box.render_task_md(task.caps(make_task(preset).data))
    assert ("Every tool result ends with a `[budget]` line: the turns and tokens\n  used so far and the time "
            "left. When the turn or token limit ends the session, the commands of your last turn still\n  run, but you do "
            "not see their output. A reply that calls no tool also ends the session.") in prompt
    assert task.BUDGET == "[budget] "


# ---- the interceptor --------------------------------------------------------------------------------------------------

def intercept(t, trace, messages, tools=(BASH,)):
    request = vf.Request(messages=messages, tools=list(tools) or None)
    return request, t.budget_notice(request.model_copy(deep=True), trace)


def test_the_rollout_collects_it_as_a_request_interceptor():
    """As verifiers' Rollout does (rollout.py): decorated methods, sorted, typed by their Request parameter."""
    for preset in ("survey", "mini"):
        hooks = discover_decorated(make_task(preset), "intercept")
        assert [h.__name__ for h in hooks] == ["budget_notice"]
        assert hook_boundary(hooks[0], allow_trace=False) is vf.Request


def test_appends_to_the_last_tool_result_only():
    t = make_task("survey")
    trace = make_trace(t, **LIMITS["survey"])
    messages = two_turns(trace)
    before, after = intercept(t, trace, messages)
    assert len(after.messages) == len(before.messages) and after.tools == before.tools
    assert after.messages[:-1] == before.messages[:-1]
    last = after.messages[-1]
    assert isinstance(last, vf.ToolMessage) and (last.tool_call_id, last.name) == ("c3", "bash")
    line = task.budget_line(t.data, trace)
    assert last.content == f"second\n{line}" and line.startswith("[budget] 2 of 60 turns used; 1,800 of 200,000")
    # A result that ends with a newline gets no blank line.
    assert intercept(t, trace, [*messages[:-1], result("c3", "second\n")])[1].messages[-1].content == (
        f"second\n{line}")


@pytest.mark.parametrize("last", [USER, calls("c9"), SYSTEM])
def test_leaves_requests_that_do_not_end_with_a_tool_result(last):
    t = make_task("mini")
    trace = make_trace(t)
    assert intercept(t, trace, [SYSTEM, USER, calls("c1"), result("c1"), last])[1] is None
    assert t.budget_notice(vf.Request(messages=[]), trace) is None


@pytest.mark.parametrize("content", ["", []])
def test_leaves_empty_results(content):
    """verifiers' tool gate probes each proposed call with an empty result; rewriting it would block the call."""
    t = make_task("mini")
    assert intercept(t, make_trace(t), [SYSTEM, USER, calls("c1"), result("c1", content)])[1] is None


def test_leaves_a_result_that_already_ends_with_a_budget_line():
    t = make_task("mini")
    trace = make_trace(t)
    _, once = intercept(t, trace, [SYSTEM, USER, calls("c1"), result("c1")])
    assert intercept(t, trace, once.messages)[1] is None


def test_list_content_gets_a_text_part():
    t = make_task("mini")
    trace = make_trace(t, **LIMITS["mini"])
    parts = [vf.TextContentPart(text="plot:"), vf.ImageUrlContentPart(image_url={"url": "data:image/png;base64,AA=="})]
    before, after = intercept(t, trace, [SYSTEM, USER, calls("c1"), result("c1", parts)])
    content = after.messages[-1].content
    assert content[:2] == parts and content[2] == vf.TextContentPart(text="\n" + task.budget_line(t.data, trace))
    assert after.messages[:-1] == before.messages[:-1] and after.tools == before.tools
    assert intercept(t, trace, after.messages)[1] is None


# ---- long results -----------------------------------------------------------------------------------------------------

LINE = task.budget_line(make_task("survey").data, make_trace(make_task("survey")))


def test_a_short_text_is_appended_unchanged():
    assert task.append_line("abc", LINE) == f"abc\n{LINE}"
    assert task.append_line("", LINE) == LINE


def test_a_truncated_result_gives_up_bytes_before_its_cut_and_is_not_cut_again():
    text = truncate_tool_output("".join(f"line {i}: {'x' * 60}\n" for i in range(4_000)))
    assert len(text.encode()) > TOOL_OUTPUT_MAX_BYTES  # verifiers' own head and tail pass its cap
    (n,) = [int(m.group(1)) for m in task.CUT.finditer(text)]
    out = task.append_line(text, LINE)
    assert len(out.encode()) <= TOOL_OUTPUT_MAX_BYTES and out.endswith("\n" + LINE)
    assert bound_tool_message({"role": "tool", "content": out})["content"] == out  # verifiers leaves it as it is
    (cut,) = list(task.CUT.finditer(out))
    head, rest = out[:cut.start()], out[cut.end():]
    original = list(task.CUT.finditer(text))[0]
    assert text.endswith("\n") and text[:original.start()].startswith(head) and rest == text[original.end():] + LINE
    dropped = len(text[:original.start()].encode()) - len(head.encode())
    assert int(cut.group(1)) == n + dropped > n


def test_a_text_just_under_the_cap_gets_a_cut_of_its_own():
    text = "y" * (TOOL_OUTPUT_MAX_BYTES - 10)
    out = task.append_line(text, LINE)
    assert len(out.encode()) <= TOOL_OUTPUT_MAX_BYTES and out.endswith("\n" + LINE)
    (cut,) = list(task.CUT.finditer(out))
    kept = len(out[:cut.start()]) + len(out[cut.end():]) - len("\n" + LINE)
    assert int(cut.group(1)) == len(text) - kept


def test_a_text_that_cannot_fit_is_left_alone():
    assert task.append_line("z" * 10, LINE, limit=20) is None


# ---- through verifiers' RolloutSession --------------------------------------------------------------------------------

def session(t, trace, gates_tools: bool = True, limits: RolloutLimits | None = None) -> RolloutSession:
    """A session as verifiers' Rollout builds one, with this task's interceptors (gates_tools is the harness's
    SUPPORTS_TOOL_INTERCEPTION; limits come from the agent config)."""
    hooks = [fn for fn in discover_decorated(t, "intercept") if hook_boundary(fn, allow_trace=False) is vf.Request]
    return RolloutSession(ctx=ModelContext(model="m", client=vf.EvalClientConfig()), trace=trace,
                          request_interceptors=hooks, gates_tools=gates_tools, limits=limits or RolloutLimits())


@pytest.mark.parametrize("gates_tools", [True, False])
def test_session_allows_every_proposed_call_through_the_tool_gate(gates_tools):
    """The server's tool gate runs the request interceptors on an empty probe result per proposed call before the
    call runs; any rewrite there would block the call (gates_tools) or end the rollout (not gates_tools)."""
    t = make_task("survey")
    trace = make_trace(t, **LIMITS["survey"])
    s = session(t, trace, gates_tools=gates_tools)
    first = [SYSTEM, USER]
    node = commit(trace, first, calls("c1", "c2"), prompt=1_000, completion=200)
    assert asyncio.run(s.gate_tool_calls(node)) is None
    assert len(s.tool_decisions) == 2 and set(s.tool_decisions.values()) == {None}
    assert trace.request_rewrites == [] and trace.stop_condition is None


def test_session_rewrites_parallel_results_and_pins_them():
    t = make_task("survey")
    trace = make_trace(t, **LIMITS["survey"])
    trace.timing.agent.start = time.time() - 600
    s = session(t, trace)
    request = vf.Request(messages=two_turns(trace), tools=[BASH])
    asyncio.run(s.gate_tool_calls(len(trace.nodes) - 1))

    rewritten, records, stopped = asyncio.run(s.rewrite_request(request))
    assert stopped is None and [r.handler for r in records] == ["budget_notice", "budget_notice"]
    assert len(rewritten.messages) == len(request.messages) and rewritten.tools == request.tools
    assert rewritten.messages[:-2] == request.messages[:-2]
    for before, after in zip(request.messages[-2:], rewritten.messages[-2:]):
        assert after.tool_call_id == before.tool_call_id and after.name == before.name
        head, line = after.content.split("\n")
        assert head == before.content and line.startswith("[budget] 2 of 60 turns used; 1,800 of 200,000 tokens used; ")
        assert line.endswith(f" of 4 h left{TAIL}")

    # The harness re-sends its own copies (a retry): the pinned results come back, with no second line.
    again, records, _ = asyncio.run(s.rewrite_request(request))
    assert again == rewritten and records == []

    # The next turn: only the new result gets a line, with the new counts; the earlier ones keep theirs.
    commit(trace, rewritten.messages, calls("c4"), prompt=2_000, completion=100)
    later = vf.Request(messages=[*request.messages, calls("c4"), result("c4", "third")], tools=[BASH])
    after, records, _ = asyncio.run(s.rewrite_request(later))
    assert len(records) == 1 and after.messages[:-2] == rewritten.messages
    # New input 1,000 + 300 + 200 (the third prompt beyond the second call's 1,800) plus 600 generated.
    assert trace.num_turns == 3 and trace.num_total_tokens == 2_100
    assert after.messages[-1].content.startswith("third\n[budget] 3 of 60 turns used; 2,100 of 200,000 tokens used")


@pytest.mark.parametrize("preset", ["survey", "mini"])
def test_session_shows_the_last_turn_note_once_and_then_refuses(preset):
    """A whole session at the preset's limits, as the interception server serves it: refused(), then
    rewrite_request, then the turn and its tool gate. Only the line before the last reply has the last-turn note, the
    highest count shown is max_turns - 1, and the request after the last reply is refused with max_turns."""
    t = make_task(preset)
    turns = LIMITS[preset]["max_turns"]
    trace = make_trace(t, **LIMITS[preset])
    s = session(t, trace, limits=RolloutLimits(**LIMITS[preset]))

    async def drive():
        harness, lines = [SYSTEM, USER], []
        while (stop := await s.refused()) is None:
            request, _, stopped = await s.rewrite_request(vf.Request(messages=harness, tools=[BASH]))
            assert stopped is None
            if isinstance(request.messages[-1], vf.ToolMessage):
                lines.append(request.messages[-1].content.rsplit("\n", 1)[-1])
            k = trace.num_turns + 1
            node = commit(trace, request.messages, calls(f"c{k}"), prompt=50 * k, completion=10)
            assert await s.gate_tool_calls(node) is None
            harness = [*harness, calls(f"c{k}"), result(f"c{k}", f"out {k}")]
        return lines, stop

    lines, stop = asyncio.run(drive())
    assert stop == "max_turns" and trace.num_turns == turns and len(lines) == turns - 1
    assert [line.split(" turns used")[0] for line in lines] == [f"[budget] {k} of {turns}" for k in range(1, turns)]
    assert [("last turn" in line) for line in lines] == [False] * (turns - 2) + [True]
