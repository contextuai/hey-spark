"""P3.4: the start_project flow (spark/tools/project.py) and its SDK tool.

Safety: nothing here may reach GitHub or open a window. An autouse fixture lets only ``git``
through ``subprocess.run`` and makes ``subprocess.Popen`` from the project module fail; git runs
for real, but only inside pytest's tmp dirs and with an isolated git config.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from spark.config import ProjectsConfig, Settings
from spark.tools import project as project_mod
from spark.tools.project import (
    ClaudeCodeLauncher,
    GhCliRepoHost,
    GitHubIssueTracker,
    NoTracker,
    ProjectError,
    ProjectService,
    RepoInfo,
    StepFailed,
    TaskInfo,
    plan_project,
    slugify,
    start_project,
    summarize_start_project,
)
from tests.fakes import FakeClock

NOT_SPEAKABLE = re.compile(r"https?://|www\.|github\.com|#\d|[{}\[\]`*_]|\\|/")
FAKE_GH = "C:/fake/bin/gh.exe"
FAKE_CLAUDE = "C:/fake/bin/claude.exe"


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def only_local_git(monkeypatch, tmp_path_factory):
    """Isolate git from the user's config, hide the real gh/claude, and refuse to run them."""
    gitconfig = tmp_path_factory.mktemp("gitcfg") / "gitconfig"
    gitconfig.write_text(
        "[user]\n\tname = Spark Test\n\temail = spark-test@example.invalid\n"
        "[init]\n\tdefaultBranch = main\n[commit]\n\tgpgsign = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)

    # Only the names matter: project.py resolves gh/claude with shutil.which before running
    # anything, so hiding them means a missed fake fails with "isn't installed" instead of
    # reaching GitHub. The run guard is a second fence. Other callers are left alone.
    forbidden = {"gh", "claude"}
    real_run, real_which = subprocess.run, project_mod.shutil.which

    def guarded_which(name, *args, **kwargs):
        if Path(str(name)).stem.lower() in forbidden:
            return None
        return real_which(name, *args, **kwargs)

    def guarded_run(argv, *args, **kwargs):
        exe = Path(str(next(iter(argv)))).stem.lower() if not isinstance(argv, str) else argv
        if exe in forbidden:
            raise AssertionError(f"test tried to run the real {exe!r}")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(project_mod.shutil, "which", guarded_which)
    monkeypatch.setattr(project_mod.subprocess, "run", guarded_run)


class FakeHost:
    def __init__(self, error: Exception | None = None, owner: str = "me") -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error
        self.owner = owner

    def create_repo(
        self, name: str, *, private: bool, description: str, source: Path | None = None
    ) -> RepoInfo:
        self.calls.append(
            {"name": name, "private": private, "description": description, "source": source}
        )
        if self.error is not None:
            raise self.error
        full = name if "/" in name else f"{self.owner}/{name}"
        return RepoInfo(full_name=full, url=f"https://github.com/{full}")


class FakeTracker:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def create_task(
        self, title: str, body: str, *, repo: RepoInfo | None = None, cwd: Path | None = None
    ) -> TaskInfo:
        self.calls.append({"title": title, "body": body, "repo": repo, "cwd": cwd})
        if self.error is not None:
            raise self.error
        return TaskInfo(ref="#1", url=f"https://github.com/{repo.full_name}/issues/1")


class FakeLauncher:
    def __init__(self, error: Exception | None = None) -> None:
        self.opened: list[Path] = []
        self.error = error

    def open(self, path: Path) -> None:
        self.opened.append(path)
        if self.error is not None:
            raise self.error


def cfg(**kw: Any) -> ProjectsConfig:
    return ProjectsConfig(**kw)


def service(
    root: Path,
    *,
    host: FakeHost | None = None,
    tracker: FakeTracker | None = None,
    launcher: FakeLauncher | None = None,
    **settings: Any,
) -> ProjectService:
    return ProjectService(
        cfg(**settings),
        root=root,
        host=host if host is not None else FakeHost(),
        tracker=tracker if tracker is not None else FakeTracker(),
        launcher=launcher if launcher is not None else FakeLauncher(),
    )


def git_log(path: Path) -> str:
    return subprocess.run(
        ["git", "log", "--format=%s"],
        cwd=path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout


def entries(path: Path) -> list[Path]:
    return list(path.iterdir())


def assert_speakable(text: str) -> None:
    assert text.strip()
    assert not NOT_SPEAKABLE.search(text), text


# --------------------------------------------------------------------------- names and plans


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("weather-bot", "weather-bot"),
        ("Weather Bot", "weather-bot"),
        ("  My_App  2 ", "my-app-2"),
        ("weather.bot!", "weather-bot"),
        ("Console", "console"),
    ],
)
def test_slugify(name: str, slug: str) -> None:
    assert slugify(name) == slug


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        "!!!",
        "../x",
        "..",
        "a/b",
        "a\\b",
        "C:evil",
        "CON",
        "con",
        "Nul",
        "aux",
        "LPT1",
        "com3",
        "x" * 65,
    ],
)
def test_bad_names_are_rejected_with_speakable_reason(name: str) -> None:
    with pytest.raises(ProjectError) as err:
        slugify(name)
    assert_speakable(str(err.value))


def test_plan_is_under_root_with_config_defaults(tmp_path: Path) -> None:
    plan = plan_project("Weather Bot", cfg(), root=tmp_path)
    assert plan.slug == "weather-bot"
    assert plan.path == (tmp_path / "weather-bot").resolve()
    assert plan.path.parent == tmp_path.resolve()
    assert plan.create_repo and plan.private and plan.tracker == "github_issue"
    assert plan.open_claude_code is False
    assert plan.repo_name == "weather-bot"


def test_plan_overrides_and_owner(tmp_path: Path) -> None:
    plan = plan_project(
        "weather-bot",
        cfg(github_owner="acme", open_claude_code=False),
        root=tmp_path,
        private=False,
        open_claude_code=True,
    )
    assert plan.private is False and plan.open_claude_code is True
    assert plan.repo_name == "acme/weather-bot"
    local = plan_project("weather-bot", cfg(repo_host="none"), root=tmp_path)
    assert local.create_repo is False and local.tracker == "none"


def test_plan_has_no_side_effects(tmp_path: Path) -> None:
    root = tmp_path / "projects"  # doesn't exist yet and must not be created
    plan_project("weather-bot", cfg(), root=root)
    summarize_start_project({"name": "weather-bot"}, cfg(), root=root)
    assert not root.exists()
    plan_project("weather-bot", cfg(), root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_plan_refuses_existing_non_empty_folder(tmp_path: Path) -> None:
    (tmp_path / "weather-bot").mkdir()
    plan_project("weather-bot", cfg(), root=tmp_path)  # empty folder is fine
    (tmp_path / "weather-bot" / "notes.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(ProjectError, match="already something called weather-bot"):
        plan_project("weather-bot", cfg(), root=tmp_path)
    (tmp_path / "file-bot").write_text("x", encoding="utf-8")
    with pytest.raises(ProjectError):
        plan_project("file-bot", cfg(), root=tmp_path)


def test_confirmation_summary(tmp_path: Path) -> None:
    text = summarize_start_project({"name": "Weather Bot"}, cfg(), root=tmp_path)
    assert text == (
        "I'll create a project called weather-bot: a local folder and git repo, a private "
        "GitHub repo and a tracking issue. Should I go ahead?"
    )
    text = summarize_start_project(
        {"name": "weather-bot", "private": False, "open_claude_code": True}, cfg(), root=tmp_path
    )
    assert "a public GitHub repo" in text and "Claude Code window" in text
    local = summarize_start_project({"name": "x"}, cfg(repo_host="none"), root=tmp_path)
    assert "GitHub" not in local and "issue" not in local
    bad = summarize_start_project({"name": "../x"}, cfg(), root=tmp_path)
    assert bad.endswith("Should I go ahead?")
    for t in (text, local):
        assert_speakable(t)


@pytest.mark.parametrize(
    "name",
    [
        "https://evil.example.com/x",
        "www.evil.example:1",
        "see deadbeefcafebabe1234:x",
        "C:\\Users\\me\\secret",
        "mail me@example.com/x",
        "**bold**/x",
        "x" * 65,
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789:x",
    ],
)
def test_invalid_name_fallback_never_speaks_urls_ids_or_markup(tmp_path: Path, name) -> None:
    """The fallback line for a name that fails validation goes through the same
    ``speakable_value`` filter as the generic confirmation summaries."""
    with pytest.raises(ProjectError):
        plan_project(name, cfg(), root=tmp_path)
    line = summarize_start_project({"name": name}, cfg(), root=tmp_path)
    assert line == "I'll try to start a new project. Should I go ahead?"
    assert_speakable(line)


def test_invalid_but_speakable_name_is_still_said(tmp_path: Path) -> None:
    line = summarize_start_project({"name": "CON"}, cfg(), root=tmp_path)  # reserved on Windows
    assert line == "I'll try to start a project called CON. Should I go ahead?"
    assert summarize_start_project({}, cfg(), root=tmp_path) == (
        "I'll try to start a new project. Should I go ahead?"
    )


# --------------------------------------------------------------------------- the flow


def test_full_flow_with_fakes(tmp_path: Path) -> None:
    host, tracker, launcher = FakeHost(), FakeTracker(), FakeLauncher()
    svc = service(tmp_path, host=host, tracker=tracker, launcher=launcher)
    plan = svc.plan({"name": "Weather Bot"})
    result = svc.run(plan)

    path = tmp_path / "weather-bot"
    assert (path / ".git").is_dir()
    assert (path / "README.md").read_text(encoding="utf-8").startswith("# weather-bot")
    assert ".env" in (path / ".gitignore").read_text(encoding="utf-8")
    assert git_log(path).strip() == "Initial commit"

    assert host.calls == [
        {
            "name": "weather-bot",
            "private": True,
            "description": "Weather Bot, started with Spark.",
            "source": path.resolve(),
        }
    ]
    assert len(tracker.calls) == 1
    call = tracker.calls[0]
    assert call["title"] == "Set up weather-bot"
    assert call["repo"].full_name == "me/weather-bot"
    assert call["cwd"] == path.resolve()
    assert launcher.opened == []  # off by default

    assert result.ok
    assert [s.key for s in result.steps] == ["folder", "git", "commit", "repo", "task"]
    summary = result.summary()
    assert summary == (
        "Done. weather-bot is set up: a local folder with a first commit, a private GitHub repo "
        "and a tracking issue. Nothing has been pushed yet."
    )
    assert_speakable(summary)


def test_flow_opens_claude_code_and_uses_owner(tmp_path: Path) -> None:
    host, launcher = FakeHost(), FakeLauncher()
    svc = service(tmp_path, host=host, launcher=launcher, github_owner="acme")
    result = svc.run(svc.plan({"name": "weather-bot", "open_claude_code": True}))
    assert result.ok
    assert host.calls[0]["name"] == "acme/weather-bot"
    assert launcher.opened == [(tmp_path / "weather-bot").resolve()]
    assert "Claude Code is open" in result.summary()


def test_local_only_flow(tmp_path: Path) -> None:
    host, tracker = FakeHost(), FakeTracker()
    svc = service(tmp_path, host=host, tracker=tracker, repo_host="none")
    result = svc.run(svc.plan({"name": "scratch"}))
    assert result.ok and host.calls == [] and tracker.calls == []
    assert result.summary() == "Done. scratch is set up: a local folder with a first commit."


def test_declined_confirmation_creates_nothing(tmp_path: Path) -> None:
    """A "no" means start_project never runs: the plan and the spoken line are computed, but no
    folder, repo, issue or window appears."""
    from spark.tools.server import spark_confirmation_summarizers

    root = tmp_path / "projects"
    host, tracker, launcher = FakeHost(), FakeTracker(), FakeLauncher()
    svc = service(root, host=host, tracker=tracker, launcher=launcher)
    summarizers = spark_confirmation_summarizers(svc)
    line = summarizers["mcp__spark__start_project"]({"name": "weather-bot"})
    assert "weather-bot" in line and line.endswith("Should I go ahead?")
    svc.plan({"name": "weather-bot"})
    # the user says "no": start_project is not called
    assert not root.exists()
    assert list(tmp_path.iterdir()) == []
    assert host.calls == tracker.calls == [] and launcher.opened == []


async def test_side_effecting_tool_is_never_pre_approved(tmp_path: Path) -> None:
    """Until P3.1's hook asks, the Phase 2 gate denies start_project; either way it is not in
    allowed_tools, so the CLI can't run it without the hook's say-so."""
    from spark.brain.mcp_config import McpSetup
    from spark.brain.tool_gate import make_tool_gate
    from spark.tools.server import SERVER_NAME, build_spark_server
    from spark.tools.timers import TimerService

    timers = TimerService(clock=FakeClock())
    try:
        server, read_only, side = build_spark_server(timers, projects=service(tmp_path))
        setup = McpSetup()
        setup.add_sdk_server(SERVER_NAME, server, read_only=read_only, side_effecting=side)
        assert "mcp__spark__start_project" not in setup.allowed_tools
        assert "mcp__spark__start_project" in setup.side_effecting
        gate = make_tool_gate(setup.policy)
        out = await gate(
            {"tool_name": "mcp__spark__start_project", "tool_input": {"name": "x"}}, "t1", None
        )
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert entries(tmp_path) == []
    finally:
        await timers.close()


def test_repo_failure_midway_is_reported(tmp_path: Path) -> None:
    host = FakeHost(error=StepFailed("the GitHub command line tool isn't logged in"))
    tracker, launcher = FakeTracker(), FakeLauncher()
    svc = service(tmp_path, host=host, tracker=tracker, launcher=launcher)
    result = svc.run(svc.plan({"name": "weather-bot", "open_claude_code": True}))

    assert not result.ok
    assert (tmp_path / "weather-bot" / ".git").is_dir()  # local folder is kept
    assert tracker.calls == []  # no repo to put the issue in
    assert launcher.opened == [(tmp_path / "weather-bot").resolve()]  # independent step
    summary = result.summary()
    assert summary == (
        "For weather-bot, I set up the local folder, the git repo, the first commit and the "
        "Claude Code window. The GitHub repo failed because the GitHub command line tool isn't "
        "logged in. I didn't get to the tracking issue. I left the local folder in place."
    )
    assert_speakable(summary)


def test_repo_timeout_warns_remote_might_exist(tmp_path: Path) -> None:
    host = FakeHost(error=StepFailed("it timed out", remote_created=None))
    svc = service(tmp_path, host=host)
    summary = svc.run(svc.plan({"name": "weather-bot"})).summary()
    assert "might exist" in summary
    assert_speakable(summary)


def test_repo_created_but_not_linked_is_said(tmp_path: Path) -> None:
    host = FakeHost(error=StepFailed("linking failed", remote_created=True))
    svc = service(tmp_path, host=host)
    summary = svc.run(svc.plan({"name": "weather-bot"})).summary()
    assert "the repo does exist on GitHub" in summary


def test_tracker_failure_after_repo_mentions_the_repo(tmp_path: Path) -> None:
    tracker = FakeTracker(error=StepFailed("GitHub couldn't be reached"))
    svc = service(tmp_path, tracker=tracker)
    result = svc.run(svc.plan({"name": "weather-bot"}))
    summary = result.summary()
    assert not result.ok
    assert "the GitHub repo" in summary.split(".")[0]  # listed as done
    assert "The tracking issue failed because GitHub couldn't be reached." in summary
    assert_speakable(summary)


def test_unexpected_host_crash_is_contained(tmp_path: Path) -> None:
    svc = service(tmp_path, host=FakeHost(error=RuntimeError("boom")))
    summary = svc.run(svc.plan({"name": "weather-bot"})).summary()
    assert "something went wrong" in summary and "might exist" in summary


def test_run_refuses_folder_that_filled_up_after_planning(tmp_path: Path) -> None:
    host = FakeHost()
    svc = service(tmp_path, host=host)
    plan = svc.plan({"name": "weather-bot"})
    (tmp_path / "weather-bot").mkdir()
    (tmp_path / "weather-bot" / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(ProjectError):
        svc.run(plan)
    assert host.calls == []
    assert not (tmp_path / "weather-bot" / ".git").exists()


def test_missing_git_skips_everything_after_the_folder(tmp_path: Path, monkeypatch) -> None:
    host = FakeHost()
    plan = plan_project("weather-bot", cfg(), root=tmp_path)
    result = start_project(plan, host, FakeTracker(), None, git_path="no-such-git-exe")
    assert host.calls == []
    assert [(s.key, s.ok, s.skipped) for s in result.steps] == [
        ("folder", True, False),
        ("git", False, False),
        ("commit", False, True),
        ("repo", False, True),
        ("task", False, True),
    ]
    assert "git isn't installed" in result.summary()


# --------------------------------------------------------------- real backends, argv only


class Recorder:
    def __init__(
        self, stdout: str = "", stderr: str = "", code: int = 0, exc: Exception | None = None
    ):
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.stdout, self.stderr, self.code, self.exc = stdout, stderr, code, exc

    def __call__(self, argv, *args, **kwargs):
        self.calls.append((list(argv), kwargs))
        if self.exc is not None:
            raise self.exc
        return subprocess.CompletedProcess(argv, self.code, self.stdout, self.stderr)


@pytest.fixture
def fake_which(monkeypatch):
    table = {"gh": FAKE_GH, "claude": FAKE_CLAUDE}
    monkeypatch.setattr(project_mod.shutil, "which", lambda name: table.get(name))


def test_gh_repo_create_argv(tmp_path: Path, monkeypatch, fake_which) -> None:
    rec = Recorder(stdout="https://github.com/acme/weather-bot\n")
    monkeypatch.setattr(project_mod.subprocess, "run", rec)
    host = GhCliRepoHost("acme", timeout_s=12)
    info = host.create_repo("weather-bot", private=True, description="A bot.", source=tmp_path)
    argv, kwargs = rec.calls[0]
    assert argv == [
        FAKE_GH, "repo", "create", "acme/weather-bot", "--private",
        "--description", "A bot.", "--source", str(tmp_path), "--remote", "origin",
    ]  # fmt: skip
    assert "--push" not in argv
    assert kwargs["timeout"] == 12 and kwargs["cwd"] == tmp_path
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert info.full_name == "acme/weather-bot"

    rec.calls.clear()
    GhCliRepoHost(None).create_repo("weather-bot", private=False, description="d")
    argv, _ = rec.calls[0]
    assert argv == [FAKE_GH, "repo", "create", "weather-bot", "--public", "--description", "d"]


def test_gh_repo_create_failures(tmp_path: Path, monkeypatch, fake_which) -> None:
    host = GhCliRepoHost("acme")
    monkeypatch.setattr(
        project_mod.subprocess, "run", Recorder(code=1, stderr="GraphQL: Name already exists")
    )
    with pytest.raises(StepFailed) as err:
        host.create_repo("weather-bot", private=True, description="d", source=tmp_path)
    assert "already exists" in str(err.value) and err.value.remote_created is False

    monkeypatch.setattr(
        project_mod.subprocess,
        "run",
        Recorder(code=1, stdout="https://github.com/acme/weather-bot\n", stderr="remote failed"),
    )
    with pytest.raises(StepFailed) as err:
        host.create_repo("weather-bot", private=True, description="d", source=tmp_path)
    assert err.value.remote_created is True

    monkeypatch.setattr(
        project_mod.subprocess, "run", Recorder(exc=subprocess.TimeoutExpired(["gh"], 60))
    )
    with pytest.raises(StepFailed) as err:
        host.create_repo("weather-bot", private=True, description="d", source=tmp_path)
    assert err.value.remote_created is None

    monkeypatch.setattr(project_mod.shutil, "which", lambda name: None)
    with pytest.raises(StepFailed, match="isn't installed"):
        host.create_repo("weather-bot", private=True, description="d")


def test_gh_issue_create_argv(tmp_path: Path, monkeypatch, fake_which) -> None:
    rec = Recorder(stdout="https://github.com/acme/weather-bot/issues/3\n")
    monkeypatch.setattr(project_mod.subprocess, "run", rec)
    task = GitHubIssueTracker(timeout_s=9).create_task(
        "Set up weather-bot", "Body.", repo=RepoInfo("acme/weather-bot"), cwd=tmp_path
    )
    argv, kwargs = rec.calls[0]
    assert argv == [
        FAKE_GH, "issue", "create", "--title", "Set up weather-bot", "--body", "Body.",
        "--repo", "acme/weather-bot",
    ]  # fmt: skip
    assert kwargs["timeout"] == 9 and kwargs["cwd"] == tmp_path
    assert task.ref == "#3"
    with pytest.raises(StepFailed, match="no GitHub repo"):
        GitHubIssueTracker().create_task("t", "b", repo=None)
    with pytest.raises(StepFailed):
        NoTracker().create_task("t", "b", repo=RepoInfo("a/b"))


def test_claude_code_launcher_argv(tmp_path: Path, fake_which, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-for-the-child")
    calls: list[tuple[list[str], dict[str, Any]]] = []
    launcher = ClaudeCodeLauncher(
        platform="win32", popen=lambda argv, **kw: calls.append((argv, kw))
    )
    launcher.open(tmp_path)
    [(argv, kw)] = calls
    assert argv == [FAKE_CLAUDE]
    assert kw["cwd"] == str(tmp_path) and kw["creationflags"] == 0x00000010
    assert "ANTHROPIC_API_KEY" not in kw["env"]  # Claude Code uses its own login
    with pytest.raises(StepFailed, match="only works on Windows"):
        ClaudeCodeLauncher(platform="linux", popen=lambda *a, **k: None).open(tmp_path)


def test_claude_code_launcher_macos_opens_terminal(tmp_path: Path, fake_which) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    folder = tmp_path / 'my "odd" app'
    ClaudeCodeLauncher(platform="darwin", popen=lambda argv, **kw: calls.append((argv, kw))).open(
        folder
    )
    [(argv, kw)] = calls
    assert argv[:2] == ["osascript", "-e"]
    script = argv[2]
    assert script.startswith('tell application "Terminal" to do script "cd ')
    # the shell command is single-quoted for sh, then its quotes escaped for AppleScript
    assert 'my \\"odd\\" app' in script and script.endswith(f'&& {FAKE_CLAUDE}"')
    assert argv[3:] == ["-e", 'tell application "Terminal" to activate']
    assert "creationflags" not in kw


def test_claude_code_launcher_default_uses_popen(tmp_path: Path, monkeypatch, fake_which) -> None:
    rec: list[Any] = []
    monkeypatch.setattr(project_mod.subprocess, "Popen", lambda argv, **kw: rec.append(argv))
    ClaudeCodeLauncher(platform="win32").open(tmp_path)
    assert rec == [[FAKE_CLAUDE]]


def test_service_from_config_picks_real_backends(tmp_path: Path) -> None:
    svc = ProjectService.from_config(cfg(github_owner="acme", command_timeout_s=5), root=tmp_path)
    assert isinstance(svc.host, GhCliRepoHost) and svc.host.owner == "acme"
    assert svc.host.timeout_s == 5
    assert isinstance(svc.tracker, GitHubIssueTracker)
    assert isinstance(svc.launcher, ClaudeCodeLauncher)
    local = ProjectService.from_config(cfg(repo_host="none", tracker="none"), root=tmp_path)
    assert local.host is None and isinstance(local.tracker, NoTracker)


def test_service_from_settings_resolves_root(tmp_path: Path) -> None:
    s = Settings(config_dir=tmp_path, projects={"root_dir": "work/projects"})
    assert ProjectService.from_settings(s).root == (tmp_path / "work" / "projects").resolve()
    home = ProjectService.from_settings(Settings(config_dir=tmp_path))
    assert home.root == (Path(os.path.expanduser("~")) / "Projects").resolve()


@pytest.mark.parametrize(
    "bad",
    [
        {"repo_host": "gitlab"},
        {"tracker": "jira"},
        {"github_owner": "a/b"},
        {"command_timeout_s": 0},
    ],
)
def test_projects_config_validation(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ProjectsConfig(**bad)


# --------------------------------------------------------------------------- SDK tool


async def test_sdk_tool_handler(tmp_path: Path) -> None:
    from spark.tools.server import build_spark_server, build_spark_tools
    from spark.tools.timers import TimerService

    timers = TimerService(clock=FakeClock())
    host = FakeHost()
    svc = service(tmp_path, host=host)
    try:
        _, read_only, side = build_spark_server(timers, projects=svc)
        assert side == ["mcp__spark__start_project"]
        assert "mcp__spark__start_project" not in read_only
        tools = {t.name: t for t in build_spark_tools(timers, projects=svc)}
        tool = tools["start_project"]
        wire = tool.annotations.model_dump(by_alias=True, exclude_none=True)
        assert wire["readOnlyHint"] is False and wire["openWorldHint"] is True
        assert tool.input_schema["required"] == ["name"]

        out = await tool.handler({"name": "weather bot"})
        assert not out.get("is_error")
        text = out["content"][0]["text"]
        assert text.startswith("Done. weather-bot is set up")
        assert_speakable(text)
        assert (tmp_path / "weather-bot" / ".git").is_dir()

        bad = await tool.handler({"name": "../escape"})
        assert bad["is_error"] is True
        assert_speakable(bad["content"][0]["text"])
        assert not (tmp_path.parent / "escape").exists()

        again = await tool.handler({"name": "weather-bot"})  # folder now has files
        assert again["is_error"] is True and "already" in again["content"][0]["text"]
        assert len(host.calls) == 1
    finally:
        await timers.close()


async def test_sdk_tool_reports_partial_failure_as_error(tmp_path: Path) -> None:
    from spark.tools.server import build_spark_tools
    from spark.tools.timers import TimerService

    timers = TimerService(clock=FakeClock())
    svc = service(tmp_path, host=FakeHost(error=StepFailed("GitHub couldn't be reached")))
    try:
        tool = {t.name: t for t in build_spark_tools(timers, projects=svc)}["start_project"]
        out = await tool.handler({"name": "weather-bot"})
        assert out["is_error"] is True
        text = out["content"][0]["text"]
        assert "The GitHub repo failed because GitHub couldn't be reached." in text
        assert_speakable(text)
    finally:
        await timers.close()


def test_run_command_scrubs_secrets_from_child_env(monkeypatch):
    import subprocess

    from spark.tools import project

    seen = {}

    def fake_run(argv, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_should_not_leak_1234567890")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.setattr(project.subprocess, "run", fake_run)
    project.run_command(["git", "--version"], None, 5)
    assert "GITHUB_TOKEN" not in seen["env"] and "ANTHROPIC_API_KEY" not in seen["env"]
    assert "PATH" in {k.upper() for k in seen["env"]}
