"""``start_project``: the "Hey Spark, let's start a new project called weather-bot" flow.

Spec §1 example, §8.3, §10 Phase 3. The steps, in order:

1. a local folder ``<projects.root_dir>/<slug>`` (refused if it exists and isn't empty),
2. ``git init`` there, plus a README, a ``.gitignore`` and an initial commit (local only),
3. a GitHub repo through the ``gh`` CLI (:class:`GhCliRepoHost`), linked as ``origin`` and
   **never pushed**,
4. a tracking task (:class:`GitHubIssueTracker`: an issue in the new repo),
5. optionally a Claude Code window in the folder (:class:`ClaudeCodeLauncher`).

Why ``gh`` and not the GitHub MCP server: the MCP server is optional and off by default, and a
fine-grained PAT for it usually can't create repos. ``gh`` keeps its own stored login, and
Spark's PAT lives in ``SPARK_GITHUB_PAT`` (not ``GITHUB_TOKEN``), so it never changes the
identity ``gh`` uses (docs/connectors.md §5).

Everything external goes through small injectable interfaces (:class:`RepoHost`,
:class:`Tracker`, :class:`SessionLauncher`, and a :data:`Runner` for subprocesses), so tests use
fakes and never touch GitHub. :func:`plan_project` has no side effects; the same
:class:`ProjectPlan` feeds the spoken confirmation (:func:`summarize_start_project`) and the run
(:func:`start_project`). A failing step never hides the others: each one is reported, the local
folder is never rolled back, and a remote that may exist is always mentioned.

Confirmation (P3.1): the tool is exported as side-effecting by
:func:`spark.tools.server.build_spark_server`, so the PreToolUse confirmation hook must ask before
it runs. The hook's summarizer registry should pick up
:func:`spark.tools.server.spark_confirmation_summarizers`, which maps
``"mcp__spark__start_project"`` to ``lambda tool_input: summarize_start_project(tool_input,
settings.projects)``, e.g.::

    for name, fn in spark_confirmation_summarizers(settings.projects).items():
        summarizers[name] = fn   # fn(tool_input: Mapping[str, Any]) -> str

Every text returned here is speakable: no URLs, IDs or raw command output.
"""

from __future__ import annotations

import logging
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from spark.brain.confirm import speakable_value  # confirm never imports spark.tools: no cycle
from spark.util.logging import child_env

if TYPE_CHECKING:
    from spark.config import ProjectsConfig, Settings

log = logging.getLogger(__name__)

TOOL_NAME = "start_project"

MAX_SLUG_LEN = 64
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(10)} | {f"LPT{i}" for i in range(10)}
)
_REPO_URL = re.compile(
    r"https://github\.com/([A-Za-z0-9-]+/[A-Za-z0-9._-]+?)(?:\.git)?/?\s*$", re.M
)
_ISSUE_URL = re.compile(r"https://github\.com/[^\s]+/issues/(\d+)")

GITIGNORE = """\
# Environments and secrets
.env
.env.*
.venv/
venv/

# Python
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
dist/
build/
*.egg-info/

# Node
node_modules/

# Editors and OS
.vscode/
.idea/
.DS_Store
Thumbs.db
"""


class ProjectError(Exception):
    """A speakable problem with the request (bad name, folder already in use, ...)."""


# --------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class ProjectPlan:
    """What :func:`start_project` will do. Computing it touches nothing."""

    name: str  # as the user said it
    slug: str  # folder and repo name, e.g. "weather-bot"
    path: Path
    description: str
    create_repo: bool
    private: bool
    github_owner: str | None
    tracker: str  # "github_issue" | "none"
    open_claude_code: bool

    @property
    def repo_name(self) -> str:
        """``owner/slug`` when an owner is configured, else ``slug`` (the gh account)."""
        return f"{self.github_owner}/{self.slug}" if self.github_owner else self.slug

    @property
    def wants_task(self) -> bool:
        return self.tracker != "none"


def slugify(name: str) -> str:
    """Validate a spoken project name and turn it into a folder/repo name.

    "Weather Bot" -> "weather-bot". Raises :class:`ProjectError` (speakable) for names that are
    empty, try to leave the projects folder, or are reserved on Windows."""
    raw = (name or "").strip()
    if not raw:
        raise ProjectError("The project needs a name.")
    if any(sep in raw for sep in ("/", "\\", ":")) or ".." in raw:
        raise ProjectError("A project name can't contain slashes, colons or double dots.")
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    if not slug:
        raise ProjectError("The project name needs at least one letter or number.")
    if len(slug) > MAX_SLUG_LEN:
        raise ProjectError(f"That name is too long, keep it under {MAX_SLUG_LEN} characters.")
    if slug.upper() in _WINDOWS_RESERVED:
        raise ProjectError(f"{slug} is a reserved name on Windows, pick another one.")
    return slug


def default_root(settings: ProjectsConfig) -> Path:
    return Path(settings.root_dir).expanduser().resolve()


def plan_project(
    name: str,
    settings: ProjectsConfig,
    *,
    root: Path | None = None,
    private: bool | None = None,
    open_claude_code: bool | None = None,
    description: str | None = None,
) -> ProjectPlan:
    """Build the plan without side effects (only reads the filesystem).

    Raises :class:`ProjectError` for a bad name or a folder that already has files in it."""
    slug = slugify(name)
    base = (root if root is not None else default_root(settings)).resolve()
    path = (base / slug).resolve()
    if path.parent != base:
        raise ProjectError("That name would put the project outside your projects folder.")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ProjectError(f"There's already something called {slug} in your projects folder.")
    desc = " ".join((description or "").split())[:200] or f"{name.strip()}, started with Spark."
    return ProjectPlan(
        name=name.strip(),
        slug=slug,
        path=path,
        description=desc,
        create_repo=settings.repo_host != "none",
        private=settings.github_private if private is None else bool(private),
        github_owner=settings.github_owner,
        tracker=settings.tracker if settings.repo_host != "none" else "none",
        open_claude_code=(
            settings.open_claude_code if open_claude_code is None else bool(open_claude_code)
        ),
    )


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def describe_plan(plan: ProjectPlan) -> str:
    """ "I'll create a project called weather-bot: a local folder and git repo, a private
    GitHub repo and a tracking issue. Should I go ahead?" """
    parts = ["a local folder and git repo"]
    if plan.create_repo:
        parts.append(f"a {'private' if plan.private else 'public'} GitHub repo")
    if plan.wants_task:
        parts.append("a tracking issue")
    if plan.open_claude_code:
        parts.append("a Claude Code window in the folder")
    return f"I'll create a project called {plan.slug}: {_join(parts)}. Should I go ahead?"


def _arg_bool(args: Mapping[str, Any], key: str) -> bool | None:
    value = args.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}
    return bool(value)


def plan_from_args(
    args: Mapping[str, Any], settings: ProjectsConfig, *, root: Path | None = None
) -> ProjectPlan:
    """The plan for a ``start_project`` tool call's arguments."""
    return plan_project(
        str(args.get("name") or ""),
        settings,
        root=root,
        private=_arg_bool(args, "private"),
        open_claude_code=_arg_bool(args, "open_claude_code"),
        description=str(args.get("description") or "") or None,
    )


def summarize_start_project(
    args: Mapping[str, Any], settings: ProjectsConfig, *, root: Path | None = None
) -> str:
    """The one spoken line the confirmation hook says before ``start_project`` runs."""
    try:
        return describe_plan(plan_from_args(args, settings, root=root))
    except ProjectError:
        # The tool itself will refuse and explain; confirming a no-op is harmless. The raw name
        # failed validation, so it may hold a URL, path, ID or markup: say it only if it passes
        # the same filter as every other spoken argument.
        name = speakable_value(args.get("name"))
        if name:
            return f"I'll try to start a project called {name}. Should I go ahead?"
        return "I'll try to start a new project. Should I go ahead?"


# --------------------------------------------------------------------------- interfaces


@dataclass(frozen=True)
class RepoInfo:
    full_name: str  # owner/name; never spoken
    url: str = ""  # never spoken
    linked: bool = True  # the local folder has it as origin


@dataclass(frozen=True)
class TaskInfo:
    ref: str  # e.g. "#1"; never spoken
    url: str = ""


class StepFailed(Exception):
    """A step failed. ``str(err)`` is speakable; ``remote_created`` says a remote exists."""

    def __init__(self, message: str, *, remote_created: bool | None = False) -> None:
        super().__init__(message)
        self.remote_created = remote_created  # None = unknown (timeout)


class RepoHost(Protocol):
    def create_repo(
        self, name: str, *, private: bool, description: str, source: Path | None = None
    ) -> RepoInfo: ...


class Tracker(Protocol):
    def create_task(
        self, title: str, body: str, *, repo: RepoInfo | None = None, cwd: Path | None = None
    ) -> TaskInfo: ...


class SessionLauncher(Protocol):
    def open(self, path: Path) -> None: ...


Runner = Callable[[Sequence[str], "Path | None", float], "subprocess.CompletedProcess[str]"]


def run_command(
    argv: Sequence[str], cwd: Path | None, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run a command without a console window or stdin, capturing text output."""
    return subprocess.run(
        list(argv),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        check=False,
        env=child_env(),  # gh/git use their own logins, never Spark's keys from .env
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _gh_error(stderr: str, what: str) -> str:
    text = stderr.lower()
    if "already exists" in text:
        return f"a {what} with that name already exists on GitHub"
    if "auth login" in text or "not logged" in text or "authentication" in text:
        return "the GitHub command line tool isn't logged in"
    if "could not resolve" in text or "connect" in text or "network" in text:
        return "GitHub couldn't be reached"
    return "the GitHub command failed"


class GhCliRepoHost:
    """Creates the repo with ``gh repo create``: private or public, linked as ``origin``, and
    deliberately without ``--push``."""

    def __init__(
        self,
        owner: str | None = None,
        *,
        gh_path: str = "gh",
        timeout_s: float = 60.0,
        runner: Runner = run_command,
    ) -> None:
        self.owner = owner
        self.gh_path = gh_path
        self.timeout_s = timeout_s
        self.runner = runner

    def argv(
        self, gh: str, name: str, *, private: bool, description: str, source: Path | None
    ) -> list[str]:
        full = f"{self.owner}/{name}" if self.owner and "/" not in name else name
        argv = [gh, "repo", "create", full, "--private" if private else "--public"]
        argv += ["--description", description]
        if source is not None:
            argv += ["--source", str(source), "--remote", "origin"]
        return argv

    def create_repo(
        self, name: str, *, private: bool, description: str, source: Path | None = None
    ) -> RepoInfo:
        gh = shutil.which(self.gh_path)
        if gh is None:
            raise StepFailed("the GitHub command line tool isn't installed")
        argv = self.argv(gh, name, private=private, description=description, source=source)
        try:
            proc = self.runner(argv, source, self.timeout_s)
        except subprocess.TimeoutExpired as err:
            raise StepFailed(
                "it timed out, so it may or may not exist on GitHub", remote_created=None
            ) from err
        except OSError as err:
            raise StepFailed("the GitHub command couldn't start") from err
        match = _REPO_URL.search(proc.stdout or "")
        if proc.returncode != 0:
            log.warning("gh repo create failed (%s): %s", proc.returncode, proc.stderr[-500:])
            if match:  # created, but adding the remote failed
                raise StepFailed(
                    "the GitHub repo was created but couldn't be linked to the local folder",
                    remote_created=True,
                )
            raise StepFailed(_gh_error(proc.stderr or "", "repo"))
        full_name = match.group(1) if match else (f"{self.owner}/{name}" if self.owner else name)
        return RepoInfo(full_name=full_name, url=match.group(0).strip() if match else "")


class GitHubIssueTracker:
    """The tracking task is an issue in the new repo, created with ``gh issue create``."""

    def __init__(
        self, *, gh_path: str = "gh", timeout_s: float = 60.0, runner: Runner = run_command
    ) -> None:
        self.gh_path = gh_path
        self.timeout_s = timeout_s
        self.runner = runner

    def argv(self, gh: str, title: str, body: str, repo: RepoInfo | None) -> list[str]:
        argv = [gh, "issue", "create", "--title", title, "--body", body]
        if repo is not None and "/" in repo.full_name:
            argv += ["--repo", repo.full_name]
        return argv

    def create_task(
        self, title: str, body: str, *, repo: RepoInfo | None = None, cwd: Path | None = None
    ) -> TaskInfo:
        if repo is None:
            raise StepFailed("there's no GitHub repo to put the issue in")
        gh = shutil.which(self.gh_path)
        if gh is None:
            raise StepFailed("the GitHub command line tool isn't installed")
        try:
            proc = self.runner(self.argv(gh, title, body, repo), cwd, self.timeout_s)
        except subprocess.TimeoutExpired as err:
            raise StepFailed("it timed out") from err
        except OSError as err:
            raise StepFailed("the GitHub command couldn't start") from err
        if proc.returncode != 0:
            log.warning("gh issue create failed (%s): %s", proc.returncode, proc.stderr[-500:])
            raise StepFailed(_gh_error(proc.stderr or "", "issue"))
        match = _ISSUE_URL.search(proc.stdout or "")
        return TaskInfo(
            ref=f"#{match.group(1)}" if match else "", url=match.group(0) if match else ""
        )


class NoTracker:
    """``tracker: none``: no task. :func:`start_project` skips the step for it."""

    def create_task(
        self, title: str, body: str, *, repo: RepoInfo | None = None, cwd: Path | None = None
    ) -> TaskInfo:
        raise StepFailed("no tracker is configured")


Popen = Callable[..., Any]


class ClaudeCodeLauncher:
    """Opens an interactive ``claude`` in a new window in the project folder.

    Windows gets a new console; macOS gets a Terminal window (its shell has the user's own
    environment, so none of Spark's keys leak in either way).
    """

    def __init__(
        self,
        *,
        claude_path: str = "claude",
        platform: str = sys.platform,
        popen: Popen | None = None,
    ) -> None:
        self.claude_path = claude_path
        self.platform = platform
        self.popen = popen

    def open(self, path: Path) -> None:
        if not self.platform.startswith(("win", "darwin")):
            raise StepFailed("opening Claude Code in a new window only works on Windows and macOS")
        exe = shutil.which(self.claude_path)
        if exe is None:
            raise StepFailed("the Claude Code command isn't installed")
        popen = self.popen or subprocess.Popen
        if self.platform == "darwin":
            self._open_terminal(popen, exe, path)
            return
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
        try:
            popen([exe], cwd=str(path), creationflags=flags, env=child_env())
        except OSError as err:
            raise StepFailed("Claude Code couldn't start") from err

    @staticmethod
    def _open_terminal(popen: Popen, exe: str, path: Path) -> None:
        command = f"cd {shlex.quote(str(path))} && {shlex.quote(exe)}"
        script = command.replace("\\", "\\\\").replace('"', '\\"')
        argv = [
            "osascript",
            "-e",
            f'tell application "Terminal" to do script "{script}"',
            "-e",
            'tell application "Terminal" to activate',
        ]
        try:
            popen(argv, env=child_env())
        except OSError as err:
            raise StepFailed("Terminal couldn't open Claude Code") from err


# --------------------------------------------------------------------------- run


@dataclass
class StepResult:
    key: str
    ok: bool
    skipped: bool = False
    detail: str = ""


@dataclass
class ProjectResult:
    plan: ProjectPlan
    steps: list[StepResult] = field(default_factory=list)
    repo: RepoInfo | None = None
    task: TaskInfo | None = None
    remote_state: str = "none"  # none | created | unknown

    def step(self, key: str) -> StepResult | None:
        return next((s for s in self.steps if s.key == key), None)

    def succeeded(self, key: str) -> bool:
        step = self.step(key)
        return step is not None and step.ok

    @property
    def ok(self) -> bool:
        return all(s.ok or s.skipped for s in self.steps)

    def summary(self) -> str:
        return summarize_result(self)


_STEP_LABELS = {
    "folder": "the local folder",
    "git": "the git repo",
    "commit": "the first commit",
    "repo": "the GitHub repo",
    "task": "the tracking issue",
    "claude": "the Claude Code window",
}


def summarize_result(result: ProjectResult) -> str:
    """A short speakable report: what was done, what failed and why, what was not attempted."""
    plan = result.plan
    done = [s for s in result.steps if s.ok]
    failed = [s for s in result.steps if not s.ok and not s.skipped]
    not_done = [s for s in result.steps if s.skipped]
    if not failed and not not_done:
        parts = ["a local folder with a first commit"]
        if result.succeeded("repo"):
            parts.append(f"a {'private' if plan.private else 'public'} GitHub repo")
        if result.succeeded("task"):
            parts.append("a tracking issue")
        text = f"Done. {plan.slug} is set up: {_join(parts)}."
        if result.succeeded("claude"):
            text += " Claude Code is open in a new window."
        if result.repo is not None:
            text += " Nothing has been pushed yet."
        return text
    sentences: list[str] = []
    if done:
        sentences.append(f"For {plan.slug}, I set up {_join([_STEP_LABELS[s.key] for s in done])}.")
    for s in failed:
        sentences.append(f"{_cap(_STEP_LABELS[s.key])} failed because {s.detail}.")
    if not_done:
        sentences.append(f"I didn't get to {_join([_STEP_LABELS[s.key] for s in not_done])}.")
    if not result.succeeded("repo"):
        if result.remote_state == "created":
            sentences.append("Heads up, the repo does exist on GitHub.")
        elif result.remote_state == "unknown":
            sentences.append("Check GitHub before trying again, the repo might exist.")
    if result.succeeded("folder"):
        sentences.append("I left the local folder in place.")
    return " ".join(sentences)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _git(runner: Runner, git: str, args: Sequence[str], cwd: Path, timeout: float) -> None:
    try:
        proc = runner([git, *args], cwd, timeout)
    except subprocess.TimeoutExpired as err:
        raise StepFailed("git timed out") from err
    except OSError as err:
        raise StepFailed("git couldn't start") from err
    if proc.returncode != 0:
        log.warning("git %s failed (%s): %s", args[0], proc.returncode, proc.stderr[-500:])
        if "tell me who you are" in (proc.stderr or "").lower():
            raise StepFailed("git doesn't know your name and email yet")
        raise StepFailed(f"git {args[0]} failed")


def start_project(
    plan: ProjectPlan,
    host: RepoHost | None,
    tracker: Tracker | None,
    launcher: SessionLauncher | None,
    *,
    runner: Runner = run_command,
    git_path: str = "git",
    timeout_s: float = 60.0,
) -> ProjectResult:
    """Run the plan. Blocking (subprocesses); call it through ``asyncio.to_thread``.

    Never raises for a failed step: each step is recorded in the result. The local folder is
    kept when a later step fails. Only a bad or already-used folder raises
    :class:`ProjectError`, before anything is created."""
    result = ProjectResult(plan=plan)
    path = plan.path
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ProjectError(f"There's already something called {plan.slug} in your projects folder.")

    def not_attempted(*keys: str) -> ProjectResult:
        for key in keys:
            result.steps.append(StepResult(key, ok=False, skipped=True, detail="not attempted"))
        return result

    optional = [
        k
        for k, wanted in (
            ("repo", plan.create_repo),
            ("task", plan.wants_task),
            ("claude", plan.open_claude_code),
        )
        if wanted
    ]

    # 1. folder
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        log.warning("start_project: mkdir failed: %s", err)
        result.steps.append(StepResult("folder", ok=False, detail="I couldn't create the folder"))
        return not_attempted("git", "commit", *optional)
    result.steps.append(StepResult("folder", ok=True))

    # 2. git init, 3. first commit
    git = shutil.which(git_path)
    if git is None:
        result.steps.append(StepResult("git", ok=False, detail="git isn't installed"))
        return not_attempted("commit", *optional)
    try:
        _git(runner, git, ["init", "-b", "main"], path, timeout_s)
    except StepFailed as err:
        result.steps.append(StepResult("git", ok=False, detail=str(err)))
        return not_attempted("commit", *optional)
    result.steps.append(StepResult("git", ok=True))
    try:
        (path / "README.md").write_text(f"# {plan.slug}\n\n{plan.description}\n", encoding="utf-8")
        (path / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
        _git(runner, git, ["add", "README.md", ".gitignore"], path, timeout_s)
        _git(runner, git, ["commit", "-q", "-m", "Initial commit"], path, timeout_s)
    except (StepFailed, OSError) as err:
        detail = str(err) if isinstance(err, StepFailed) else "I couldn't write the starter files"
        result.steps.append(StepResult("commit", ok=False, detail=detail))
        # A repo without a commit is still usable; keep going with the remote steps.
    else:
        result.steps.append(StepResult("commit", ok=True))

    # 4. remote repo
    if plan.create_repo:
        if host is None:
            result.steps.append(StepResult("repo", ok=False, detail="no GitHub host is set up"))
        else:
            try:
                result.repo = host.create_repo(
                    plan.repo_name,
                    private=plan.private,
                    description=plan.description,
                    source=path,
                )
                result.remote_state = "created"
                result.steps.append(StepResult("repo", ok=True))
            except StepFailed as err:
                if err.remote_created is None:
                    result.remote_state = "unknown"
                elif err.remote_created:
                    result.remote_state = "created"
                result.steps.append(StepResult("repo", ok=False, detail=str(err)))
            except Exception:
                log.exception("start_project: repo host crashed")
                result.remote_state = "unknown"
                result.steps.append(StepResult("repo", ok=False, detail="something went wrong"))

    # 5. tracking task (needs the repo when it's a GitHub issue)
    if plan.wants_task:
        if result.repo is None or tracker is None:
            result.steps.append(StepResult("task", ok=False, skipped=True, detail="not attempted"))
        else:
            try:
                result.task = tracker.create_task(
                    f"Set up {plan.slug}",
                    f"Tracking issue for {plan.name}, created by Spark.\n\n{plan.description}",
                    repo=result.repo,
                    cwd=path,
                )
                result.steps.append(StepResult("task", ok=True))
            except StepFailed as err:
                result.steps.append(StepResult("task", ok=False, detail=str(err)))
            except Exception:
                log.exception("start_project: tracker crashed")
                result.steps.append(StepResult("task", ok=False, detail="something went wrong"))

    # 6. Claude Code
    if plan.open_claude_code:
        if launcher is None:
            result.steps.append(
                StepResult("claude", ok=False, detail="no Claude Code launcher is set up")
            )
        else:
            try:
                launcher.open(path)
                result.steps.append(StepResult("claude", ok=True))
            except StepFailed as err:
                result.steps.append(StepResult("claude", ok=False, detail=str(err)))
            except Exception:
                log.exception("start_project: launcher crashed")
                result.steps.append(StepResult("claude", ok=False, detail="something went wrong"))

    log.info(
        "start_project %s: %s",
        plan.slug,
        ", ".join(
            f"{s.key}={'ok' if s.ok else 'skip' if s.skipped else 'fail'}" for s in result.steps
        ),
    )
    return result


# --------------------------------------------------------------------------- service


class ProjectService:
    """Settings plus the external interfaces, as used by the ``start_project`` SDK tool."""

    def __init__(
        self,
        settings: ProjectsConfig,
        *,
        root: Path | None = None,
        host: RepoHost | None = None,
        tracker: Tracker | None = None,
        launcher: SessionLauncher | None = None,
        runner: Runner = run_command,
    ) -> None:
        self.settings = settings
        self.root = (root if root is not None else default_root(settings)).resolve()
        self.host = host
        self.tracker = tracker
        self.launcher = launcher
        self.runner = runner

    @classmethod
    def from_config(cls, settings: ProjectsConfig, *, root: Path | None = None) -> ProjectService:
        """The real implementations: ``gh`` for the repo and issue, ``claude`` for the window."""
        timeout = settings.command_timeout_s
        host: RepoHost | None = None
        if settings.repo_host == "gh":
            host = GhCliRepoHost(settings.github_owner, gh_path=settings.gh_path, timeout_s=timeout)
        tracker: Tracker = (
            GitHubIssueTracker(gh_path=settings.gh_path, timeout_s=timeout)
            if settings.tracker == "github_issue"
            else NoTracker()
        )
        launcher = ClaudeCodeLauncher(claude_path=settings.claude_path)
        return cls(settings, root=root, host=host, tracker=tracker, launcher=launcher)

    @classmethod
    def from_settings(cls, settings: Settings) -> ProjectService:
        """Resolve ``projects.root_dir`` like other config paths (``~`` and config-relative)."""
        return cls.from_config(settings.projects, root=settings.path(settings.projects.root_dir))

    def plan(self, args: Mapping[str, Any]) -> ProjectPlan:
        return plan_from_args(args, self.settings, root=self.root)

    def summarize(self, args: Mapping[str, Any]) -> str:
        return summarize_start_project(args, self.settings, root=self.root)

    def run(self, plan: ProjectPlan) -> ProjectResult:
        return start_project(
            plan,
            self.host,
            self.tracker,
            self.launcher,
            runner=self.runner,
            timeout_s=self.settings.command_timeout_s,
        )
