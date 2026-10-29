"""Command line: ``python -m spark`` (spec §11).

python -m spark [run] [--text [--speak]] [--no-tts] [--debug-audio] [--brain B] [--stt S] [--tts T]
python -m spark doctor
python -m spark models download
python -m spark bench stt|tts|brain|latency|false-wake [--trials N] [--minutes M]
python -m spark record-fixture NAME [--seconds S]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

    from spark.config import Settings

log = logging.getLogger("spark")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_UNAVAILABLE = 2


def _utf8_stdio() -> None:
    """Windows consoles default to cp1252, which can't print ✔/✖ or some transcripts."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spark", description="Spark: a wake-word voice assistant.")
    p.add_argument("--config", help="path to config.yaml (default: ./config.yaml or $SPARK_CONFIG)")
    p.add_argument("--text", action="store_true", help="type input and print output; no audio")
    p.add_argument("--no-tts", action="store_true", help="print replies instead of speaking")
    p.add_argument(
        "--speak", action="store_true", help="with --text: also speak replies through the TTS"
    )
    p.add_argument(
        "--debug-audio", action="store_true", help="save the last utterance to debug_audio/"
    )
    p.add_argument("--brain", help="brain backend override (e.g. mock, claude_agent)")
    p.add_argument("--stt", help="STT backend override (e.g. faster_whisper, deepgram)")
    p.add_argument("--tts", help="TTS backend override (e.g. piper, kokoro, elevenlabs, print)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")
    sub.add_parser("run", help="run the voice assistant (default)")
    sub.add_parser("doctor", help="check the environment and print fixes")
    sub.add_parser("desktop", help="open the desktop window (needs: uv sync --extra desktop)")

    models = sub.add_parser("models", help="manage local models")
    models_sub = models.add_subparsers(dest="models_cmd", required=True, metavar="ACTION")
    dl = models_sub.add_parser("download", help="download whisper + the configured voice")
    dl.add_argument("--kokoro", action="store_true", help="also download Kokoro (~340 MB)")

    bench = sub.add_parser("bench", help="measure latency")
    bench_sub = bench.add_subparsers(dest="bench_cmd", required=True, metavar="WHAT")
    for name, default in (("stt", 5), ("tts", 5), ("brain", 5), ("latency", 10)):
        b = bench_sub.add_parser(name, help=f"benchmark {name}")
        b.add_argument("--trials", type=int, default=default)
    fw = bench_sub.add_parser("false-wake", help="count wake-word triggers on background audio")
    fw.add_argument("--minutes", type=float, default=30.0)

    rec = sub.add_parser("record-fixture", help="record a test fixture from the mic")
    rec.add_argument("name", help="file name, e.g. cmd_01 or wake_hey_spark_real")
    rec.add_argument("--seconds", type=float, default=4.0)
    return p


def load(args: argparse.Namespace) -> Settings:
    from spark.config import load_settings

    overrides: dict[str, Any] = {}
    if args.brain:
        overrides["brain"] = {"backend": args.brain}
    if args.stt:
        overrides["stt"] = {"backend": args.stt}
    if args.tts:
        overrides["tts"] = {"backend": args.tts}
    if args.no_tts or (args.text and not args.speak):
        overrides["tts"] = {"backend": "print"}
    if args.debug_audio:
        overrides["debug_audio"] = {"enabled": True}
    if args.verbose:
        overrides["logging"] = {"level": "DEBUG"}
    settings = load_settings(args.config, **overrides)
    if args.brain:
        settings.pin("brain.backend")  # beats the Brain window's choice in prefs.json
    return settings


def _console() -> Console:
    from rich.console import Console

    return Console(highlight=False)


def _safe(text: str) -> str:
    """Error text for the console: secrets masked, rich markup escaped."""
    from rich.markup import escape

    from spark.util.logging import redact

    return escape(redact(text))


def _unavailable(console: Console, exc: Exception) -> int:
    from spark.interfaces import BackendUnavailable

    if isinstance(exc, BackendUnavailable):
        console.print(f"[red]✖ {_safe(exc.backend)}[/]: {_safe(exc.reason)}")
        console.print(f"  fix: {_safe(exc.fix)}")
    else:
        console.print(f"[red]✖[/] {_safe(str(exc))}")
    return EXIT_UNAVAILABLE


# --- run ---------------------------------------------------------------------------------------


async def run_text(settings: Settings, *, speak: bool, console: Console) -> int:
    from spark.app import SparkApp

    app = SparkApp.from_settings(settings, audio=False, speaker=speak)
    if speak:
        # Without --speak the print TTS already writes the timer announcement to stdout.
        app.on_announce = lambda text: console.print(f"[bold]spark>[/] {text}")
    interactive = sys.stdin.isatty()
    # One stdin reader for the REPL and the y/n confirmation prompt, so they never race for
    # a line (spark/brain/confirm.py LineReader).
    from spark.brain.confirm import LineReader

    reader = app.line_reader or LineReader()
    try:
        await app.warmup()
        if interactive:
            console.print("[dim]Spark text mode. Type a message; 'quit' or Ctrl+D to exit.[/]")
        while True:
            if interactive:
                console.print("[bold]you>[/] ", end="")
            line = await reader.readline()
            if not line:
                break
            text = line.strip()
            if text.lower() in {"quit", "exit"}:
                break
            if not text:
                continue
            try:
                reply = await app.text_turn(text)
            except Exception as exc:
                console.print(f"[red]error:[/] {_safe(str(exc))}")
                continue
            if speak:
                console.print(f"[bold]spark>[/] {reply}")
    finally:
        await app.close()
    return EXIT_OK


async def run_voice(settings: Settings, console: Console) -> int:
    from spark.app import SparkApp, State

    app = SparkApp.from_settings(settings)

    def show(_old: State, new: State) -> None:
        console.print(f"[dim]· {new.value}[/]")

    app.on_state = show
    try:
        console.print("[dim]Loading models and connecting...[/]")
        await app.warmup()
        console.print("[green]Ready.[/] Say the wake word. Ctrl+C to quit.")
        await app.run()
    finally:
        await app.close()
    return EXIT_OK


# --- other commands ----------------------------------------------------------------------------


def _print_stats(console: Console, title: str, result: dict[str, Any]) -> None:
    from rich.table import Table

    table = Table(title=title)
    for col in ("metric", "n", "median ms", "p90 ms", "min ms", "max ms"):
        table.add_column(col)
    scalars: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, dict) and "median" in value:
            table.add_row(
                key,
                f"{value['n']:.0f}",
                f"{value['median']:.0f}",
                f"{value['p90']:.0f}",
                f"{value['min']:.0f}",
                f"{value['max']:.0f}",
            )
        else:
            scalars[key] = value
    console.print(table)
    console.print(json.dumps(scalars, indent=2, ensure_ascii=False))


async def run_bench(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    from spark import bench

    what = args.bench_cmd
    if what == "stt":
        _print_stats(console, "STT", await bench.bench_stt(settings, args.trials))
    elif what == "tts":
        _print_stats(console, "TTS (one sentence)", await bench.bench_tts(settings, args.trials))
    elif what == "brain":
        _print_stats(console, "Brain", await bench.bench_brain(settings, args.trials))
    elif what == "latency":
        from spark.app import SparkApp

        app = SparkApp.from_settings(settings)
        try:
            await app.warmup()
            console.print(
                f"Say the wake word and a short question, {args.trials} times "
                "(e.g. 'Hey Spark, what's the capital of Japan?')."
            )

            def progress(n: int, timer: Any) -> None:
                total = timer.breakdown().get("total_ms")
                console.print(f"  trial {n}: wake→first audio {total:.0f} ms")

            result = await bench.bench_latency(app, args.trials, on_turn=progress)
        finally:
            await app.close()
        from rich.table import Table

        table = Table(title="wake → first audio")
        for col in ("stage", "n", "median ms", "p90 ms"):
            table.add_column(col)
        for stage, s in result["summary"].items():
            table.add_row(stage, f"{s['n']:.0f}", f"{s['median']:.0f}", f"{s['p90']:.0f}")
        console.print(table)
        if result["response_ms"]:
            r = result["response_ms"]
            console.print(
                f"end of speech → first audio: median {r['median']:.0f} ms, p90 {r['p90']:.0f} ms"
            )
        verdict = "[green]PASS[/]" if result["passed"] else "[red]FAIL[/]"
        console.print(f"median total {result['median_total_ms']} ms (target < 2000) {verdict}")
        return EXIT_OK if result["passed"] else EXIT_FAIL
    elif what == "false-wake":
        console.print(f"Listening for {args.minutes:g} min; play TV or background talk.")
        result = await bench.bench_false_wake(
            settings,
            args.minutes,
            on_wake=lambda at: console.print(f"  wake at {at:.1f} s"),
        )
        console.print(json.dumps(result, indent=2))
        return EXIT_OK if result["passed"] else EXIT_FAIL
    return EXIT_OK


async def run_record(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    from spark.bench import record_fixture

    console.print(f"Recording {args.seconds:g} s... speak now.")
    path = await record_fixture(settings, args.name, args.seconds)
    console.print(f"[green]saved[/] {path}")
    return EXIT_OK


def run_models(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    from spark.models import InsufficientDiskSpace, download_models

    try:
        kokoro = args.kokoro or settings.tts.backend == "kokoro"
        download_models(settings, kokoro=kokoro, console=console)
    except InsufficientDiskSpace as exc:
        console.print(f"[red]✖ not enough disk space:[/] {exc}")
        return EXIT_FAIL
    return EXIT_OK


def run_doctor(settings: Settings, console: Console) -> int:
    from spark.doctor import main as doctor_main

    return doctor_main(settings, console=console)


async def dispatch(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    cmd = args.cmd or "run"
    if cmd == "run":
        if args.text:
            return await run_text(settings, speak=args.speak and not args.no_tts, console=console)
        return await run_voice(settings, console)
    if cmd == "bench":
        return await run_bench(args, settings, console)
    if cmd == "record-fixture":
        return await run_record(args, settings, console)
    raise ValueError(f"unknown command {cmd}")


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    args = build_parser().parse_args(argv)
    if args.cmd == "desktop":
        from spark.desktop.window import main as desktop_main

        return desktop_main(["--config", args.config] if args.config else [])
    console = _console()
    from spark.interfaces import BackendUnavailable
    from spark.registry import UnknownBackend
    from spark.util.logging import setup_logging

    try:
        settings = load(args)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]config error:[/] {_safe(str(exc))}")
        return EXIT_FAIL
    setup_logging(settings, console=not args.text or args.verbose)
    try:
        if args.cmd == "doctor":
            return run_doctor(settings, console)
        if args.cmd == "models":
            return run_models(args, settings, console)
        return asyncio.run(dispatch(args, settings, console))
    except KeyboardInterrupt:
        console.print("\n[dim]bye[/]")
        return EXIT_OK
    except (BackendUnavailable, UnknownBackend) as exc:
        return _unavailable(console, exc)


if __name__ == "__main__":
    sys.exit(main())
