#!/usr/bin/env python3
"""Run popcorn-cli and report the current public rank for a leaderboard user."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProblemConfig:
    folder: str
    leaderboard: str
    leaderboard_id: int


PROBLEMS: dict[str, ProblemConfig] = {
    "mxfp4-mm": ProblemConfig(
        folder="mxfp4-mm",
        leaderboard="amd-mxfp4-mm",
        leaderboard_id=763,
    ),
    "moe-mxfp4": ProblemConfig(
        folder="moe-mxfp4",
        leaderboard="amd-moe-mxfp4",
        leaderboard_id=764,
    ),
    "mixed-mla": ProblemConfig(
        folder="mixed-mla",
        leaderboard="amd-mixed-mla",
        leaderboard_id=765,
    ),
}

RATE_LIMIT_RETRY_PATTERN = re.compile(r"Try again in (\d+)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a problem with popcorn-cli and print the current leaderboard rank.",
    )
    parser.add_argument(
        "problem",
        choices=sorted(PROBLEMS),
        help="Problem folder to submit or inspect.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["test", "benchmark", "leaderboard", "profile"],
        default=[],
        help="Submission modes to run in order. Omit to only inspect rank and recent submissions.",
    )
    parser.add_argument(
        "--username",
        default="oldzhu",
        help="Public leaderboard username to look up.",
    )
    parser.add_argument(
        "--file",
        default="submission_clean.py",
        help="Submission file name relative to the problem folder.",
    )
    parser.add_argument(
        "--gpu",
        default="MI355X",
        help="GPU type passed to popcorn-cli.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=15.0,
        help="Seconds to wait before refreshing the public rank after a leaderboard submission.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=5,
        help="How many recent submissions to print from popcorn-cli submissions list.",
    )
    parser.add_argument(
        "--retry-rate-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically wait and retry when popcorn-cli hits the leaderboard submission rate limit.",
    )
    parser.add_argument(
        "--rate-limit-buffer-seconds",
        type=float,
        default=15.0,
        help="Extra seconds to wait past the reported cooldown before retrying a rate-limited submission.",
    )
    parser.add_argument(
        "--max-rate-limit-retries",
        type=int,
        default=1,
        help="Maximum number of automatic retries after rate-limit waits.",
    )
    return parser.parse_args()


def capture_command(command: list[str], cwd: Path) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    return completed.returncode, completed.stdout, completed.stderr


def print_command_output(stdout: str, stderr: str) -> None:
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)


def extract_rate_limit_seconds(stdout: str, stderr: str) -> int | None:
    match = RATE_LIMIT_RETRY_PATTERN.search(f"{stdout}\n{stderr}")
    if match is None:
        return None
    return int(match.group(1))


def run_submit_command(
    command: list[str],
    cwd: Path,
    retry_rate_limit: bool,
    rate_limit_buffer_seconds: float,
    max_rate_limit_retries: int,
) -> int:
    attempts = 0
    while True:
        print(f"$ {' '.join(command)}", flush=True)
        exit_code, stdout, stderr = capture_command(command, cwd)
        print_command_output(stdout, stderr)

        retry_after = extract_rate_limit_seconds(stdout, stderr)
        if exit_code == 0 or not retry_rate_limit or retry_after is None:
            return exit_code
        if attempts >= max_rate_limit_retries:
            return exit_code

        wait_seconds = retry_after + rate_limit_buffer_seconds
        resume_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + wait_seconds))
        print(
            f"Rate limit hit. Waiting {wait_seconds:.0f}s before retrying at {resume_at}...",
            flush=True,
        )
        time.sleep(wait_seconds)
        attempts += 1


def fetch_leaderboard(problem: ProblemConfig) -> dict:
    url = f"https://www.gpumode.com/api/leaderboard/{problem.leaderboard_id}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    return payload["data"]


def find_user_rank(problem: ProblemConfig, username: str) -> dict | None:
    data = fetch_leaderboard(problem)
    rankings = data.get("rankings", {})
    for group_name, rows in rankings.items():
        for row in rows:
            if (row.get("user_name") or "").lower() == username.lower():
                result = dict(row)
                result["gpu_group"] = group_name
                return result
    return None


def format_score_us(score_seconds: float | int | None) -> str:
    if score_seconds in (None, ""):
        return "-"
    return f"{float(score_seconds) * 1e6:.3f} us"


def print_rank(problem: ProblemConfig, username: str) -> int:
    try:
        row = find_user_rank(problem, username)
        data = fetch_leaderboard(problem)
    except urllib.error.URLError as error:
        print(f"Failed to fetch leaderboard API: {error}", file=sys.stderr)
        return 2

    rankings = data.get("rankings", {}).get("MI355X", [])
    top10 = rankings[:10]

    print("\nPublic leaderboard snapshot")
    print("-" * 60)
    if row is None:
        print(f"User {username!r} not found on {problem.leaderboard}.")
    else:
        print(f"User:        {row['user_name']}")
        print(f"Leaderboard: {problem.leaderboard}")
        print(f"Rank:        #{row['rank']}")
        print(f"Score:       {format_score_us(row.get('score'))}")
        print(f"Submission:  {row.get('submission_id', '-')}")
        print(f"File:        {row.get('file_name', '-')}")

    if top10:
        cutoff = top10[-1]
        print(f"Top 10 cut:  #{cutoff['rank']} at {format_score_us(cutoff.get('score'))}")
        print(f"Top 1:       {top10[0]['user_name']} at {format_score_us(top10[0].get('score'))}")
    return 0


def print_recent_submissions(problem: ProblemConfig, cwd: Path, limit: int) -> int:
    code, stdout, stderr = capture_command(
        [
            "popcorn-cli",
            "submissions",
            "list",
            "--leaderboard",
            problem.leaderboard,
            "--limit",
            str(limit),
        ],
        cwd,
    )
    print("\nRecent CLI submissions")
    print("-" * 60)
    if stdout:
        print(stdout.rstrip())
    if stderr:
        print(stderr.rstrip(), file=sys.stderr)
    return code


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    problem = PROBLEMS[args.problem]
    problem_dir = root / problem.folder

    if not problem_dir.exists():
        print(f"Problem folder not found: {problem_dir}", file=sys.stderr)
        return 2

    submission_file = problem_dir / args.file
    if not submission_file.exists():
        print(f"Submission file not found: {submission_file}", file=sys.stderr)
        return 2

    for mode in args.modes:
        exit_code = run_submit_command(
            [
                "popcorn-cli",
                "submit",
                "--gpu",
                args.gpu,
                "--leaderboard",
                problem.leaderboard,
                "--mode",
                mode,
                args.file,
                "--no-tui",
            ],
            problem_dir,
            retry_rate_limit=args.retry_rate_limit and mode == "leaderboard",
            rate_limit_buffer_seconds=args.rate_limit_buffer_seconds,
            max_rate_limit_retries=args.max_rate_limit_retries,
        )
        if exit_code != 0:
            print(f"Submission failed for mode {mode}.", file=sys.stderr)
            return exit_code
        if mode == "leaderboard":
            time.sleep(args.poll_seconds)

    submissions_code = print_recent_submissions(problem, root, args.list_limit)
    rank_code = print_rank(problem, args.username)
    return submissions_code or rank_code


if __name__ == "__main__":
    raise SystemExit(main())