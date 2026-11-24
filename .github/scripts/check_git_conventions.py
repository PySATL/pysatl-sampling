"""Validate branch names and commit headers against CONTRIBUTING.md."""

from __future__ import annotations

import os
import re
import subprocess
import sys

TYPES = r"(?:feat|fix|refactor|test|struct|ci|docs|chore)"
WORD_BLOCK = r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"
BRANCH_PATTERN = re.compile(rf"(?:{TYPES}|repo)/{WORD_BLOCK}(?:/{WORD_BLOCK})*")
COMMIT_PATTERN = re.compile(rf"{TYPES}(?:\((?P<scope>[^()\r\n]+)\))?: [a-z][^\r\n]*")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout


def _current_branch() -> str | None:
    if branch := os.environ.get("PYSATL_BRANCH"):
        return branch
    result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode == 1:
        return None
    result.check_returncode()
    return result.stdout.strip()


def _branch_error(branch: str) -> str | None:
    if branch == "main" or branch.startswith("dependabot/") or BRANCH_PATTERN.fullmatch(branch):
        return None
    return (
        f"Invalid branch name {branch!r}. Expected <type>/<name> with types from "
        "CONTRIBUTING.md, slash-separated blocks, and words separated by hyphens. "
        "Uppercase letters are allowed."
    )


def _commit_error(message: str) -> str | None:
    header = message.split("\n", 1)[0]
    match = COMMIT_PATTERN.fullmatch(header)
    if match is None or (match["scope"] is not None and not match["scope"].strip()):
        return (
            f"Invalid commit header {header!r}. Expected <type>: <description> or "
            "<type>(<scope>): <description>, with a nonempty scope, a type from "
            "CONTRIBUTING.md, and a description starting with a lowercase English letter."
        )
    if header.rstrip().endswith("."):
        return f"Invalid commit header {header!r}. The description must not end with a period."
    return None


def _history_errors(base: str, head: str) -> list[str]:
    if _git("rev-parse", "--is-shallow-repository").strip() == "true":
        raise ValueError("Full Git history is required. Set fetch-depth: 0 in actions/checkout.")
    base_commit = _git("rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}").strip()
    head_commit = _git("rev-parse", "--verify", "--end-of-options", f"{head}^{{commit}}").strip()
    fields = _git("log", "-z", "--format=%h%x00%B", f"{base_commit}..{head_commit}", "--").split(
        "\0"
    )[:-1]
    errors = []
    for revision, message in zip(fields[::2], fields[1::2], strict=True):
        if error := _commit_error(message):
            errors.append(f"{revision}: {error}")
    return errors


def main() -> int:
    """Validate the branch and CI commit range.

    Returns
    -------
    int
        Zero when all applicable conventions are satisfied, one otherwise.
    """
    errors = []
    try:
        branch = _current_branch()
        if branch is not None and (error := _branch_error(branch)):
            errors.append(error)
        if head := os.environ.get("PYSATL_COMMIT_HEAD"):
            errors.extend(
                _history_errors(os.environ.get("PYSATL_COMMIT_BASE", "origin/main"), head)
            )
    except (OSError, ValueError, subprocess.CalledProcessError) as exception:
        print(f"Git convention check failed: {exception}", file=sys.stderr)
        if isinstance(exception, subprocess.CalledProcessError) and exception.stderr:
            print(exception.stderr.strip(), file=sys.stderr)
        return 1
    for error in errors:
        print(error, file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
