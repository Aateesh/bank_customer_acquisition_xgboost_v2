"""Configuration contract and secret-hygiene tests."""

import ast
import os
import re
import shutil
import subprocess
import sys

import pytest

from conftest import PROJECT_ROOT


def _required_config_vars():
    """Names passed as get_env("NAME", required=True) in src/config.py."""
    tree = ast.parse((PROJECT_ROOT / "src" / "config.py").read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "get_env"
            and any(
                kw.arg == "required" and getattr(kw.value, "value", False)
                for kw in node.keywords
            )
        ):
            names.append(node.args[0].value)
    return names


REQUIRED_VARS = _required_config_vars()


def test_required_vars_discovered():
    assert "SNOWFLAKE_PASSWORD" in REQUIRED_VARS
    assert "S3_BUCKET_NAME" in REQUIRED_VARS


def test_env_example_documents_every_required_variable():
    text = (PROJECT_ROOT / "config" / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", text, flags=re.MULTILINE))
    assert set(REQUIRED_VARS) <= documented


def test_env_example_contains_no_aws_keys():
    text = (PROJECT_ROOT / "config" / ".env.example").read_text(encoding="utf-8")
    assert not re.search(r"^AWS_(ACCESS_KEY_ID|SECRET_ACCESS_KEY)=", text, re.MULTILINE)
    assert not re.search(r"AKIA[0-9A-Z]{16}", text)


def test_airflow_dag_forwards_every_required_variable():
    tree = ast.parse(
        (PROJECT_ROOT / "airflow" / "dags" / "ml_pipeline_dag.py").read_text(encoding="utf-8")
    )
    env_vars = next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "ENV_VARS" for t in node.targets)
    )
    assert set(REQUIRED_VARS) <= set(env_vars)


def test_config_fails_fast_when_required_variable_missing():
    env = dict(os.environ, SNOWFLAKE_PASSWORD="")
    result = subprocess.run(
        [sys.executable, "-c", "import src.config"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "SNOWFLAKE_PASSWORD" in result.stderr


@pytest.mark.skipif(
    shutil.which("git") is None or not (PROJECT_ROOT / ".git").exists(),
    reason="needs a git checkout",
)
def test_no_secrets_tracked_in_git():
    tracked = subprocess.run(
        ["git", "ls-files", "config/.env", ".env"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert tracked == "", "config/.env must not be tracked"

    leaked = subprocess.run(
        ["git", "grep", "-I", "-l", "-E", r"AKIA[0-9A-Z]{16}", "--", ".", ":!*.csv"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    ).stdout.strip()
    assert leaked == "", f"AWS access key pattern found in tracked files: {leaked}"
