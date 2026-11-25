"""The batch pipeline's stages must exist, import cleanly, and agree across runners."""

import ast
import importlib

import pytest

import run_pipeline
from conftest import PROJECT_ROOT

STAGE_MODULES = [module for _, module in run_pipeline.STAGES]


def test_stage_order_is_the_batch_contract():
    assert STAGE_MODULES == [
        "src.ingestion.snowflake_ingestion",
        "src.ingestion.snowflake_to_s3",
        "src.processing.s3_to_pyspark",
        "src.scoring.score_model",
        "src.evaluation.evaluate_model",
    ]


@pytest.mark.parametrize("module", STAGE_MODULES)
def test_stage_module_imports(module):
    # Catches missing modules/broken imports that would only surface as a
    # failed stage hours into a production run.
    importlib.import_module(module)


def test_airflow_dag_runs_the_same_stages_in_the_same_order():
    tree = ast.parse(
        (PROJECT_ROOT / "airflow" / "dags" / "ml_pipeline_dag.py").read_text(encoding="utf-8")
    )
    commands = [
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "DockerOperator"
        for kw in node.keywords
        if kw.arg == "command"
    ]
    dag_modules = [command.split("-m", 1)[1].strip() for command in commands]
    assert dag_modules == STAGE_MODULES
