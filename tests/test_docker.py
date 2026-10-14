"""Static checks of the Dockerfile, .dockerignore and docker-compose.yml.

They run without Docker (CI builds, smoke-tests and scans the image in the
`docker` job), so a change that would leak data into the image, run it as root
or break the compose stack fails in the fast test suite.
"""

import fnmatch
import re
import shlex

import yaml

from conftest import REPO_ROOT

DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text()
DOCKERIGNORE = [
    line.strip()
    for line in (REPO_ROOT / ".dockerignore").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]
COMPOSE = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())


def _stages() -> list[str]:
    return re.findall(r"^FROM .+ AS ([\w-]+)$", DOCKERFILE, flags=re.MULTILINE)


def _stage(name: str) -> str:
    """Body of stage ``name`` (up to the next FROM)."""
    return DOCKERFILE.split(f" AS {name}\n", 1)[1].split("\nFROM ", 1)[0]


def _runtime() -> str:
    return _stage("runtime")


def _in_context(path: str) -> bool:
    """Whether ``path`` reaches the build context (.dockerignore rules, last match wins)."""
    included = True
    for pattern in DOCKERIGNORE:
        negate = pattern.startswith("!")
        rule = pattern.removeprefix("!").rstrip("/")
        if fnmatch.fnmatch(path, rule) or path.startswith(rule + "/"):
            included = negate
    return included


def test_the_images_are_multi_stage_and_the_api_is_the_default_target() -> None:
    assert _stages() == ["uv", "builder", "builder-dashboard", "base", "dashboard", "runtime"]
    assert _stages()[-1] == "runtime"  # `docker build .` produces the API image
    assert "--no-dev" in DOCKERFILE
    assert "--frozen" in DOCKERFILE
    assert "--extra dashboard" in _stage("builder-dashboard")
    builder = _stage("builder")
    assert "--extra dashboard" not in builder  # the API image has no Streamlit
    assert "--extra postgres" in builder  # but it can use a PostgreSQL MLflow store (Cloud SQL)
    assert "--extra postgres" not in _stage("builder-dashboard")  # uv sync is exact


def test_the_final_images_run_unprivileged_with_a_healthcheck() -> None:
    user = re.findall(r"^USER (\S+)$", _stage("base"), flags=re.MULTILINE)
    assert user and not user[-1].startswith(("root", "0"))
    for stage, probe in (("runtime", "/health"), ("dashboard", "/_stcore/health")):
        body = _stage(stage)
        assert re.search(rf"^FROM base AS {stage}$", DOCKERFILE, flags=re.MULTILINE), stage
        assert probe in body.split("HEALTHCHECK", 1)[1], stage
        copied = re.findall(r"^COPY --from=([\w-]+)", body, flags=re.MULTILINE)
        assert copied and "uv" not in copied, stage


def test_the_default_command_serves_the_api_on_all_interfaces() -> None:
    entrypoint = re.search(r"^ENTRYPOINT (.+)$", _stage("base"), flags=re.MULTILINE)
    command = re.search(r"^CMD (.+)$", _runtime(), flags=re.MULTILINE)
    assert entrypoint and command

    args = [*yaml.safe_load(entrypoint.group(1)), *yaml.safe_load(command.group(1))]

    assert args[:2] == ["brent", "serve"]
    assert "0.0.0.0" in args


def test_the_build_context_holds_only_code_and_configuration() -> None:
    for path in ("pyproject.toml", "uv.lock", "src/brent_forecast/cli.py", "configs/default.yaml"):
        assert _in_context(path), path
    for path in (
        "data/raw/oil.csv",
        "results/metrics.json",
        "mlruns/mlflow.db",
        ".env",
        ".git/config",
        ".dvc/cache/x",
        "tests/conftest.py",
        "src/brent_forecast/__pycache__/cli.cpython-312.pyc",
    ):
        assert not _in_context(path), path


def test_compose_runs_the_api_and_mlflow_on_shared_volumes() -> None:
    services = COMPOSE["services"]

    assert {"api", "dashboard", "mlflow", "trainer"} <= set(services)
    assert services["trainer"]["profiles"] == ["train"]  # never started by `up`
    for name in ("api", "mlflow", "trainer"):
        volumes = services[name]["volumes"]
        assert "mlruns:/app/mlruns" in volumes, name
        assert "results:/app/results" in volumes, name
        assert "./data:/app/data:ro" in volumes, name
    assert set(COMPOSE["volumes"]) == {"mlruns", "results"}
    assert services["api"]["ports"] == ["${API_PORT:-8000}:8000"]


def test_the_dashboard_reads_results_and_calls_the_api_service() -> None:
    dashboard = COMPOSE["services"]["dashboard"]

    assert dashboard["build"]["target"] == "dashboard"
    assert "results:/app/results:ro" in dashboard["volumes"]
    assert dashboard["environment"]["BRENT_DASHBOARD__API_URL"] == "http://api:8000"
    assert dashboard["depends_on"] == ["api"]


def test_the_mlflow_server_uses_the_same_store_as_the_api() -> None:
    command = COMPOSE["services"]["mlflow"]["command"]
    default_uri = yaml.safe_load((REPO_ROOT / "configs/default.yaml").read_text())["tracking"][
        "uri"
    ]

    assert "--backend-store-uri=sqlite:////app/mlruns/mlflow.db" in command
    # The API resolves the relative default URI against WORKDIR /app: the same file.
    assert default_uri == "sqlite:///mlruns/mlflow.db"
    assert re.search(r"^WORKDIR /app$", _stage("base"), flags=re.MULTILINE)


def test_the_trainer_trains_reports_and_explains() -> None:
    trainer = COMPOSE["services"]["trainer"]

    steps = [shlex.split(step) for step in trainer["command"][0].split("&&")]

    assert steps == [["brent", "train"], ["brent", "report"], ["brent", "explain"]]
