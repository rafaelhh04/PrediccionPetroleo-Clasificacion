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
    return re.findall(r"^FROM .+ AS (\w+)$", DOCKERFILE, flags=re.MULTILINE)


def _runtime() -> str:
    return DOCKERFILE.split("AS runtime", 1)[1]


def _in_context(path: str) -> bool:
    """Whether ``path`` reaches the build context (.dockerignore rules, last match wins)."""
    included = True
    for pattern in DOCKERIGNORE:
        negate = pattern.startswith("!")
        rule = pattern.removeprefix("!").rstrip("/")
        if fnmatch.fnmatch(path, rule) or path.startswith(rule + "/"):
            included = negate
    return included


def test_the_image_is_multi_stage_with_a_runtime_stage() -> None:
    assert _stages() == ["uv", "builder", "runtime"]
    assert "--no-dev" in DOCKERFILE
    assert "--frozen" in DOCKERFILE


def test_the_runtime_stage_runs_unprivileged_with_a_healthcheck() -> None:
    runtime = _runtime()
    user = re.findall(r"^USER (\S+)$", runtime, flags=re.MULTILINE)

    assert user and not user[-1].startswith(("root", "0"))
    assert "HEALTHCHECK" in runtime
    assert "/health" in runtime.split("HEALTHCHECK", 1)[1]
    assert "uv" not in re.findall(r"^COPY --from=(\w+)", runtime, flags=re.MULTILINE)


def test_the_default_command_serves_the_api_on_all_interfaces() -> None:
    entrypoint = re.search(r"^ENTRYPOINT (.+)$", DOCKERFILE, flags=re.MULTILINE)
    command = re.search(r"^CMD (.+)$", DOCKERFILE, flags=re.MULTILINE)
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

    assert {"api", "mlflow", "trainer"} <= set(services)
    assert services["trainer"]["profiles"] == ["train"]  # never started by `up`
    for name in ("api", "mlflow", "trainer"):
        volumes = services[name]["volumes"]
        assert "mlruns:/app/mlruns" in volumes, name
        assert "results:/app/results" in volumes, name
        assert "./data:/app/data:ro" in volumes, name
    assert set(COMPOSE["volumes"]) == {"mlruns", "results"}
    assert services["api"]["ports"] == ["${API_PORT:-8000}:8000"]


def test_the_mlflow_server_uses_the_same_store_as_the_api() -> None:
    command = COMPOSE["services"]["mlflow"]["command"]
    default_uri = yaml.safe_load((REPO_ROOT / "configs/default.yaml").read_text())["tracking"][
        "uri"
    ]

    assert "--backend-store-uri=sqlite:////app/mlruns/mlflow.db" in command
    # The API resolves the relative default URI against WORKDIR /app: the same file.
    assert default_uri == "sqlite:///mlruns/mlflow.db"
    assert re.search(r"^WORKDIR /app$", _runtime(), flags=re.MULTILINE)


def test_the_trainer_trains_then_reports() -> None:
    trainer = COMPOSE["services"]["trainer"]

    steps = [shlex.split(step) for step in trainer["command"][0].split("&&")]

    assert steps == [["brent", "train"], ["brent", "report"]]
