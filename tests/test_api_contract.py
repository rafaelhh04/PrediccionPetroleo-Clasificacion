"""Contract tests of the API: the OpenAPI schema is a reviewed, versioned artefact.

``tests/snapshots/openapi.json`` is the published contract. Any change to a path,
a status code or a field fails here until the snapshot is regenerated on purpose:

    UPDATE_SNAPSHOTS=1 uv run pytest tests/test_api_contract.py

so the diff of the snapshot shows reviewers exactly how clients are affected.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from brent_forecast.api import create_app
from brent_forecast.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from brent_forecast.api.service import ServedModel
from brent_forecast.config import Settings
from brent_forecast.models.baselines import PersistenceClassifier
from conftest import EXPECTED_FEATURES

SNAPSHOT = Path(__file__).parent / "snapshots" / "openapi.json"


def _unavailable(settings: Settings, alias: str) -> ServedModel:
    raise RuntimeError("no registry in contract tests")


@pytest.fixture
def schema(settings: Settings) -> dict[str, Any]:
    spec = create_app(settings, loader=_unavailable).openapi()
    spec["info"].pop("version")  # the package version is not part of the contract
    return spec


def test_openapi_schema_matches_the_snapshot(schema: dict[str, Any]) -> None:
    rendered = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOT.write_text(rendered, encoding="utf-8")
    assert SNAPSHOT.read_text(encoding="utf-8") == rendered, (
        "The OpenAPI contract changed; review it and regenerate the snapshot with "
        "UPDATE_SNAPSHOTS=1 uv run pytest tests/test_api_contract.py"
    )


def test_every_endpoint_documents_its_status_codes(schema: dict[str, Any]) -> None:
    paths = schema["paths"]
    codes = {
        (path, method): set(operation["responses"])
        for path, item in paths.items()
        for method, operation in item.items()
    }

    assert codes == {
        ("/health", "get"): {"200", "503"},
        ("/health/live", "get"): {"200"},
        ("/model-info", "get"): {"200", "503"},
        ("/predict", "post"): {"200", "422", "503"},
    }


def test_the_request_contract(schema: dict[str, Any]) -> None:
    request = schema["components"]["schemas"]["PredictRequest"]

    assert request["additionalProperties"] is False
    assert "required" not in request  # an empty body predicts the latest day
    features = request["properties"]["features"]["anyOf"][0]
    assert features["type"] == "object"
    assert features["additionalProperties"]["type"] == "number"
    body = schema["paths"]["/predict"]["post"]["requestBody"]
    assert body.get("required", False) is False


def test_the_documented_examples_are_valid_requests(schema: dict[str, Any]) -> None:
    examples = schema["components"]["schemas"]["PredictRequest"]["examples"]

    requests = [PredictRequest.model_validate(example) for example in examples]

    assert requests[0].features is None
    assert requests[1].features is not None
    assert sorted(requests[1].features) == sorted(EXPECTED_FEATURES)


def test_live_responses_conform_to_the_documented_models(
    settings: Settings, raw_data_dir: Path
) -> None:
    import pandas as pd

    model = PersistenceClassifier().fit(pd.DataFrame({"lag_ret_1": [0.1, -0.1]}), [0, 1])
    served = ServedModel("champion", model, tuple(EXPECTED_FEATURES), {"version": "3"})
    with TestClient(create_app(settings, loader=lambda s, a: served)) as client:
        HealthResponse.model_validate(client.get("/health").json())
        ModelInfoResponse.model_validate(client.get("/model-info").json())
        PredictResponse.model_validate(client.post("/predict").json())
        invalid = client.post("/predict", json={"features": {"vix": "high"}}).json()

    assert set(invalid) == {"detail"}
    assert all(set(error) == {"loc", "msg", "type"} for error in invalid["detail"])
