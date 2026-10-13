"""Load test of the inference API with Locust (not part of the pytest suite).

Start the API against a populated registry, then run (``make load-test``)::

    uv run brent serve --port 8000 &
    uv run --group load locust -f tests/load/locustfile.py --headless \\
        --host http://127.0.0.1:8000 --users 10 --spawn-rate 5 --run-time 60s \\
        --csv results/load/locust

The mix is mostly predictions: from the latest day (cached feature row) and
from client features (validation against the contract), plus probes and an
occasional invalid request that must be rejected with 422.
"""

import random
from typing import Any

from locust import HttpUser, between, task


class ApiUser(HttpUser):
    """A client polling the service."""

    wait_time = between(0.5, 1.5)
    features: list[str]

    def on_start(self) -> None:
        """Read the input contract once, as a real client would."""
        info = self.client.get("/model-info").json()
        self.features = info["features"] or []

    def _body(self) -> dict[str, Any]:
        return {"features": {name: random.uniform(-1.0, 1.0) for name in self.features}}

    @task(4)
    def predict_latest(self) -> None:
        """Predict the next day from the latest available data."""
        self.client.post("/predict", name="POST /predict (latest)")

    @task(4)
    def predict_features(self) -> None:
        """Predict from features sent by the client."""
        self.client.post("/predict", json=self._body(), name="POST /predict (features)")

    @task(1)
    def health(self) -> None:
        """Readiness probe."""
        self.client.get("/health")

    @task(1)
    def model_info(self) -> None:
        """Model metadata."""
        self.client.get("/model-info")

    @task(1)
    def invalid(self) -> None:
        """A request with a missing feature must be a 422, never a 5xx."""
        body = self._body()
        body["features"].pop(next(iter(body["features"]), ""), None)
        with self.client.post(
            "/predict", json=body, name="POST /predict (invalid)", catch_response=True
        ) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"expected 422, got {response.status_code}")
