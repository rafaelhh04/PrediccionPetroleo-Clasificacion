"""HTTP inference API (FastAPI) serving the registered model (``brent serve``)."""

from brent_forecast.api.app import create_app

__all__ = ["create_app"]
