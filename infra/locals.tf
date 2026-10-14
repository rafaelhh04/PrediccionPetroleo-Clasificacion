locals {
  bucket   = coalesce(var.bucket_name, "${var.project_id}-brent-ml")
  registry = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_id}"
  images = {
    api       = "${local.registry}/api:${var.image_tag}"
    dashboard = "${local.registry}/dashboard:${var.image_tag}"
  }
  cloudsql = var.mlflow_backend == "cloudsql"
  deployer = coalesce(var.deployer_service_account, "brent-deployer@${var.project_id}.iam.gserviceaccount.com")

  # Every container mounts the bucket folders at the paths used by docker-compose, so the
  # absolute artefact locations MLflow records (file:///app/mlruns/artifacts/...) resolve
  # in all of them. Bucket layout: data/ (raw/ + live/), mlruns/ (registry), results/, dvc/.
  mount_root   = "/app"
  sqlite_uri   = "sqlite:////app/mlruns/mlflow.db"
  artifact_dir = "/app/mlruns/artifacts"

  # Tracking store of the services: plain value (SQLite) or Secret Manager (Cloud SQL URI).
  tracking_env = merge(
    { BRENT_TRACKING__ARTIFACT_DIR = local.artifact_dir },
    local.cloudsql ? {} : { BRENT_TRACKING__URI = local.sqlite_uri },
  )
  tracking_secret_env = local.cloudsql ? { BRENT_TRACKING__URI = google_secret_manager_secret.mlflow_backend_uri[0].secret_id } : {}
}
