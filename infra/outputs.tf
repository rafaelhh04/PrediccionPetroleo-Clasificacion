output "api_urls" {
  description = "Public URL of the API per environment (OpenAPI docs at /docs)."
  value       = { for env, svc in google_cloud_run_v2_service.api : env => svc.uri }
}

output "dashboard_urls" {
  description = "Public URL of the dashboard per environment."
  value       = { for env, svc in google_cloud_run_v2_service.dashboard : env => svc.uri }
}

output "mlflow_url" {
  description = "MLflow UI (private: `gcloud run services proxy brent-mlflow`)."
  value       = google_cloud_run_v2_service.mlflow.uri
}

output "bucket" {
  description = "ML bucket (data/, mlruns/, results/, dvc/)."
  value       = google_storage_bucket.ml.name
}

output "dvc_remote" {
  description = "DVC remote URL: uv run dvc remote add -d gcs <this>."
  value       = "gs://${google_storage_bucket.ml.name}/dvc"
}

output "image_repository" {
  description = "Docker repository the services pull from."
  value       = local.registry
}

output "service_accounts" {
  description = "Service account of each workload."
  value = {
    api       = google_service_account.api.email
    dashboard = google_service_account.dashboard.email
    mlflow    = google_service_account.mlflow.email
  }
}
