# Secrets live in Secret Manager and reach the containers as environment variables resolved
# by Cloud Run at start-up: never in the image, the repository or the workflow logs.

# Backend store URI of MLflow on Cloud SQL (it embeds the database password).
resource "google_secret_manager_secret" "mlflow_backend_uri" {
  count = local.cloudsql ? 1 : 0

  secret_id = "brent-mlflow-backend-uri"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "mlflow_backend_uri" {
  count = local.cloudsql ? 1 : 0

  secret      = google_secret_manager_secret.mlflow_backend_uri[0].id
  secret_data = "postgresql://${google_sql_user.mlflow[0].name}:${random_password.mlflow_db[0].result}@/${google_sql_database.mlflow[0].name}?host=/cloudsql/${google_sql_database_instance.mlflow[0].connection_name}"
}

resource "google_secret_manager_secret_iam_member" "mlflow_backend_uri" {
  for_each = local.cloudsql ? {
    api    = google_service_account.api.member
    mlflow = google_service_account.mlflow.member
  } : {}

  secret_id = google_secret_manager_secret.mlflow_backend_uri[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = each.value
}

# Webhook (Slack/Discord/Teams-compatible URL) for retraining and drift notifications.
# Terraform creates the empty secret; add the value out of band so it never enters the state:
#   printf '%s' "$WEBHOOK_URL" | gcloud secrets versions add brent-notify-webhook --data-file=-
resource "google_secret_manager_secret" "notify_webhook" {
  secret_id = "brent-notify-webhook"
  replication {
    auto {}
  }
}
