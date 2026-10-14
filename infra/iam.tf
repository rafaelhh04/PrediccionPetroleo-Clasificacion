# One service account per workload, each with the smallest role on the resources it uses.
# No project-wide storage or secret roles: bindings are on the bucket and on each secret.
resource "google_service_account" "api" {
  account_id   = "brent-api"
  display_name = "Brent API (Cloud Run)"
  description  = "Reads the registry, data and nothing else."
}

resource "google_service_account" "dashboard" {
  account_id   = "brent-dashboard"
  display_name = "Brent dashboard (Cloud Run)"
  description  = "Reads results and data; calls the public API."
}

resource "google_service_account" "mlflow" {
  account_id   = "brent-mlflow"
  display_name = "MLflow tracking server (Cloud Run)"
  description  = "Reads and writes the MLflow store in the bucket."
}

# Read-only mounts (GCS FUSE needs objects.get and objects.list).
resource "google_storage_bucket_iam_member" "readers" {
  for_each = {
    api       = google_service_account.api.member
    dashboard = google_service_account.dashboard.member
  }

  bucket = google_storage_bucket.ml.name
  role   = "roles/storage.objectViewer"
  member = each.value
}

# Read-write mount (no bucket administration, no IAM changes).
resource "google_storage_bucket_iam_member" "mlflow" {
  bucket = google_storage_bucket.ml.name
  role   = "roles/storage.objectUser"
  member = google_service_account.mlflow.member
}

# Cloud SQL connector (mlflow_backend = "cloudsql"): connect only, no administration.
resource "google_project_iam_member" "cloudsql_client" {
  for_each = local.cloudsql ? {
    api    = google_service_account.api.member
    mlflow = google_service_account.mlflow.member
  } : {}

  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = each.value
}

# The CD workflow deploys revisions that run as these accounts: it must be allowed to "act as"
# each of them, and nothing else (no project-wide serviceAccountUser).
resource "google_service_account_iam_member" "deployer_act_as" {
  for_each = {
    api       = google_service_account.api.name
    dashboard = google_service_account.dashboard.name
    mlflow    = google_service_account.mlflow.name
  }

  service_account_id = each.value
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${local.deployer}"
}
