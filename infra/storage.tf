# One bucket for the ML lifecycle, split by prefix:
#   data/raw/      raw snapshot (plain files, mounted read-only by the services)
#   data/live/     `brent data ingest` output (written by the scheduled job)
#   mlruns/        MLflow store: mlflow.db (SQLite backend) and artifacts/
#   results/       reports, plots and explanations shown by the dashboard
#   dvc/           DVC remote (content-addressed cache): `dvc remote add gcs gs://<bucket>/dvc`
# Google-managed encryption keys: a customer-managed KMS key adds cost and key management
# without a compliance requirement to meet. No legacy usage logs (CSV objects in another
# bucket): Cloud Audit Logs already record who changes the bucket and its IAM, and Google
# recommends them over usage logs.
#trivy:ignore:AVD-GCP-0066
#trivy:ignore:AVD-GCP-0077
#trivy:ignore:GCP-0077
resource "google_storage_bucket" "ml" {
  name                        = local.bucket
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = !var.deletion_protection

  # Versioning turns an overwrite or deletion (a bad training run, a corrupted mlflow.db)
  # into a recoverable event; the lifecycle below bounds what it costs.
  versioning {
    enabled = true
  }

  # Earlier versions are already kept by versioning: no extra soft-delete copy to pay for.
  soft_delete_policy {
    retention_duration_seconds = 0
  }

  lifecycle_rule {
    condition {
      num_newer_versions = var.noncurrent_versions_kept
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      days_since_noncurrent_time = var.noncurrent_version_days
      with_state                 = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 1
    }
    action {
      type = "AbortIncompleteMultipartUpload"
    }
  }
}
