# MLflow tracking server: the UI over the registry the retraining job writes and the API reads.
#
# It runs the API image with the `mlflow` entrypoint, on a read-write mount of mlruns/.
# Never public (the MLflow UI has no authentication and can edit the registry): open it with
#   gcloud run services proxy brent-mlflow --region <region>   # http://localhost:8080
# after granting yourself roles/run.invoker through var.mlflow_ui_members.
#
# Backend store (var.mlflow_backend), the main cost trade-off of the stack:
# - sqlite: mlruns/mlflow.db on GCS FUSE. Free, but GCS objects have no locks and every
#   write rewrites the whole file, so there must be ONE writer at a time: max 1 instance here,
#   and the retraining job never overlaps with UI edits (it writes to the file directly).
#   Fine for a portfolio with one scheduled writer; not for a team.
# - cloudsql: PostgreSQL (db-f1-micro, ~9-10 USD/month incl. storage and backups). Concurrent
#   writers, point-in-time backups, standard MLflow production setup.
resource "google_cloud_run_v2_service" "mlflow" {
  name                = "brent-mlflow"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = var.deletion_protection
  labels              = { component = "mlflow" }

  template {
    service_account                  = google_service_account.mlflow.email
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    max_instance_request_concurrency = 20

    scaling {
      min_instance_count = 0
      max_instance_count = 1 # single writer (see above)
    }

    containers {
      image   = local.images.api
      command = ["mlflow"]
      args = [
        "server",
        "--default-artifact-root=${local.artifact_dir}",
        "--host=0.0.0.0",
        "--port=5000",
        "--workers=1",
        # Host-header check against DNS rebinding: the run.app URL and the local proxy.
        "--allowed-hosts=brent-mlflow-*.run.app,localhost:*,127.0.0.1:*",
      ]

      ports {
        container_port = 5000
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      dynamic "env" {
        for_each = local.cloudsql ? {} : { MLFLOW_BACKEND_STORE_URI = local.sqlite_uri }
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = local.cloudsql ? { MLFLOW_BACKEND_STORE_URI = google_secret_manager_secret.mlflow_backend_uri[0].secret_id } : {}
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        period_seconds    = 5
        timeout_seconds   = 3
        failure_threshold = 24
      }

      volume_mounts {
        name       = "mlruns"
        mount_path = "${local.mount_root}/mlruns"
      }
      dynamic "volume_mounts" {
        for_each = local.cloudsql ? [1] : []
        content {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }
      }
    }

    volumes {
      name = "mlruns"
      gcs {
        bucket        = google_storage_bucket.ml.name
        read_only     = false
        mount_options = ["only-dir=mlruns"]
      }
    }
    dynamic "volumes" {
      for_each = local.cloudsql ? [1] : []
      content {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [google_sql_database_instance.mlflow[0].connection_name]
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
      client,
      client_version,
      traffic, # CD tags the candidate revision and moves traffic after the smoke test
    ]
  }

  depends_on = [
    google_storage_bucket_iam_member.mlflow,
    google_secret_manager_secret_iam_member.mlflow_backend_uri,
    google_secret_manager_secret_version.mlflow_backend_uri,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "mlflow_ui" {
  for_each = toset(var.mlflow_ui_members)

  name     = google_cloud_run_v2_service.mlflow.name
  location = var.region
  role     = "roles/run.invoker"
  member   = each.value
}

# ── optional Cloud SQL backend ─────────────────────────────────
resource "google_sql_database_instance" "mlflow" {
  count = local.cloudsql ? 1 : 0

  name                = "brent-mlflow"
  database_version    = "POSTGRES_17"
  region              = var.region
  deletion_protection = var.deletion_protection

  settings {
    tier              = var.cloudsql_tier
    edition           = "ENTERPRISE" # shared-core tiers are not available in Enterprise Plus
    availability_type = "ZONAL"
    disk_type         = "PD_HDD" # half the price of SSD; MLflow metadata is tiny
    disk_size         = 10
    disk_autoresize   = false

    backup_configuration {
      enabled    = true
      start_time = "03:00"
      backup_retention_settings {
        retained_backups = 7
      }
    }

    # Public IP without authorized networks: the only way in is the Cloud SQL connector
    # (IAM-authorised, mutual TLS), as Cloud Run uses it. A private IP would need a VPC with
    # private services access and VPC egress for Cloud Run: more moving parts for no gain here.
    #trivy:ignore:AVD-GCP-0017
    ip_configuration {
      ipv4_enabled = true
      ssl_mode     = "TRUSTED_CLIENT_CERTIFICATE_REQUIRED" # the connector presents client certs
    }

    database_flags {
      name  = "log_connections"
      value = "on"
    }
    database_flags {
      name  = "log_disconnections"
      value = "on"
    }
    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }
    database_flags {
      name  = "log_lock_waits"
      value = "on"
    }
    database_flags {
      name  = "log_temp_files"
      value = "0"
    }
  }
}

resource "google_sql_database" "mlflow" {
  count = local.cloudsql ? 1 : 0

  name     = "mlflow"
  instance = google_sql_database_instance.mlflow[0].name
}

resource "random_password" "mlflow_db" {
  count = local.cloudsql ? 1 : 0

  length  = 32
  special = false # it goes into a URI
}

# The password is in the Terraform state (private, versioned bucket) and in Secret Manager.
resource "google_sql_user" "mlflow" {
  count = local.cloudsql ? 1 : 0

  name     = "mlflow"
  instance = google_sql_database_instance.mlflow[0].name
  password = random_password.mlflow_db[0].result
}
