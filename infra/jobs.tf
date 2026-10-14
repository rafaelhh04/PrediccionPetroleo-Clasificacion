# Scheduled work: Cloud Run Jobs started by Cloud Scheduler (both inside the free tiers).
#
#   brent-ingest   Mon-Fri  `brent data ingest`       appends the new trading days to data/live/
#   brent-retrain  weekly   `brent retrain`           ingest -> train -> report + promotion -> explain
#
# The retraining job is the registry writer: it works on a local copy of mlflow.db and replaces the
# object once the run succeeded (`--stage-store`), so a failed run leaves the registry untouched and
# GCS FUSE does not rewrite the database on every SQLite write. It is idempotent (it skips when the
# data, the training configuration and the code are unchanged) and notifies the outcome.
# The API notices the moved `champion` alias within serving.refresh_interval_seconds.

resource "google_service_account" "trainer" {
  account_id   = "brent-trainer"
  display_name = "Brent scheduled jobs (Cloud Run Jobs)"
  description  = "Ingests data, trains, registers models and writes reports."
}

resource "google_service_account" "scheduler" {
  account_id   = "brent-scheduler"
  display_name = "Brent job scheduler"
  description  = "Cloud Scheduler identity: may only start the Brent jobs."
}

resource "google_storage_bucket_iam_member" "trainer" {
  bucket = google_storage_bucket.ml.name
  role   = "roles/storage.objectUser"
  member = google_service_account.trainer.member
}

resource "google_secret_manager_secret_iam_member" "trainer_webhook" {
  secret_id = google_secret_manager_secret.notify_webhook.id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.trainer.member
}

resource "google_secret_manager_secret_iam_member" "trainer_backend_uri" {
  count = local.cloudsql ? 1 : 0

  secret_id = google_secret_manager_secret.mlflow_backend_uri[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.trainer.member
}

resource "google_project_iam_member" "trainer_cloudsql" {
  count = local.cloudsql ? 1 : 0

  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = google_service_account.trainer.member
}

locals {
  jobs = {
    retrain = {
      args    = ["retrain", "--stage-store"]
      cpu     = var.retrain.cpu
      memory  = var.retrain.memory
      timeout = var.retrain.timeout
      mounts  = ["data", "mlruns", "results"]
    }
    ingest = {
      args    = ["data", "ingest"]
      cpu     = "1"
      memory  = "1Gi"
      timeout = "600s"
      mounts  = ["data"]
    }
  }
  job_secret_env = merge(
    local.tracking_secret_env,
    var.notify_webhook ? { BRENT_NOTIFY__WEBHOOK_URL = google_secret_manager_secret.notify_webhook.secret_id } : {},
  )
}

resource "google_cloud_run_v2_job" "this" {
  for_each = local.jobs

  name                = "brent-${each.key}"
  location            = var.region
  deletion_protection = var.deletion_protection
  labels              = { component = each.key }

  template {
    task_count  = 1
    parallelism = 1

    template {
      service_account       = google_service_account.trainer.email
      execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
      timeout               = each.value.timeout
      max_retries           = 1 # safe: a failed run publishes nothing and a re-run is idempotent

      containers {
        image = local.images.api
        args  = each.value.args

        resources {
          limits = {
            cpu    = each.value.cpu
            memory = each.value.memory
          }
        }

        dynamic "env" {
          for_each = local.tracking_env
          content {
            name  = env.key
            value = env.value
          }
        }

        dynamic "env" {
          for_each = local.job_secret_env
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

        dynamic "volume_mounts" {
          for_each = toset(each.value.mounts)
          content {
            name       = volume_mounts.value
            mount_path = "${local.mount_root}/${volume_mounts.value}"
          }
        }
        dynamic "volume_mounts" {
          for_each = local.cloudsql ? [1] : []
          content {
            name       = "cloudsql"
            mount_path = "/cloudsql"
          }
        }
      }

      dynamic "volumes" {
        for_each = toset(each.value.mounts)
        content {
          name = volumes.value
          gcs {
            bucket        = google_storage_bucket.ml.name
            read_only     = false
            mount_options = ["only-dir=${volumes.value}"]
          }
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
  }

  lifecycle {
    ignore_changes = [
      template[0].template[0].containers[0].image, # deployed by CD with the production release
      client,
      client_version,
    ]
  }

  depends_on = [
    google_storage_bucket_iam_member.trainer,
    google_secret_manager_secret_iam_member.trainer_webhook,
    google_secret_manager_secret_iam_member.trainer_backend_uri,
  ]
}

# The scheduler identity may start these two jobs and nothing else.
resource "google_cloud_run_v2_job_iam_member" "scheduler" {
  for_each = google_cloud_run_v2_job.this

  name     = each.value.name
  location = var.region
  role     = "roles/run.invoker"
  member   = google_service_account.scheduler.member
}

resource "google_cloud_scheduler_job" "this" {
  for_each = {
    retrain = var.retrain.schedule
    ingest  = var.ingest_schedule
  }

  name             = "brent-${each.key}"
  description      = "Starts the brent-${each.key} Cloud Run job."
  region           = var.region
  schedule         = each.value
  time_zone        = var.schedule_time_zone
  attempt_deadline = "180s" # the Run API answers as soon as the execution starts

  retry_config {
    retry_count = 1
  }

  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/${google_cloud_run_v2_job.this[each.key].id}:run"

    oauth_token {
      service_account_email = google_service_account.scheduler.email
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }

  depends_on = [google_cloud_run_v2_job_iam_member.scheduler]
}
