# Inference API and Streamlit dashboard, one pair per environment (staging, prod).
#
# Request-based billing (cpu_idle) and min_instances = 0: an idle service costs nothing and a
# cold start takes a few seconds (imports + model load; startup CPU boost shortens it).
# Terraform creates the services with var.image_tag; afterwards the CD workflow deploys new
# images by digest, so the image is ignored here (Terraform must not roll them back).

resource "google_cloud_run_v2_service" "api" {
  for_each = var.environments

  name                = "brent-api-${each.key}"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = var.deletion_protection
  labels              = { component = "api", environment = each.key }

  template {
    service_account                  = google_service_account.api.email
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    timeout                          = var.api.timeout
    max_instance_request_concurrency = var.api.concurrency

    scaling {
      min_instance_count = var.api.min_instances
      max_instance_count = var.api.max_instances
    }

    containers {
      image = local.images.api
      args  = ["serve", "--host", "0.0.0.0", "--port", "8000"]

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = var.api.cpu
          memory = var.api.memory
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      dynamic "env" {
        for_each = local.tracking_env
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = local.tracking_secret_env
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

      # Liveness only: /health (readiness) is 503 until a champion is registered, and the
      # service must stay up meanwhile. The CD smoke test checks /health == 200.
      startup_probe {
        http_get {
          path = "/health/live"
        }
        period_seconds    = 5
        timeout_seconds   = 3
        failure_threshold = 24
      }

      liveness_probe {
        http_get {
          path = "/health/live"
        }
        period_seconds  = 30
        timeout_seconds = 5
      }

      volume_mounts {
        name       = "mlruns"
        mount_path = "${local.mount_root}/mlruns"
      }
      volume_mounts {
        name       = "data"
        mount_path = "${local.mount_root}/data"
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
        read_only     = true
        mount_options = ["only-dir=mlruns"]
      }
    }
    volumes {
      name = "data"
      gcs {
        bucket        = google_storage_bucket.ml.name
        read_only     = true
        mount_options = ["only-dir=data"]
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
    google_storage_bucket_iam_member.readers,
    google_secret_manager_secret_iam_member.mlflow_backend_uri,
    google_secret_manager_secret_version.mlflow_backend_uri,
  ]
}

resource "google_cloud_run_v2_service" "dashboard" {
  for_each = var.environments

  name                = "brent-dashboard-${each.key}"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = var.deletion_protection
  labels              = { component = "dashboard", environment = each.key }

  template {
    service_account                  = google_service_account.dashboard.email
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    max_instance_request_concurrency = var.dashboard.concurrency
    # Streamlit keeps a websocket per browser session: route a session to the same instance.
    session_affinity = true
    timeout          = "3600s"

    scaling {
      min_instance_count = 0
      max_instance_count = var.dashboard.max_instances
    }

    containers {
      image = local.images.dashboard
      args  = ["dashboard", "--host", "0.0.0.0", "--port", "8501"]

      ports {
        container_port = 8501
      }

      resources {
        limits = {
          cpu    = var.dashboard.cpu
          memory = var.dashboard.memory
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      env {
        name  = "BRENT_DASHBOARD__API_URL"
        value = google_cloud_run_v2_service.api[each.key].uri
      }
      env {
        name  = "BRENT_DASHBOARD__TIMEOUT_SECONDS"
        value = "20" # the first call may hit a cold API instance
      }

      startup_probe {
        http_get {
          path = "/_stcore/health"
        }
        period_seconds    = 5
        timeout_seconds   = 3
        failure_threshold = 24
      }

      volume_mounts {
        name       = "results"
        mount_path = "${local.mount_root}/results"
      }
      volume_mounts {
        name       = "data"
        mount_path = "${local.mount_root}/data"
      }
    }

    volumes {
      name = "results"
      gcs {
        bucket        = google_storage_bucket.ml.name
        read_only     = true
        mount_options = ["only-dir=results"]
      }
    }
    volumes {
      name = "data"
      gcs {
        bucket        = google_storage_bucket.ml.name
        read_only     = true
        mount_options = ["only-dir=data"]
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

  depends_on = [google_storage_bucket_iam_member.readers]
}

# Public demo: anyone can call the API and open the dashboard (no user data is involved).
resource "google_cloud_run_v2_service_iam_member" "api_public" {
  for_each = var.public ? var.environments : toset([])

  name     = google_cloud_run_v2_service.api[each.key].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "dashboard_public" {
  for_each = var.public ? var.environments : toset([])

  name     = google_cloud_run_v2_service.dashboard[each.key].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
