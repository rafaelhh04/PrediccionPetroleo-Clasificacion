# Plan-level tests of the main stack with mocked providers: no credentials, no API calls.
#   terraform -chdir=infra init -backend=false && terraform -chdir=infra test
mock_provider "google" {
  mock_data "google_project" {
    defaults = {
      number = "123456789"
    }
  }
}

mock_provider "random" {}

variables {
  project_id = "demo-project"
  image_tag  = "abc123"
}

run "defaults_are_free_tier_friendly" {
  command = plan

  assert {
    condition     = toset(keys(google_cloud_run_v2_service.api)) == toset(["staging", "prod"])
    error_message = "One API service per environment is expected."
  }
  assert {
    condition     = google_cloud_run_v2_service.api["prod"].template[0].scaling[0].min_instance_count == 0
    error_message = "The API must scale to zero (no idle cost)."
  }
  assert {
    condition     = google_cloud_run_v2_service.api["prod"].template[0].containers[0].resources[0].cpu_idle
    error_message = "The API must use request-based billing."
  }
  assert {
    condition     = google_cloud_run_v2_service.mlflow.template[0].scaling[0].max_instance_count == 1
    error_message = "The MLflow server must have a single instance (one SQLite writer)."
  }
  assert {
    condition     = length(google_sql_database_instance.mlflow) == 0 && length(google_billing_budget.monthly) == 0
    error_message = "No paid database and no budget by default."
  }
  assert {
    condition     = google_storage_bucket.ml.name == "demo-project-brent-ml"
    error_message = "Default bucket name."
  }
  assert {
    condition     = google_cloud_run_v2_service.api["prod"].template[0].containers[0].image == "us-central1-docker.pkg.dev/demo-project/brent/api:abc123"
    error_message = "The initial image comes from the project's Artifact Registry."
  }
}

run "read_only_mounts_for_the_api_and_dashboard" {
  command = plan

  assert {
    condition = alltrue([
      for v in google_cloud_run_v2_service.api["prod"].template[0].volumes : v.gcs[0].read_only
    ])
    error_message = "The API mounts the bucket read-only."
  }
  assert {
    condition = alltrue([
      for v in google_cloud_run_v2_service.dashboard["prod"].template[0].volumes : v.gcs[0].read_only
    ])
    error_message = "The dashboard mounts the bucket read-only."
  }
  assert {
    condition     = google_storage_bucket_iam_member.readers["api"].role == "roles/storage.objectViewer"
    error_message = "The API service account can only read objects."
  }
  assert {
    condition     = length(google_cloud_run_v2_service_iam_member.mlflow_ui) == 0
    error_message = "The MLflow UI is never public."
  }
}

run "private_services" {
  command = plan

  variables {
    public = false
  }

  assert {
    condition     = length(google_cloud_run_v2_service_iam_member.api_public) == 0
    error_message = "public = false grants no allUsers invoker."
  }
}

run "cloudsql_backend" {
  command = plan

  variables {
    mlflow_backend  = "cloudsql"
    billing_account = "000000-000000-000000"
  }

  assert {
    condition     = google_sql_database_instance.mlflow[0].settings[0].tier == "db-f1-micro"
    error_message = "Cheapest Cloud SQL tier."
  }
  assert {
    condition     = length(google_secret_manager_secret_version.mlflow_backend_uri) == 1
    error_message = "The backend URI (with the password) goes to Secret Manager."
  }
  assert {
    condition = anytrue([
      for e in google_cloud_run_v2_service.api["prod"].template[0].containers[0].env :
      e.name == "BRENT_TRACKING__URI" && length(e.value_source) == 1
    ])
    error_message = "The API reads the tracking URI from the secret."
  }
  assert {
    condition     = length(google_project_iam_member.cloudsql_client) == 2
    error_message = "Only the API and MLflow may connect to Cloud SQL."
  }
  assert {
    condition     = length(google_billing_budget.monthly) == 1
    error_message = "A billing account enables the budget alert."
  }
}

run "rejects_an_unknown_backend" {
  command = plan

  variables {
    mlflow_backend = "mysql"
  }

  expect_failures = [var.mlflow_backend]
}
