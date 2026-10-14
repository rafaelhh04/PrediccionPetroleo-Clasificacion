# Keyless authentication from GitHub Actions (OIDC -> Workload Identity Federation).
#
# GitHub issues a short-lived OIDC token per job; Google exchanges it for a short-lived access
# token of a service account. No JSON key exists anywhere. Who can become what:
#
#   deployer  <- jobs of this repository bound to a GitHub *environment* (staging, production).
#                The environment rules (branch = main, required reviewers for production) are
#                therefore part of the authorization, not just of the workflow file.
#   planner   <- any job of this repository (read-only: terraform plan on pull requests).
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"
  description               = "OIDC tokens of ${var.github_repository}."

  depends_on = [google_project_service.this]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-actions"
  display_name                       = "GitHub Actions OIDC"

  attribute_mapping = {
    "google.subject"        = "assertion.sub"
    "attribute.repository"  = "assertion.repository"
    "attribute.ref"         = "assertion.ref"
    "attribute.environment" = "has(assertion.environment) ? assertion.environment : 'none'"
  }

  # Tokens of any other repository (or of a renamed owner) are rejected before any binding.
  attribute_condition = "assertion.repository == '${var.github_repository}' && assertion.repository_owner_id == '${var.github_repository_owner_id}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

locals {
  pool = google_iam_workload_identity_pool.github.name
}

# ── deployer: push images, roll out Cloud Run revisions ─────────
resource "google_service_account" "deployer" {
  account_id   = "brent-deployer"
  display_name = "GitHub Actions deployer"
  description  = "Pushes images and deploys Cloud Run revisions (CD workflow)."
}

resource "google_service_account_iam_member" "deployer_wif" {
  for_each = var.deploy_environments

  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${local.pool}/attribute.environment/${each.value}"
}

resource "google_artifact_registry_repository_iam_member" "deployer" {
  repository = google_artifact_registry_repository.docker.name
  location   = google_artifact_registry_repository.docker.location
  role       = "roles/artifactregistry.writer"
  member     = google_service_account.deployer.member
}

# Update services and jobs (not their IAM policies). Acting as the runtime service accounts is
# granted per account in the main stack (iam.serviceAccountUser).
resource "google_project_iam_member" "deployer_run" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = google_service_account.deployer.member
}

# ── planner: read-only terraform plan in CI ─────────────────────
resource "google_service_account" "planner" {
  account_id   = "brent-planner"
  display_name = "GitHub Actions terraform plan"
  description  = "Reads the infrastructure and the state for terraform plan (CI)."
}

resource "google_service_account_iam_member" "planner_wif" {
  service_account_id = google_service_account.planner.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${local.pool}/attribute.repository/${var.github_repository}"
}

resource "google_project_iam_member" "planner" {
  for_each = toset(["roles/viewer", "roles/iam.securityReviewer"])

  project = var.project_id
  role    = each.value
  member  = google_service_account.planner.member
}

# The plan runs with -lock=false: reading the state is enough.
resource "google_storage_bucket_iam_member" "planner_state" {
  bucket = google_storage_bucket.tfstate.name
  role   = "roles/storage.objectViewer"
  member = google_service_account.planner.member
}
