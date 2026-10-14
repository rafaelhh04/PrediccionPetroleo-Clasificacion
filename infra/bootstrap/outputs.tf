output "state_bucket" {
  description = "Bucket of the main stack's state: terraform -chdir=infra init -backend-config=bucket=<this>."
  value       = google_storage_bucket.tfstate.name
}

output "image_repository" {
  description = "Docker repository URL (<region>-docker.pkg.dev/<project>/<repository>)."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}

output "workload_identity_provider" {
  description = "GitHub variable GCP_WORKLOAD_IDENTITY_PROVIDER."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deployer_service_account" {
  description = "GitHub variable GCP_DEPLOYER_SA (and var.deployer_service_account of the main stack)."
  value       = google_service_account.deployer.email
}

output "planner_service_account" {
  description = "GitHub variable GCP_PLANNER_SA."
  value       = google_service_account.planner.email
}
