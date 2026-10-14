output "state_bucket" {
  description = "Bucket of the main stack's state: terraform -chdir=infra init -backend-config=bucket=<this>."
  value       = google_storage_bucket.tfstate.name
}

output "image_repository" {
  description = "Docker repository URL (<region>-docker.pkg.dev/<project>/<repository>)."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}
