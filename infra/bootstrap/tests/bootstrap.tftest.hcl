# Plan-level tests of the bootstrap stack with mocked providers (no credentials).
mock_provider "google" {
  mock_resource "google_iam_workload_identity_pool" {
    defaults = {
      name = "projects/123456789/locations/global/workloadIdentityPools/github"
    }
  }
  mock_resource "google_service_account" {
    defaults = {
      name   = "projects/demo-project/serviceAccounts/mocked-sa@demo-project.iam.gserviceaccount.com"
      email  = "mocked-sa@demo-project.iam.gserviceaccount.com"
      member = "serviceAccount:mocked-sa@demo-project.iam.gserviceaccount.com"
    }
  }
}

variables {
  project_id = "demo-project"
}

run "only_this_repository_can_authenticate" {
  command = plan

  assert {
    condition     = strcontains(google_iam_workload_identity_pool_provider.github.attribute_condition, "assertion.repository == 'rafaelhh04/PrediccionPetroleo-Clasificacion'")
    error_message = "The provider must reject tokens of other repositories."
  }
  assert {
    condition     = strcontains(google_iam_workload_identity_pool_provider.github.attribute_condition, "assertion.repository_owner_id == '132058207'")
    error_message = "The provider must pin the owner id (rename-safe)."
  }
}

run "deploys_only_from_github_environments" {
  command = apply # mocked: computes the pool name used in the principal sets

  assert {
    condition = toset([for m in google_service_account_iam_member.deployer_wif : m.member]) == toset([
      "principalSet://iam.googleapis.com/projects/123456789/locations/global/workloadIdentityPools/github/attribute.environment/staging",
      "principalSet://iam.googleapis.com/projects/123456789/locations/global/workloadIdentityPools/github/attribute.environment/production",
    ])
    error_message = "Only jobs bound to the staging and production environments may deploy."
  }
  assert {
    condition     = google_artifact_registry_repository.docker.docker_config[0].immutable_tags
    error_message = "Image tags must be immutable."
  }
  assert {
    condition     = google_storage_bucket.tfstate.versioning[0].enabled && google_storage_bucket.tfstate.public_access_prevention == "enforced"
    error_message = "The state bucket is versioned and never public."
  }
}
