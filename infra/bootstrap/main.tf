resource "google_project_service" "this" {
  for_each = var.services

  service = each.value
  # Disabling an API on destroy can break resources that other stacks still own.
  disable_on_destroy = false
}

# ── Terraform state of the main stack ──────────────────────────
# The GCS backend locks the state with a lock object (no extra table needed);
# object versioning keeps earlier states to recover from a bad apply.
# Google-managed encryption keys: a customer-managed KMS key adds cost and key management
# without a compliance requirement to meet.
#trivy:ignore:AVD-GCP-0066
resource "google_storage_bucket" "tfstate" {
  name                        = "${var.project_id}-tfstate"
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 20
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.this]
}

# ── Docker images ──────────────────────────────────────────────
# Tags are immutable: CI pushes one tag per commit SHA and deploys by digest.
resource "google_artifact_registry_repository" "docker" {
  repository_id = var.repository_id
  location      = var.region
  format        = "DOCKER"
  description   = "Images of the Brent direction classifier (API and dashboard)."

  docker_config {
    immutable_tags = true
  }

  # Keep the free tier (0.5 GB): the last N versions of each image, nothing untagged.
  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = var.keep_image_versions
    }
  }
  cleanup_policies {
    id     = "delete-old"
    action = "DELETE"
    condition {
      tag_state  = "ANY"
      older_than = "2592000s" # 30 days, unless kept above
    }
  }

  depends_on = [google_project_service.this]
}
