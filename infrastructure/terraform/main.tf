# GCP Pub/Sub and IAM configuration for CI/CD pipeline
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    github = {
      source  = "integrations/github"
      version = "~> 5.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
  }
}


provider "google" {
  project = local.project_id
  region  = local.region
}

# Primary GitHub provider using bootstrap token
provider "github" {
  token = local.github_bootstrap_token
  owner = local.github_owner
  alias = "bootstrap"
}

# Create GitHub Personal Access Token for CI/CD
resource "github_actions_pat" "cicd_token" {
  provider         = github.bootstrap
  name             = "terraform-managed-cicd-token"
  repository       = "chumchat"
  selected_repositories = ["chumchat"]
  permissions {
    contents = "read"
    packages = "read"
  }
  expiration = "2030-01-01"
}

# Secondary GitHub provider using the created PAT
provider "github" {
  token = github_actions_pat.cicd_token.token
  owner = local.github_owner
  alias = "pat"
}

# Pub/Sub Topics
resource "google_pubsub_topic" "build_triggers" {
  name = local.build_topic_name

  labels = {
    environment = local.environment
    purpose     = "ci-cd-build"
  }
}

resource "google_pubsub_topic" "app_triggers" {
  name = local.app_topic_name

  labels = {
    environment = local.environment
    purpose     = "ci-cd-app"
  }
}

# Pub/Sub Subscriptions
resource "google_pubsub_subscription" "build_subscription" {
  name  = local.build_subscription_name
  topic = google_pubsub_topic.build_triggers.name

  ack_deadline_seconds       = 600
  message_retention_duration = "604800s" # 7 days

  labels = {
    environment = local.environment
    purpose     = "ci-cd-build"
  }
}

resource "google_pubsub_subscription" "app_subscription" {
  name  = local.app_subscription_name
  topic = google_pubsub_topic.app_triggers.name

  ack_deadline_seconds       = 600
  message_retention_duration = "604800s" # 7 days

  labels = {
    environment = local.environment
    purpose     = "ci-cd-app"
  }
}

# Service Account for GitHub Actions
resource "google_service_account" "github_actions" {
  account_id   = local.service_account_name
  display_name = "GitHub Actions CI/CD Service Account"
  description  = "Service account for GitHub Actions to publish to Pub/Sub"
}

# IAM Policy Bindings
resource "google_project_iam_member" "github_actions_pubsub_publisher" {
  project = local.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.github_actions.email}"
}

resource "google_project_iam_member" "github_actions_pubsub_subscriber" {
  project = local.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.github_actions.email}"
}

# Service Account Key (for GitHub Actions)
resource "google_service_account_key" "github_actions_key" {
  service_account_id = google_service_account.github_actions.name
  public_key_type    = "TYPE_X509_PEM_FILE"
}

# Application Service Account
resource "google_service_account" "app_service_account" {
  account_id   = local.app_service_account_name
  display_name = "Application Service Account"
  description  = "Service account for application to access GCP resources"
}

# Application Service Account Key
resource "google_service_account_key" "app_service_account_key" {
  service_account_id = google_service_account.app_service_account.name
  public_key_type    = "TYPE_X509_PEM_FILE"
}

# Save the key to a local file (ignored by Git)
resource "local_file" "app_service_account_key_file" {
  content  = base64decode(google_service_account_key.app_service_account_key.private_key)
  filename = "${path.module}/keys/app-service-account-key.json"
}

# IAM Policy Bindings for Application Service Account
resource "google_project_iam_member" "app_pubsub_subscriber" {
  project = local.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.app_service_account.email}"
}

resource "google_project_iam_member" "app_storage_object_viewer" {
  project = local.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.app_service_account.email}"
}
