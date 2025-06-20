# GCP Pub/Sub and IAM configuration for CI/CD pipeline
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}


provider "google" {
  project = local.project_id
  region  = local.region
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