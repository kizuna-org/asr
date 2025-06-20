# Outputs for GCP Pub/Sub CI/CD infrastructure

output "project_id" {
  description = "GCP Project ID"
  value       = local.project_id
}

output "build_topic_name" {
  description = "Name of the build triggers topic"
  value       = google_pubsub_topic.build_triggers.name
}

output "app_topic_name" {
  description = "Name of the app triggers topic"
  value       = google_pubsub_topic.app_triggers.name
}

output "build_subscription_name" {
  description = "Name of the build triggers subscription"
  value       = google_pubsub_subscription.build_subscription.name
}

output "app_subscription_name" {
  description = "Name of the app triggers subscription"
  value       = google_pubsub_subscription.app_subscription.name
}

output "service_account_email" {
  description = "Email of the GitHub Actions service account"
  value       = google_service_account.github_actions.email
}

output "service_account_key" {
  description = "Base64 encoded service account key for GitHub Actions"
  value       = google_service_account_key.github_actions_key.private_key
  sensitive   = true
}

output "github_secrets_instructions" {
  description = "Instructions for setting up GitHub secrets"
  value       = <<-EOT
    Add the following secrets to your GitHub repository:
    
    GCP_PROJECT_ID: ${local.project_id}
    GCP_SA_KEY: ${google_service_account_key.github_actions_key.private_key}
    
    Environment variables for GPU server:
    GCP_PROJECT_ID=${local.project_id}
    BUILD_SUBSCRIPTION=${google_pubsub_subscription.build_subscription.name}
    APP_SUBSCRIPTION=${google_pubsub_subscription.app_subscription.name}
  EOT
  sensitive   = true
}

output "r2_bucket_name" {
  description = "Name of the Cloudflare R2 bucket"
  value       = cloudflare_r2_bucket.app_bucket.name
}

output "r2_endpoint_url" {
  description = "Endpoint URL for Cloudflare R2"
  value       = local.r2_endpoint_url
}
