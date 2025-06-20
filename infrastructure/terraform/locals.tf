# Local values for GCP Pub/Sub CI/CD infrastructure

locals {
  # Core configuration
  project_id = "chumchat"
  region     = "asia-northeast1"
  zone       = "asia-northeast1-a"

  # Environment configuration
  environment = var.environment  # Keeping as variable for flexibility

  # Resource naming
  build_topic_name = "build-triggers"
  app_topic_name = "app-triggers"
  build_subscription_name = "build-triggers-sub"
  app_subscription_name = "app-triggers-sub"
  service_account_name = "github-actions-cicd"
  app_service_account_name = "app-service-account"
}
