# Local values for GCP Pub/Sub CI/CD infrastructure

locals {
  # Core configuration
  project_id = "chumchat"
  region     = "asia-northeast1"
  zone       = "asia-northeast1-a"

  # Environment configuration
  environment = "prod"  # Previously was a variable

  # Resource naming
  build_topic_name = "build-triggers"
  app_topic_name = "app-triggers"
  build_subscription_name = "build-triggers-sub"
  app_subscription_name = "app-triggers-sub"
  service_account_name = "github-actions-cicd"
  app_service_account_name = "app-service-account"
  
  # GitHub configuration
  github_owner = "kizuna-org"
  
  # GitHub initial token for bootstrapping PAT creation
  # This should be a personal access token with admin:org scope
  # that will only be used to create the github_actions_pat resource
  github_bootstrap_token = "ghp_example_replace_with_actual_token"
}
