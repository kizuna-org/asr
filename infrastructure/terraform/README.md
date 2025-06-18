# Terraform Infrastructure for GCP Pub/Sub CI/CD

This directory contains Terraform configuration files to set up the GCP infrastructure required for the CI/CD pipeline.

## Prerequisites

1. **Terraform**: Install Terraform >= 1.0
2. **GCP CLI**: Install and authenticate with `gcloud auth application-default login`
3. **GCP Project**: Ensure you have a GCP project with billing enabled
4. **Required APIs**: Enable the following APIs in your GCP project:
   ```bash
   gcloud services enable pubsub.googleapis.com
   gcloud services enable iam.googleapis.com
   ```

## Setup

1. **Copy the variables file**:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. **Edit terraform.tfvars** with your actual values:
   ```hcl
   project_id = "your-actual-project-id"
   region     = "us-central1"
   environment = "prod"
   ```

3. **Initialize Terraform**:
   ```bash
   terraform init
   ```

4. **Plan the deployment**:
   ```bash
   terraform plan
   ```

5. **Apply the configuration**:
   ```bash
   terraform apply
   ```

## What gets created

- **Pub/Sub Topics**:
  - `build-triggers`: For triggering container builds
  - `app-triggers`: For triggering application runs

- **Pub/Sub Subscriptions**:
  - `build-triggers-sub`: Subscription for build triggers
  - `app-triggers-sub`: Subscription for app triggers

- **Service Account**:
  - `github-actions-cicd`: Service account for GitHub Actions
  - IAM roles: `pubsub.publisher` and `pubsub.subscriber`

- **Service Account Key**: For GitHub Actions authentication

## After deployment

1. **Get the service account key**:
   ```bash
   terraform output -raw service_account_key
   ```

2. **Add GitHub Secrets**:
   - `GCP_PROJECT_ID`: Your GCP project ID
   - `GCP_SA_KEY`: The base64-encoded service account key from step 1

3. **Update your GPU server environment variables**:
   ```bash
   terraform output github_secrets_instructions
   ```

## Migration from bash script

If you're migrating from the existing `setup-pubsub.sh` script:

1. **Import existing resources** (if they exist):
   ```bash
   # Import topics
   terraform import google_pubsub_topic.build_triggers projects/YOUR_PROJECT_ID/topics/build-triggers
   terraform import google_pubsub_topic.app_triggers projects/YOUR_PROJECT_ID/topics/app-triggers
   
   # Import subscriptions
   terraform import google_pubsub_subscription.build_subscription projects/YOUR_PROJECT_ID/subscriptions/build-triggers-sub
   terraform import google_pubsub_subscription.app_subscription projects/YOUR_PROJECT_ID/subscriptions/app-triggers-sub
   
   # Import service account
   terraform import google_service_account.github_actions projects/YOUR_PROJECT_ID/serviceAccounts/github-actions-cicd@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

2. **Run terraform plan** to verify no changes are needed for imported resources

3. **Apply any new configurations**:
   ```bash
   terraform apply
   ```

## Cleanup

To destroy all resources:
```bash
terraform destroy
```

## Files

- `main.tf`: Main Terraform configuration
- `variables.tf`: Variable definitions
- `outputs.tf`: Output definitions
- `terraform.tfvars.example`: Example variables file
- `README.md`: This documentation