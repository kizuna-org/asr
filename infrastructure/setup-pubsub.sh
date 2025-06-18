#!/bin/bash
# Setup GCP Pub/Sub topics and subscriptions for CI/CD pipeline
# 
# ⚠️  DEPRECATED: This script is deprecated in favor of Terraform configuration.
# Please use the Terraform configuration in infrastructure/terraform/ instead.
# See infrastructure/terraform/README.md for migration instructions.
#
# This script will be removed in a future version.

echo "⚠️  WARNING: This script is DEPRECATED!"
echo "Please use Terraform configuration in infrastructure/terraform/ instead."
echo "See infrastructure/terraform/README.md for setup instructions."
echo ""
read -p "Do you want to continue with this deprecated script? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting. Please use Terraform configuration instead."
    exit 1
fi

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
BUILD_TOPIC="build-triggers"
APP_TOPIC="app-triggers"
BUILD_SUBSCRIPTION="build-triggers-sub"
APP_SUBSCRIPTION="app-triggers-sub"

echo "🚀 Setting up GCP Pub/Sub for project: $PROJECT_ID"

# Set the project
gcloud config set project $PROJECT_ID

# Create topics
echo "📢 Creating Pub/Sub topics..."
gcloud pubsub topics create $BUILD_TOPIC --quiet || echo "Topic $BUILD_TOPIC already exists"
gcloud pubsub topics create $APP_TOPIC --quiet || echo "Topic $APP_TOPIC already exists"

# Create subscriptions
echo "📥 Creating Pub/Sub subscriptions..."
gcloud pubsub subscriptions create $BUILD_SUBSCRIPTION \
    --topic=$BUILD_TOPIC \
    --ack-deadline=600 \
    --message-retention-duration=7d \
    --quiet || echo "Subscription $BUILD_SUBSCRIPTION already exists"

gcloud pubsub subscriptions create $APP_SUBSCRIPTION \
    --topic=$APP_TOPIC \
    --ack-deadline=600 \
    --message-retention-duration=7d \
    --quiet || echo "Subscription $APP_SUBSCRIPTION already exists"

# Create service account for GitHub Actions
SERVICE_ACCOUNT_NAME="github-actions-cicd"
SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

echo "🔐 Creating service account for GitHub Actions..."
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
    --display-name="GitHub Actions CI/CD Service Account" \
    --description="Service account for GitHub Actions to publish to Pub/Sub" \
    --quiet || echo "Service account already exists"

# Grant necessary permissions
echo "🔑 Granting permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/pubsub.publisher" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/pubsub.subscriber" \
    --quiet

# Create and download service account key
KEY_FILE="github-actions-key.json"
echo "🗝️  Creating service account key..."
gcloud iam service-accounts keys create $KEY_FILE \
    --iam-account=$SERVICE_ACCOUNT_EMAIL \
    --quiet

echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Add the following secrets to your GitHub repository:"
echo "   - GCP_PROJECT_ID: $PROJECT_ID"
echo "   - GCP_SA_KEY: $(cat $KEY_FILE | base64 -w 0)"
echo ""
echo "2. Configure your GPU server with the following environment variables:"
echo "   - GCP_PROJECT_ID=$PROJECT_ID"
echo "   - BUILD_SUBSCRIPTION=$BUILD_SUBSCRIPTION"
echo "   - APP_SUBSCRIPTION=$APP_SUBSCRIPTION"
echo ""
echo "3. Set up Cloudflare R2 credentials on your GPU server"
echo "4. Deploy the subscriber scripts to your GPU server"
echo ""
echo "⚠️  Remember to securely store and then delete the key file: $KEY_FILE"