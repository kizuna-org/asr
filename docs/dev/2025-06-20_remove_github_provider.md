# Remove GitHub Provider from Terraform Configuration

## Issue
The GitHub provider was causing issues and was not needed for the current infrastructure setup.

## Changes Made
1. Removed the GitHub provider from the `required_providers` block in `main.tf`
2. Removed all GitHub provider configurations from `main.tf`
3. Removed GitHub-related variables (`github_owner` and `github_bootstrap_token`) from `locals.tf`
4. Kept the GitHub Actions service account name for GCP resources that are still needed

## Implementation Date
2025-06-20