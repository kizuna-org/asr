# GitHub PAT Resource Fix

## Issue
The Terraform configuration was using a non-existent resource type `github_actions_pat` which is not supported by the GitHub provider (`integrations/github`).

## Changes Made
1. Removed the `github_actions_pat` resource from `infrastructure/terraform/main.tf`
2. Simplified the GitHub provider configuration to use a single provider instance without aliases
3. Updated comments to explain that GitHub PATs need to be created manually through the GitHub UI

## Future Considerations
If automated PAT management is needed, consider:
1. Using the GitHub API directly through a local-exec provisioner
2. Using GitHub Actions secrets (`github_actions_secret` resource) to store tokens
3. Checking for updates to the GitHub provider that might add PAT management capabilities

## Implementation Date
2025-06-20