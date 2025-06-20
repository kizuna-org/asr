# GitHub Provider Setup for kizuna-org

Date: 2025-06-20

## Overview

This document describes the implementation of the GitHub provider in our Terraform configuration for the kizuna-org organization. The primary purpose of this configuration is to enable pulling from GitHub Container Registry (GHCR) using a Personal Access Token with minimal permissions.

## Implementation Details

### Added GitHub Provider

The GitHub provider was added to the Terraform configuration with the following settings:

- Provider: `integrations/github`
- Version: `~> 5.0`
- Organization: `kizuna-org`
- Authentication: Personal Access Token (PAT)

### Configuration Files Modified

1. `infrastructure/terraform/main.tf`:
   - Added GitHub provider to required_providers block
   - Added GitHub provider configuration with token and owner parameters

2. `infrastructure/terraform/variables.tf`:
   - Added `github_token` variable with sensitive flag

3. `infrastructure/terraform/terraform.tfvars.example`:
   - Added example for `github_token` variable

## How to Use

### Creating a GitHub Personal Access Token (PAT)

1. Go to GitHub Settings: https://github.com/settings/tokens
2. Click "Generate new token" (classic)
3. Give your token a descriptive name (e.g., "GHCR Pull for kizuna-org")
4. Select only the required scope:
   - `read:packages` (to pull from GitHub Container Registry)
5. Click "Generate token"
6. Copy the token value (it will only be shown once)

### Configuring Terraform

1. Copy `terraform.tfvars.example` to `terraform.tfvars` if not already done
2. Add your GitHub PAT to the `github_token` variable in `terraform.tfvars`
3. Run `terraform init` to initialize the GitHub provider
4. Run `terraform plan` and `terraform apply` as usual

## Security Considerations

- The GitHub PAT is marked as sensitive in the Terraform configuration
- Never commit the actual PAT to version control
- Consider using a CI/CD secret or environment variable for the PAT in production environments
- Following the principle of least privilege, the PAT only has `read:packages` scope for pulling from GHCR
- Rotate the PAT periodically according to your security policies