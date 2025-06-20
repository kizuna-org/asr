# Cloudflare Credentials as Variables

Date: 2025-06-20

## Overview

This document describes the change to move the Cloudflare credentials (API token and account ID) from local values to variables in the Terraform configuration. This improves security by treating the API token as a sensitive value and keeping credentials out of the main configuration files.

## Changes Made

1. Added new variables in `variables.tf`:
   - `cloudflare_api_token` (marked as sensitive)
   - `cloudflare_account_id`
2. Updated the Cloudflare provider configuration in `main.tf` to use the variables
3. Updated the R2 bucket resource to use the account ID variable
4. Removed the Cloudflare credentials from `locals.tf`
5. Updated the `terraform.tfvars.example` file to include the new variables
6. Updated documentation to reflect these changes

## Rationale

Sensitive values like API tokens should be handled as variables in Terraform for several reasons:

1. **Security**: Variables marked as sensitive won't be displayed in logs or outputs
2. **Flexibility**: Values can be provided through various methods (environment variables, tfvars files, etc.)
3. **Best Practice**: Separating configuration from credentials is a security best practice

## Usage

To use this configuration, create a `terraform.tfvars` file with your Cloudflare credentials:

```
cloudflare_api_token = "your-cloudflare-api-token"
cloudflare_account_id = "your-cloudflare-account-id"
```

The API token will be used to authenticate with the Cloudflare API, and the account ID will be used to identify your Cloudflare account when creating and managing R2 resources.