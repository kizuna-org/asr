# Cloudflare API Token Variable

Date: 2025-06-20

## Overview

This document describes the change to move the Cloudflare API token from a local value to a variable in the Terraform configuration. This improves security by treating the API token as a sensitive value and keeping it out of the main configuration files.

## Changes Made

1. Added a new variable `cloudflare_api_token` in `variables.tf`
2. Updated the Cloudflare provider configuration in `main.tf` to use the variable
3. Removed the `cloudflare_api_token` from `locals.tf`
4. Updated the `terraform.tfvars.example` file to include the new variable
5. Updated documentation to reflect these changes

## Rationale

Sensitive values like API tokens should be handled as variables in Terraform for several reasons:

1. **Security**: Variables marked as sensitive won't be displayed in logs or outputs
2. **Flexibility**: Values can be provided through various methods (environment variables, tfvars files, etc.)
3. **Best Practice**: Separating configuration from credentials is a security best practice

## Usage

To use this configuration, create a `terraform.tfvars` file with your Cloudflare API token:

```
cloudflare_api_token = "your-cloudflare-api-token"
```

This token will be used to authenticate with the Cloudflare API when creating and managing R2 resources.