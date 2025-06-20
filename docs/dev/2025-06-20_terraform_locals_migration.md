# Terraform Variables to Locals Migration

## Overview

This document describes the migration of Terraform variables to locals in the infrastructure/terraform directory. The goal was to convert as many variables as possible to locals for better maintainability and to hardcode specific values for the project.

## Changes Made

1. Created a new `locals.tf` file with the following values:
   ```terraform
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
   }
   ```

2. Removed the following variables from `variables.tf`:
   - `project_id`
   - `region`
   - `build_topic_name`
   - `app_topic_name`
   - `build_subscription_name`
   - `app_subscription_name`
   - `service_account_name`

3. Updated all references to these variables in `main.tf` and `outputs.tf` to use the locals instead.

4. Updated `terraform.tfvars.example` to reflect these changes.

## Retained Variables

The following variable was retained for flexibility:
- `environment`: To allow different environments to be specified without changing the code.

## Benefits

1. Hardcoded project-specific values directly in the code
2. Reduced the need for variable files
3. Improved code readability by grouping related configuration together
4. Made it clearer which values are expected to change (variables) vs. which are fixed for the project (locals)