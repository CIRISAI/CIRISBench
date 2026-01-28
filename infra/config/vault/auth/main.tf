# HE-300 Vault Auth Configuration
# Terraform configuration for Vault authentication methods

terraform {
  required_providers {
    vault = {
      source  = "hashicorp/vault"
      version = "~> 3.0"
    }
  }
}

# GitHub Actions OIDC Authentication
resource "vault_jwt_auth_backend" "github" {
  path               = "jwt-github"
  type               = "jwt"
  oidc_discovery_url = "https://token.actions.githubusercontent.com"
  bound_issuer       = "https://token.actions.githubusercontent.com"
  
  default_role = "he300-deploy"
  
  tune {
    default_lease_ttl = "1h"
    max_lease_ttl     = "4h"
    token_type        = "default-service"
  }
}

# Role for HE-300 benchmark deployment from GitHub Actions
resource "vault_jwt_auth_backend_role" "he300_deploy" {
  backend        = vault_jwt_auth_backend.github.path
  role_name      = "he300-deploy"
  token_policies = ["he300-deploy"]

  bound_claims = {
    repository = "rng-ops/he300-integration"
  }
  
  bound_audiences = ["sigstore"]
  user_claim      = "actor"
  role_type       = "jwt"
  token_ttl       = 3600
  token_max_ttl   = 14400
}

# Role for Terraform infrastructure deployment
resource "vault_jwt_auth_backend_role" "terraform_deploy" {
  backend        = vault_jwt_auth_backend.github.path
  role_name      = "terraform-deploy"
  token_policies = ["terraform-deploy"]

  bound_claims = {
    repository = "rng-ops/he300-integration"
    ref        = "refs/heads/main"
  }
  
  bound_claims_type = "glob"
  bound_audiences   = ["sigstore"]
  user_claim        = "actor"
  role_type         = "jwt"
  token_ttl         = 3600
  token_max_ttl     = 7200
}

# AppRole Authentication for CI/CD pipelines
resource "vault_auth_backend" "approle" {
  type = "approle"
  path = "approle"
}

# AppRole for automated installer scripts
resource "vault_approle_auth_backend_role" "installer" {
  backend        = vault_auth_backend.approle.path
  role_name      = "he300-installer"
  token_policies = ["he300-deploy"]
  
  token_ttl         = 1800   # 30 minutes
  token_max_ttl     = 3600   # 1 hour
  secret_id_ttl     = 86400  # 24 hours
  secret_id_num_uses = 10
}

# AppRole for GPU host services
resource "vault_approle_auth_backend_role" "gpu_host" {
  backend        = vault_auth_backend.approle.path
  role_name      = "he300-gpu-host"
  token_policies = ["he300-deploy"]
  
  token_ttl         = 3600   # 1 hour
  token_max_ttl     = 86400  # 24 hours
  token_num_uses    = 0      # Unlimited
  secret_id_ttl     = 0      # Never expires
  secret_id_num_uses = 0     # Unlimited
  
  # Bind to specific CIDR if needed
  # token_bound_cidrs = ["10.0.0.0/24"]
}

# AppRole for dashboard service
resource "vault_approle_auth_backend_role" "dashboard" {
  backend        = vault_auth_backend.approle.path
  role_name      = "he300-dashboard"
  token_policies = ["dashboard"]
  
  token_ttl         = 3600
  token_max_ttl     = 86400
  token_num_uses    = 0
  secret_id_ttl     = 0
  secret_id_num_uses = 0
}

# Outputs for use in other configurations
output "github_jwt_path" {
  value = vault_jwt_auth_backend.github.path
}

output "approle_path" {
  value = vault_auth_backend.approle.path
}

output "installer_role_id" {
  value     = vault_approle_auth_backend_role.installer.role_id
  sensitive = true
}
