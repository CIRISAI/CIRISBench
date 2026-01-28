# Terraform Deploy Policy
# Access for Terraform to manage infrastructure secrets

# Read infrastructure secrets
path "secret/data/he300/terraform/*" {
  capabilities = ["read", "list"]
}

# Read cloud provider credentials
path "secret/data/he300/aws/*" {
  capabilities = ["read"]
}

path "secret/data/he300/lambda/*" {
  capabilities = ["read"]
}

# Write deployment state secrets
path "secret/data/he300/deployments/*" {
  capabilities = ["create", "read", "update"]
}

# Read SSH keys for provisioning
path "secret/data/he300/ssh/*" {
  capabilities = ["read"]
}
