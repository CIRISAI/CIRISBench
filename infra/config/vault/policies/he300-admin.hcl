# HE-300 Admin Policy
# Full access to all HE-300 secrets for administrators

# Full access to all he300 secrets
path "secret/data/he300/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/metadata/he300/*" {
  capabilities = ["read", "list", "delete"]
}

# Manage database roles
path "database/roles/he300-*" {
  capabilities = ["create", "read", "update", "delete"]
}

path "database/creds/he300-*" {
  capabilities = ["read"]
}

# Manage policies
path "sys/policies/acl/he300-*" {
  capabilities = ["create", "read", "update", "delete"]
}

# Manage auth methods
path "auth/approle/role/he300-*" {
  capabilities = ["create", "read", "update", "delete"]
}

path "auth/jwt-github/role/he300-*" {
  capabilities = ["create", "read", "update", "delete"]
}
