# HE-300 Deployment Policy
# Grants read access to secrets needed for benchmark deployment

# API keys for services
path "secret/data/he300/api-keys/*" {
  capabilities = ["read", "list"]
}

# Database credentials
path "secret/data/he300/database/*" {
  capabilities = ["read"]
}

# WireGuard keys
path "secret/data/he300/wireguard/*" {
  capabilities = ["read"]
}

# TLS certificates
path "secret/data/he300/tls/*" {
  capabilities = ["read"]
}

# JWT signing keys
path "secret/data/he300/jwt/*" {
  capabilities = ["read"]
}

# Dynamic database credentials
path "database/creds/he300-postgres" {
  capabilities = ["read"]
}

# Allow token self-lookup
path "auth/token/lookup-self" {
  capabilities = ["read"]
}
