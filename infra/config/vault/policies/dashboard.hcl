# Dashboard Policy
# Access for the results dashboard service

# Read only API keys needed for verification
path "secret/data/he300/api-keys/dashboard" {
  capabilities = ["read"]
}

# Read webhook secrets
path "secret/data/he300/webhook/*" {
  capabilities = ["read"]
}

# Read database credentials
path "secret/data/he300/database/dashboard" {
  capabilities = ["read"]
}

# Read S3 credentials for artifacts
path "secret/data/he300/s3/*" {
  capabilities = ["read"]
}
