# Docker Compose Deployment Outputs

output "env_file_path" {
  description = "Path to generated .env file"
  value       = local_file.env_file.filename
}

output "compose_override_path" {
  description = "Path to generated docker-compose.override.yml"
  value       = local_file.compose_override.filename
}

output "cirisnode_url" {
  description = "CIRISNode API URL"
  value       = "http://${var.target_host}:${var.cirisnode_port}"
}

output "eee_url" {
  description = "EthicsEngine Enterprise API URL"
  value       = "http://${var.target_host}:${var.eee_port}"
}

output "dashboard_url" {
  description = "Dashboard URL"
  value       = var.enable_dashboard ? "http://${var.target_host}:${var.dashboard_port}" : null
}

output "db_password" {
  description = "Database password (generated or provided)"
  value       = var.db_password != "" ? var.db_password : random_password.db_password.result
  sensitive   = true
}

output "jwt_secret" {
  description = "JWT secret (generated or provided)"
  value       = var.jwt_secret != "" ? var.jwt_secret : random_password.jwt_secret.result
  sensitive   = true
}

output "webhook_secret" {
  description = "Webhook secret (generated or provided)"
  value       = var.webhook_secret != "" ? var.webhook_secret : random_password.webhook_secret.result
  sensitive   = true
}

output "deploy_command" {
  description = "Command to manually deploy"
  value       = "cd ${var.deploy_path} && docker compose up -d"
}

output "status_command" {
  description = "Command to check deployment status"
  value       = "cd ${var.deploy_path} && docker compose ps"
}
