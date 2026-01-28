# HE-300 Development Environment
# Deploys via Docker Compose to local or remote host

terraform {
  required_version = ">= 1.5.0"
}

# Docker Compose deployment
module "he300" {
  source = "../../modules/docker-compose"

  environment = "dev"
  deploy_path = var.deploy_path
  target_host = var.target_host
  auto_deploy = var.auto_deploy

  # Ports
  cirisnode_port = 8000
  eee_port       = 8080
  dashboard_port = 3000

  # Features
  enable_gpu       = var.enable_gpu
  enable_dashboard = true

  # Model settings
  default_model      = var.default_model
  model_quantization = var.model_quantization
  model_cache_dir    = "${var.data_dir}/models"

  # GPU
  cuda_visible_devices = var.cuda_visible_devices

  # Paths
  data_dir  = var.data_dir
  log_dir   = var.log_dir
  log_level = "DEBUG"

  # Secrets (leave empty to auto-generate)
  db_password    = var.db_password
  redis_password = var.redis_password
  jwt_secret     = var.jwt_secret
  webhook_secret = var.webhook_secret
}

# Variables
variable "deploy_path" {
  description = "Path to HE-300 installation"
  type        = string
  default     = "/opt/he300"
}

variable "target_host" {
  description = "Target host (localhost or IP)"
  type        = string
  default     = "localhost"
}

variable "auto_deploy" {
  description = "Auto-deploy after config generation"
  type        = bool
  default     = false
}

variable "enable_gpu" {
  description = "Enable GPU support"
  type        = bool
  default     = true
}

variable "default_model" {
  description = "Default model for inference"
  type        = string
  default     = "Qwen/Qwen2.5-7B-Instruct"
}

variable "model_quantization" {
  description = "Model quantization level"
  type        = string
  default     = "Q4_K_M"
}

variable "cuda_visible_devices" {
  description = "CUDA devices to use"
  type        = string
  default     = "0"
}

variable "data_dir" {
  description = "Data directory"
  type        = string
  default     = "/var/lib/he300"
}

variable "log_dir" {
  description = "Log directory"
  type        = string
  default     = "/var/log/he300"
}

variable "db_password" {
  type      = string
  default   = ""
  sensitive = true
}

variable "redis_password" {
  type      = string
  default   = ""
  sensitive = true
}

variable "jwt_secret" {
  type      = string
  default   = ""
  sensitive = true
}

variable "webhook_secret" {
  type      = string
  default   = ""
  sensitive = true
}

# Outputs
output "cirisnode_url" {
  value = module.he300.cirisnode_url
}

output "eee_url" {
  value = module.he300.eee_url
}

output "dashboard_url" {
  value = module.he300.dashboard_url
}

output "deploy_command" {
  value = module.he300.deploy_command
}

output "env_file" {
  value = module.he300.env_file_path
}
