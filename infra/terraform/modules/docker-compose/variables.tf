# Docker Compose Deployment Variables

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "deploy_path" {
  description = "Path where Docker Compose files are located"
  type        = string
  default     = "/opt/he300"
}

variable "target_host" {
  description = "Target host for deployment (localhost or IP/hostname)"
  type        = string
  default     = "localhost"
}

variable "ssh_user" {
  description = "SSH user for remote deployment"
  type        = string
  default     = "root"
}

variable "docker_host" {
  description = "Docker host URL (e.g., unix:///var/run/docker.sock or tcp://host:2376)"
  type        = string
  default     = ""
}

variable "auto_deploy" {
  description = "Automatically deploy after generating configs"
  type        = bool
  default     = false
}

# Service Ports
variable "cirisnode_port" {
  description = "CIRISNode API port"
  type        = number
  default     = 8000
}

variable "eee_port" {
  description = "EthicsEngine Enterprise API port"
  type        = number
  default     = 8080
}

variable "dashboard_port" {
  description = "Dashboard web UI port"
  type        = number
  default     = 3000
}

# Feature Flags
variable "enable_gpu" {
  description = "Enable GPU support in Docker Compose"
  type        = bool
  default     = true
}

variable "enable_dashboard" {
  description = "Enable the results dashboard"
  type        = bool
  default     = true
}

# Model Configuration
variable "default_model" {
  description = "Default model for inference"
  type        = string
  default     = "Qwen/Qwen2.5-7B-Instruct"
}

variable "model_quantization" {
  description = "Model quantization (Q4_K_M, Q5_K_M, Q8_0, none)"
  type        = string
  default     = "Q4_K_M"
}

variable "model_cache_dir" {
  description = "Directory for caching models"
  type        = string
  default     = "/var/lib/he300/models"
}

variable "cuda_visible_devices" {
  description = "CUDA devices to use (e.g., 0, 0,1, or all)"
  type        = string
  default     = "0"
}

# Secrets (leave empty to auto-generate)
variable "db_password" {
  description = "Database password (auto-generated if empty)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "redis_password" {
  description = "Redis password (auto-generated if empty)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "jwt_secret" {
  description = "JWT secret (auto-generated if empty)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "webhook_secret" {
  description = "Webhook secret (auto-generated if empty)"
  type        = string
  default     = ""
  sensitive   = true
}

# Paths
variable "data_dir" {
  description = "Data directory for results and artifacts"
  type        = string
  default     = "/var/lib/he300"
}

variable "log_dir" {
  description = "Log directory"
  type        = string
  default     = "/var/log/he300"
}

variable "log_level" {
  description = "Logging level (DEBUG, INFO, WARNING, ERROR)"
  type        = string
  default     = "INFO"
}
