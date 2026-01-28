# HE-300 Production Environment

terraform {
  required_version = ">= 1.5.0"
}

module "he300" {
  source = "../../modules/docker-compose"

  environment = "prod"
  deploy_path = var.deploy_path
  target_host = var.target_host
  auto_deploy = var.auto_deploy

  cirisnode_port = 8000
  eee_port       = 8080
  dashboard_port = 3000

  enable_gpu       = var.enable_gpu
  enable_dashboard = true

  default_model      = var.default_model
  model_quantization = var.model_quantization
  model_cache_dir    = "${var.data_dir}/models"

  cuda_visible_devices = var.cuda_visible_devices

  data_dir  = var.data_dir
  log_dir   = var.log_dir
  log_level = "WARNING"

  db_password    = var.db_password
  redis_password = var.redis_password
  jwt_secret     = var.jwt_secret
  webhook_secret = var.webhook_secret
}

variable "deploy_path" {
  type    = string
  default = "/opt/he300"
}

variable "target_host" {
  type    = string
  default = "localhost"
}

variable "auto_deploy" {
  type    = bool
  default = false
}

variable "enable_gpu" {
  type    = bool
  default = true
}

variable "default_model" {
  type    = string
  default = "Qwen/Qwen2.5-7B-Instruct"
}

variable "model_quantization" {
  type    = string
  default = "Q4_K_M"
}

variable "cuda_visible_devices" {
  type    = string
  default = "0"
}

variable "data_dir" {
  type    = string
  default = "/var/lib/he300"
}

variable "log_dir" {
  type    = string
  default = "/var/log/he300"
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
