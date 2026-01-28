#!/bin/bash
#
# install-nvidia-docker.sh - Install NVIDIA drivers and container toolkit
#
# GPU: A10 (24 GB PCIe) - Detected via lspci
# Server: Lambda Labs Ubuntu instance
#
# HITL Reference: docs/hitl.md
# Security Reference: docs/sec.md
#
# ⚠️ SECURITY NOTE: This installs NVIDIA drivers with root privileges
# The packages come from nvidia.github.io and Ubuntu repositories
#

set -e

HOST="ubuntu@163.192.58.165"

echo "=========================================="
echo "  NVIDIA Driver & Container Toolkit Setup"
echo "  GPU: A10 (24 GB VRAM)"
echo "=========================================="
echo ""
echo "⚠️  This will install:"
echo "    - NVIDIA drivers (from Ubuntu repos)"
echo "    - NVIDIA Container Toolkit (from nvidia.github.io)"
echo ""
echo "Press Enter to continue or Ctrl+C to abort..."
read

# Step 1: Install NVIDIA drivers
echo ""
echo "[1/5] Installing NVIDIA drivers..."
echo "     This may take several minutes..."
ssh $HOST "sudo apt-get update && sudo apt-get install -y nvidia-driver-550 nvidia-utils-550"

# Step 2: Install NVIDIA Container Toolkit
echo ""
echo "[2/5] Adding NVIDIA Container Toolkit repository..."
ssh $HOST "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
ssh $HOST "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"

echo ""
echo "[3/5] Installing NVIDIA Container Toolkit..."
ssh $HOST "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"

# Step 3: Configure Docker for NVIDIA
echo ""
echo "[4/5] Configuring Docker for NVIDIA runtime..."
ssh $HOST "sudo nvidia-ctk runtime configure --runtime=docker"
ssh $HOST "sudo systemctl restart docker"

# Step 4: Verify installation
echo ""
echo "[5/5] Verifying installation..."
echo ""
echo "Driver version:"
ssh $HOST "cat /proc/driver/nvidia/version 2>/dev/null || echo 'Driver may require reboot'"
echo ""
echo "Docker NVIDIA support:"
ssh $HOST "docker info 2>/dev/null | grep -i nvidia || echo 'May require reboot'"

echo ""
echo "=========================================="
echo "  Installation Complete"
echo "=========================================="
echo ""
echo "⚠️  A REBOOT may be required for drivers to load."
echo ""
echo "To reboot (will disconnect SSH):"
echo "  ssh $HOST 'sudo reboot'"
echo ""
echo "After reboot, verify with:"
echo "  ssh $HOST 'nvidia-smi'"
echo ""
echo "Then run:"
echo "  ./setup-gpu-ollama.sh"
echo ""
