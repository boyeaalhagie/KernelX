#!/bin/bash
#SBATCH --job-name="kx_diag_v100"
#SBATCH --output=diag_v100_%j.out
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1

echo "=== Node ==="
hostname

echo "=== CUDA paths ==="
find /usr/local -maxdepth 2 -name nvcc 2>/dev/null
ls /usr/local/ | grep -i cuda 2>/dev/null || true

echo "=== cmake ==="
cmake --version 2>/dev/null || echo "cmake not found"
find /usr /opt /data -maxdepth 5 -name cmake -type f 2>/dev/null | head -5

echo "=== module system ==="
module --version 2>/dev/null || true
module list 2>&1 || true
module load cuda 2>&1 || true
which nvcc 2>/dev/null || echo "nvcc not in PATH"
echo "PATH=$PATH"
