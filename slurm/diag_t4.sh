#!/bin/bash
#SBATCH --job-name="kx_diag_t4"
#SBATCH --output=diag_t4_%j.out
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1

echo "=== Node ==="
hostname

echo "=== CUDA paths ==="
find /usr/local -maxdepth 2 -name nvcc 2>/dev/null
ls /usr/local/ | grep cuda

echo "=== cmake ==="
cmake --version 2>/dev/null || echo "cmake not in PATH"
find /usr /opt /home -maxdepth 5 -name cmake -type f 2>/dev/null | head -5

echo "=== PATH after module load cuda/12.1 ==="
module load cuda/12.1 2>&1
which nvcc 2>/dev/null || echo "nvcc not in PATH after module load"
echo "PATH=$PATH"
