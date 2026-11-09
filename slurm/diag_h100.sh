#!/bin/bash
#SBATCH --job-name="kx_diag_h100"
#SBATCH --output=diag_h100_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_username@your_institution.edu
#SBATCH --partition=dgxh100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1

echo "=== Node ==="
hostname
date

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader

echo ""
echo "=== CUDA toolkits installed ==="
ls /usr/local/ | grep -i cuda

echo ""
echo "=== cmake ==="
cmake --version

echo ""
echo "=== Available CUDA modules ==="
module avail cuda 2>&1

echo ""
echo "=== Try loading each CUDA module and check nvcc ==="
for VER in 12.9 12.5 12.1 11.7; do
    module load cuda/${VER} 2>/dev/null && \
        echo "cuda/${VER}: nvcc=$(which nvcc 2>/dev/null)  version=$(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')" || \
        echo "cuda/${VER}: failed to load"
    module unload cuda 2>/dev/null
done
