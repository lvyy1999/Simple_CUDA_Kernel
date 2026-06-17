#!/bin/bash

echo "========================================"
echo " Hardware Information"
echo "========================================"

echo ""
echo "CPU Information"
echo "----------------------------------------"

if command -v lscpu >/dev/null 2>&1; then
    CPU_MODEL=$(lscpu | grep "Model name" | sed 's/Model name:[ \t]*//')
    CPU_CORES=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
    CPU_THREADS_PER_CORE=$(lscpu | grep "Thread(s) per core" | awk '{print $4}')
    CPU_CORES_PER_SOCKET=$(lscpu | grep "Core(s) per socket" | awk '{print $4}')
    CPU_SOCKETS=$(lscpu | grep "Socket(s)" | awk '{print $2}')

    echo "CPU Model           : ${CPU_MODEL}"
    echo "Total CPU Threads   : ${CPU_CORES}"
    echo "Threads per Core    : ${CPU_THREADS_PER_CORE}"
    echo "Cores per Socket    : ${CPU_CORES_PER_SOCKET}"
    echo "Sockets             : ${CPU_SOCKETS}"
else
    echo "lscpu command not found."
    echo "Trying /proc/cpuinfo..."

    CPU_MODEL=$(grep -m 1 "model name" /proc/cpuinfo | cut -d ':' -f2 | sed 's/^ //')
    CPU_CORES=$(grep -c "^processor" /proc/cpuinfo)

    echo "CPU Model           : ${CPU_MODEL}"
    echo "Total CPU Threads   : ${CPU_CORES}"
fi

echo ""
echo "GPU Information"
echo "----------------------------------------"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

    echo ""
    echo "Detailed GPU Info"
    echo "----------------------------------------"
    nvidia-smi
else
    echo "nvidia-smi command not found."
    echo "No NVIDIA GPU detected or NVIDIA driver is not installed."
fi

echo ""
echo "CUDA Information"
echo "----------------------------------------"

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "nvcc command not found."
fi

echo ""
