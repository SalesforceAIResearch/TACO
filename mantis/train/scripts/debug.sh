# Get the list of GPU indices
gpu_indices=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)

# Check if GPU indices were found
if [ -n "$gpu_indices" ]; then
    CUDA_VISIBLE_DEVICES=$gpu_indices
    echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
else
    echo "No GPUs found or an error occurred."
fi
