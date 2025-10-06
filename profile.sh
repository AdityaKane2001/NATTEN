echo "****** Profiling 1D use case..."
echo "--- Profiling Flash Attention..."
python3 -u -m natten.profiler --backprop --fmha-backend cutlass-fmha --optimize -n 24 -d 128 -i 32768
python3 -u -m natten.profiler --backprop --backend fa -n 24 -d 128 -i 32768

echo "****** Profiling 2D use case..."
echo "--- Profiling Flash Attention..."
python3 -u -m natten.profiler --backprop --fmha-backend cutlass-fmha --optimize -n 24 -d 128 -i 256 256
python3 -u -m natten.profiler --backprop --backend fa -n 24 -d 128 -i 256 256

echo "****** Profiling 3D use case..."
echo "--- Profiling Flash Attention..."
python3 -u -m natten.profiler --backprop --fmha-backend cutlass-fmha --optimize -n 24 -d 128 -i 30 48 80
python3 -u -m natten.profiler --backprop --backend fa -n 24 -d 128 -i 30 48 80

# echo "****** Profiling 1D use case..."
# echo "--- Profiling Flash Attention..."
# python3 -u -m natten.profiler --backprop --backend fav2 -n 24 -d 128 -i 32768
# 
# echo "****** Profiling 2D use case..."
# echo "--- Profiling Flash Attention..."
# python3 -u -m natten.profiler --backprop --backend fav2 -n 24 -d 128 -i 256 256
# 
# echo "****** Profiling 3D use case..."
# echo "--- Profiling Flash Attention..."
# python3 -u -m natten.profiler --backprop --backend fav2 -n 24 -d 128 -i 30 48 80


# echo "--- Profiling cudnn..."
# python3 -u -m natten.profiler --backprop \
#     --backend cudnn \
#     -d 128 \
#     -i 32768
# 
# echo "--- Profiling xformers..."
# python3 -u -m natten.profiler --backprop \
#     --backend xformers \
#     -d 128 \
#     -i 32768

# echo "--- Profiling cudnn..."
# python3 -u -m natten.profiler --backprop \
#     --backend cudnn \
#     -n 24 \
#     -d 128 \
#     -i 256 256
# 
# echo "--- Profiling xformers..."
# python3 -u -m natten.profiler --backprop \
#     --backend xformers \
#     -n 24 \
#     -d 128 \
#     -i 256 256

# echo "--- Profiling cudnn..."
# python3 -u -m natten.profiler --backprop \
#     --backend cudnn \
#     -n 24 \
#     -d 128 \
#     -i 30 48 80
# 
# echo "--- Profiling xformers..."
# python3 -u -m natten.profiler --backprop \
#     --backend xformers \
#     -n 24 \
#     -d 128 \
#     -i 30 48 80
