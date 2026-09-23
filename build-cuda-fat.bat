@echo off
REM Real code for Turing through Blackwell plus compute_90 PTX, so newer cards JIT.
REM ggml rewrites 120/121 to 120a/121a.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "CUDA_PATH=B:\cuda-toolkit"
set "PATH=B:\cuda-toolkit\bin;%PATH%"
set "CUDAToolkit_ROOT=B:\cuda-toolkit"
cd /d B:\llama.cpp-src
echo === configuring ===
cmake -B build-cuda -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120-real;121-real;90-virtual" ^
  -DCMAKE_CUDA_COMPILER=B:/cuda-toolkit/bin/nvcc.exe ^
  -DCUDAToolkit_ROOT=B:/cuda-toolkit ^
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_CURL=OFF
echo === building llama-server llama-quantize ===
cmake --build build-cuda --target llama-server llama-quantize --config Release -j 16
