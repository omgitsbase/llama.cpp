@echo off
REM The Windows CUDA runtime llmash ships: real code for Turing through Blackwell plus compute_90 PTX for newer
REM cards to JIT (ggml rewrites 120/121 to 120a/121a), and the CPU backend once per instruction set, every backend
REM a DLL loaded at start. Needs Visual Studio 2022 and CUDA 13, with CUDA_PATH naming the toolkit.
setlocal
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set "VS=%%i"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 || exit /b 1
if not defined CUDA_PATH (
    echo CUDA_PATH is not set
    exit /b 1
)
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d "%~dp0"
cmake -B build-cuda-dl -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF ^
  -DGGML_CUDA=ON -DGGML_CUDA_COMPRESSION_MODE=size ^
  -DCMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120-real;121-real;90-virtual" ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF || exit /b 1
cmake --build build-cuda-dl -j %NUMBER_OF_PROCESSORS% || exit /b 1
REM llama-server finds the CUDA runtime beside it
copy /y "%CUDA_PATH%\bin\cudart64_13.dll" build-cuda-dl\bin >nul
copy /y "%CUDA_PATH%\bin\cublas64_13.dll" build-cuda-dl\bin >nul
copy /y "%CUDA_PATH%\bin\cublasLt64_13.dll" build-cuda-dl\bin >nul
