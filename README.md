# struc_light

## Installation
Tested on ubuntu18.04 + python3.6 + cuda10.2 and win10 + python3.8 + cuda11

  ```
  pip3 install numpy opencv-python open3d pycuda numba Cython
  ```

- 安装Python3.8, 建议 Python3.8.6 or Python3.8.10，安装时注意选择把python添加进path
- 下载安装VS2019 (Linux跳过此步): https://visualstudio.microsoft.com/zh-hans/vs/community/
    安装VS2019完毕后需要添加"cl.exe"的环境变量，示例: `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64`

- 下载安装CUDA11.4：https://developer.nvidia.com/cuda-downloads
- 通过Pip安装packages
  ```
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  pip install numpy opencv-python pyserial hidapi open3d pycuda
  ```

## Running examples
  Windows:
  ```
  python.exe structured_light.py pattern_examples\struli_test1\
  python.exe structured_light_cuda.py pattern_examples\struli_test1\
  ```
  Linux:
  ```
  python3 structured_light.py pattern_examples/struli_test1/
  python3 structured_light_cuda.py pattern_examples/struli_test1/
  ```

  Result is in ".\pattern_examples\struli_test1\res". You can use meshlab to visualize the ply file. https://www.meshlab.net/#download

  When runing for the first time or src code is modified, it will need more time for numba or cuda compiling.
  
  To make running faster when using CPU, set option ```use_parallel_computing``` to True in "structured_light.py". Note this may cause numba llvm compiling fails for some version of numba.
