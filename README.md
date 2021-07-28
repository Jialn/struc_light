# struc_light

## Installation
Tested on ubuntu18.04 + python3.6 + cuda10.2 and win10 + python3.8 + cuda11

  ```
  pip3 install numpy opencv-python open3d pycuda numba
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
