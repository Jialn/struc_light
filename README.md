# struc_light

## Installation
Tested on ubuntu18.04 + python3.6 and win10 + python3.8.

  ```
  pip3 install numpy opencv-python numba open3d
  ```

## Running examples
  ```
  python3 structured_light.py pattern_examples\struli_test1\
  ```
  or on linux:
  ```
  python3 structured_light.py pattern_examples/struli_test1/
  ```

  Result is in ".\pattern_examples\struli_test1\res". You can use meshlab to visualize the ply file. https://www.meshlab.net/#download

  When runing for the first time or src code is modified, it will need more time for numba compiling.
  
  To make running faster, set option ```use_parallel_computing``` to True in "structured_light.py". Note this may cause numba llvm compiling fails for some version of numba.
