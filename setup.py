############
# run with: python.exe .\setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize
import os, shutil


module_list = ["depth_map_utils", "stereo_rectify", "structured_light_cuda"]

# compile and copy files
for module_name in module_list:
    setup(
        ext_modules = cythonize(
        module_name + ".py",
    ))

for module_name in module_list:
    shutil.copy(module_name + ".cp38-win_amd64.pyd", "../x3d_camera/")

# cleaning
for module_name in module_list:
    os.remove(module_name + ".cp38-win_amd64.pyd")
    os.remove(module_name + ".c")

# copy cuda bin file
shutil.copy("structured_light_cuda_core.cubin", "../x3d_camera/")

