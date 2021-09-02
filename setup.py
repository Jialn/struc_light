############
# run with: python setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize
import sys, os, shutil


module_list = ["depth_map_utils", "stereo_rectify", "structured_light_cuda"]

if sys.platform == 'win32': sys_appendix = ".cp38-win_amd64.pyd"
elif os.uname()[4] == "aarch64": sys_appendix = ".cpython-38-aarch64-linux-gnu.so"  # arm64 linux
else: sys_appendix = ".cpython-38-x86_64-linux-gnu.so"

# compile and copy files
for module_name in module_list:
    setup(
        ext_modules = cythonize(
        module_name + ".py",
    ))

for module_name in module_list:
    shutil.copy(module_name + sys_appendix, "../x3d_camera/")

# cleaning
for module_name in module_list:
    os.remove(module_name + sys_appendix)
    os.remove(module_name + ".c")

# copy cuda bin file
shutil.copy("structured_light_cuda_core.cubin", "../x3d_camera/")

