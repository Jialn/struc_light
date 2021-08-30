from distutils.core import setup
from Cython.Build import cythonize
import shutil

# run with: python.exe .\setup.py build_ext --inplace

setup(
    ext_modules = cythonize(
    "depth_map_utils.py",
))
setup(
    ext_modules = cythonize(
    "stereo_rectify.py",
))
setup(
    ext_modules = cythonize(
    "structured_light_cuda.py",
))

shutil.copy("depth_map_utils.cp38-win_amd64.pyd", "../x3d_camera/")
shutil.copy("stereo_rectify.cp38-win_amd64.pyd", "../x3d_camera/")
shutil.copy("structured_light_cuda.cp38-win_amd64.pyd", "../x3d_camera/")
shutil.copy("structured_light_cuda_core.cubin", "../x3d_camera/")