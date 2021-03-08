from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="nms",
    packages=["nms"],
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            "nms.details",
            ["src/nms.cpp", "src/nms_kernel.cu"],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
