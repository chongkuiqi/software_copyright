# 进行编译 python setup.py build_ext --inplace
from setuptools import setup
from setuptools import Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

def make_cuda_ext(name, module, sources):
    define_macros = []
    define_macros += [("WITH_CUDA", None)]
    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })

# example_module = Extension(name='numpy_demo',  # 模块名称
#                            sources=['example.cpp'],    # 源码
#                            include_dirs=[r'/home/data/CM/10_device/pybind11/include']
#                            )

setup(
    ext_modules=[
        # 可变形卷积和可变形池化
        make_cuda_ext(
                name='deform_conv_cuda',
                module='models.dcn',
                sources=[
                    'src/deform_conv_cuda.cpp',
                    'src/deform_conv_cuda_kernel.cu'
                ]),
        make_cuda_ext(
            name='deform_pool_cuda',
            module='models.dcn',
            sources=[
                'src/deform_pool_cuda.cpp',
                'src/deform_pool_cuda_kernel.cu'
            ]),
        

        # 旋转框iou计算方法
        make_cuda_ext(
            name='box_iou_rotated_cuda',
            module='utils.box_iou_rotated',
            sources=[
                'src/box_iou_rotated_cpu.cpp',
                'src/box_iou_rotated_cuda.cu'
            ]),
        
        # # 旋转框多类别NMS
        # make_cuda_ext(
        #         name='nms_rotated_cuda',
        #         module='utils.nms_rotated',
        #         sources=['src/nms_rotated_cpu.cpp', 'src/nms_rotated_cuda.cu']),
        make_cuda_ext(
            name='ml_nms_rotated_cuda',
            module='utils.ml_nms_rotated',
            sources=['src/nms_rotated_cpu.cpp', 'src/nms_rotated_cuda.cu']),
        
    ],
        
    #  cmdclass 为python setup.py build_ext命令,指定为BuildExtension
    cmdclass={'build_ext': BuildExtension},
)

