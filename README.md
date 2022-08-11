# Cell_specific_attenuation
efficient and simple implementation of cell specific attenuation using Cohen-Sutherland algorithm for GMM development


## Highlights:
1. pre-specify the static types of input and output of the function
2. use JIT (Just-In-Time) compliation with LLVM (Low Level Virtual Machine) complier to achieve the same speed as C
3. use fastmath optimization and Intel SVML (short vector math library) intrinsic functions to
 override the strict IEEE 754 compliance for floating point arithmetics

## How to Use
There are three ways to run the script, details are documented in the script's header as well as comments. See the test.ipynb for more details.
1. use JIT version (clipping_jit_version.py) which requires installation of numpy and numba. Just directly import the clipping_jit_version.py module

2. use precompiled version (the clipping.cp38-win_amd64.pyd file, currently compiled only for 64bit win10), which requires numpy installation only.
 Just directly import the module name: clipping. This precompiled module only works for 64bit win10 platform and is not optimized for your specific CPU model.
 
3. precompile from  the source code, which requires numpy and numba. Just run the clipping_precompiled_src.py as the main module,
a .pyd file should be genearated for the current platform and CPU, which can be redistributed to other similar platform directly.



## Reference:
Liu, C., Macedo, J., & Kottke, A. R. (2022). Evaluating the performance of nonergodic ground motion models in the ridgecrest area. Bulletin of Earthquake Engineering, 1-27.

Cite as:
@article{liu2022evaluating,
  title={Evaluating the performance of nonergodic ground motion models in the ridgecrest area},
  author={Liu, Chenying and Macedo, Jorge and Kottke, Albert R},
  journal={Bulletin of Earthquake Engineering},
  pages={1--27},
  year={2022},
  publisher={Springer}
}
