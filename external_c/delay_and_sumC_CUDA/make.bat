@ECHO OFF
nvcc delay_and_sum_nearest_DP_CUDA.cu delay_and_sum_nearest_SP_CUDA.cu delay_and_sum_linear_DP_CUDA.cu delay_and_sum_linear_SP_CUDA.cu -o delay_and_sumC_CUDA.lib --lib --compiler-options="/openmp /Ox /MD"
rem python setup.py build_ext --inplace
