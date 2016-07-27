@ECHO OFF
python setup.py clean --all
del *.lib *.pyd
del delay_and_sumC_CUDA.c
del delay_and_sumC_CUDA.cpp
