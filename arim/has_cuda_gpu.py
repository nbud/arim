#!/usr/bin/env python3

"""
Created on Mon July 18 14:23:38 2016

@author: rb13781

Module for determining whether system has compatible CUDA GPU

Will attempt to find executable has_cuda_gpu.exe within same folder as this
py file unless specified differently via pre-defining the function_folder during
calling of this python function.

Returns:
    has_gpu , an integer. Values possible are (0) = No GPU Found (1) GPU Available.

"""

__all__ = ['test']

has_gpu=0
gpu_checked=0

def test(function_folder=None):
    global has_gpu
    global gpu_checked
    import os
    
    if os.name=='nt':
        # Windows OS In use
        pass
    else:
        #Not currently programmed for (i.e. no appropriate executable to call). Terminating.
        print("Expected Windows Operating System. Current OS not yet available. Skipping GPU Check")
        has_gpu=0
        gpu_checked=0
        return has_gpu
        #raise SystemError
    
    if function_folder is not None:
        cmd=function_folder + 'has_cuda_gpu.exe'
    else:
        function_folder=os.path.dirname( os.path.realpath( __file__ ) )
        cmd=function_folder+'/has_cuda_gpu.exe'
    try:
        has_gpu=os.system(cmd)
        # Will return 0 if no GPU CUDA devices found, or <0 if found. abs(has_gpu) = No. of GPU CUDA devices.
        # Reason being: returns error value = 1 if no executable found.
    except OSError as e:
        pass

    if has_gpu == 1:
        print("Could not locate GPU Checking function. Using CPU") 
        has_gpu=0
    elif has_gpu == 0:
        print("CUDA Device not detected. Using CPU")
        has_gpu=0
    else:
        print("GPU CUDA option available if desired")
        has_gpu=1
    
    gpu_checked=1    
    return has_gpu