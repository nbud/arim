//*****************************************************
// FILE CONTAINS 
//      GPU DEVICE KERNEL
//      EXTERNAL C CODE (GPU calls)
//*****************************************************

//=========================================================================================================
//=========================================================================================================
// (GPU Cuda Kernel) Double Precision Delay and Sum Algorithm, using linear interpolation 
//=========================================================================================================
//=========================================================================================================
__global__ void c_delay_and_sum_amplitudes_linear_Kernel_DP( const double* scanlines,  const unsigned int* tx,  const unsigned int* rx,  const double* lookup_times_tx,  const double* lookup_times_rx,
                 const double* amplitudes_tx,  const double* amplitudes_rx, double invdt, double t0, double fillvalue,double* result,
                int numpoints,int numsamples,int numelements, int numscanlines)
{
    int point = threadIdx.x + blockIdx.x * blockDim.x;
    if (point < numpoints)
    {
        int scan,lookup_index,t_ind,r_ind,set_val,pixie,point2,lookup_index1,set_val1;
        double tot_real1,tot_imag1,amp_corr,lookup_time,frac1;
        tot_real1=0.0;
        tot_imag1=0.0;
        pixie=numelements*point;
        for (scan=0;scan<numscanlines;scan++)
        {  
            t_ind=pixie+tx[scan];
            r_ind=pixie+rx[scan];
            lookup_time = (lookup_times_tx[t_ind] + lookup_times_rx[r_ind] - t0)* invdt;
            lookup_index = (int)lookup_time;
            frac1 = lookup_time - lookup_index;
            lookup_index1=lookup_index+1;
            if (lookup_index < 0)  
            {
            }
            else if (lookup_index1 >= numsamples)
            {
            }
            else
            {
                amp_corr= amplitudes_tx[t_ind] * amplitudes_rx[r_ind];
                set_val=scan*numsamples+lookup_index;
                set_val=set_val*2;
                set_val1=set_val+2;
                tot_real1 += amp_corr * scanlines[set_val]+frac1*(scanlines[set_val1]-scanlines[set_val]); 
                tot_imag1 += amp_corr * scanlines[set_val+1]+frac1*(scanlines[set_val1+1]-scanlines[set_val+1]); 
            }  
        }  
        point2=point*2;
        result[point2]=tot_real1;
        result[point2+1]=tot_imag1;  
    }    
}  

//=========================================================================================================
//=========================================================================================================
// DOUBLE PRECISION EXTERNAL C CODE (LINEAR INTERPOLATION)
//=========================================================================================================
//=========================================================================================================
int c_delayAndSum_Algorithm_linear_DP_CUDA(
    const double* scanlines,                     //INPUT double complex array as [:,:,2] 2 doubles [numscanlines x numsamples x 2]
    const unsigned int* tx,                      //INPUT unsigned int array [:] with size [numscanlines] transmitter
    const unsigned int* rx,                      //INPUT unsigned int array [:] with size [numscanlines] receiver
    const double* lookup_times_tx,               //INPUT double array [:,:] with size [numpoints * numelements] Flight time to transmitter from each point
    const double* lookup_times_rx,               //INPUT double array [:,:] with size [numpoints * numelements] Flight time to receiver from each point
    const double* amplitudes_tx,                 //INPUT double array [:,:] with size [numpoints * numelements] Transmitter Amplitude Array  
    const double* amplitudes_rx,                 //INPUT double array [:,:] with size [numpoints * numelements] Receiver Amplitude Array  
    double invdt,                                //INPUT double inverse time step 1/dt
    double t0,                                   //INPUT double initial time instance
    double fillvalue,                            //INPUT double fillvalue (if calculated lookup time outside sample range)
    double* result,                              //OUTPUT double array [:,2] with size [numpoints x 2 ] 2 doubles representing double complex.
    int numpoints,                               //INPUT int number of points in TFM
    int numsamples,                              //INPUT int number of time samples in scanlines
    int numelements,                             //INPUT int number of elements in Probe Array
    int numscanlines,                            //INPUT int number of scanlines - which is 0.5*(numelements*numelements+numelements) for HMC, numelements*numelements for FMC
    int NTHREADS,                                //INPUT int Number of threads (for GPU)
    int txUpdate,                                //INPUT int txUpdate = 0 (don't update Transmitter focal law information on GPU) 1 (Update this information) 
    int rxUpdate,                                //INPUT int rxUpdate = 0 (don't update Receiver focal law information on GPU) 1 (Update this information)
    int ExpDataUpdate,                           //INPUT int ExpDataUpdate = 0 (don't update scanlines information on GPU) 1 (Update this information)
    int finaliseOpt)                             //INPUT int finaliseOpt <0 Don't run finalise Stage, finaliseOpt = 0 finalise after processing, finaliseOpt=1 finalise and immediate return.

{
    /*********************************************************************************\
     Calculation of Linear Interpolation DELAY and SUM Algorithm 
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     Algorithm uses GPU
      
     When using GPU, Four stages can be considered
     1. Optional Initialisation (Memory Allocation on GPU) if not yet allocated.
     2. Transfer of information to GPU (altered values only)
     3. Calculation usign KERNEL, transfer of OUTPUT data from GPU to Host
     
     4. Optional Cleanup of GPU (Deallocation of GPU Memory)  
        STAGE 4 can be run independently. Only Stage 4 is run if finaliseOpt = 1
    
     
        
    \********************************************************************************/

    //Local
    int Nscan=numscanlines*numsamples;
    int Nlook=numpoints*numelements;
    //Input Data Pointers    
    static double *scanlines_gpu;
    static unsigned int *tx_gpu;
    static unsigned int *rx_gpu;
    static double *lookup_times_tx_gpu;
    static double *lookup_times_rx_gpu;
    static double *amplitudes_tx_gpu;
    static double *amplitudes_rx_gpu;
    //Output Data Pointers
    static double *result_gpu; 
    //Returned
    static int Marker=0;

    // Initialise GPU Device if required
    // Marker = 0 (GPU Memory not yet allocated)
    if (Marker < 1) //Not yet allocated
    {
        // Input
        cudaMalloc(&scanlines_gpu, Nscan*2*sizeof(double)); 
        cudaMalloc(&tx_gpu, numscanlines*sizeof(unsigned int)); 
        cudaMalloc(&rx_gpu, numscanlines*sizeof(unsigned int)); 
        cudaMalloc(&lookup_times_tx_gpu, Nlook*sizeof(double));
        cudaMalloc(&lookup_times_rx_gpu, Nlook*sizeof(double));
        cudaMalloc(&amplitudes_tx_gpu, Nlook*sizeof(double));
        cudaMalloc(&amplitudes_rx_gpu, Nlook*sizeof(double));
        // Output
        cudaMalloc(&result_gpu, numpoints*2*sizeof(double));
        //Mark that initialisation has occurred
        //======================================================================
        Marker=1;        
        // Ensure that data is transferred after initialisation
        txUpdate=1;
        rxUpdate=1;
        ExpDataUpdate=1;
    }
    
    // OPTIONAL USER FORCED CLEANUP (AND IMMEDIATE RETURN)
    if (finaliseOpt == 1)
    {
            cudaFree(scanlines_gpu);
            cudaFree(tx_gpu);
            cudaFree(rx_gpu);
            cudaFree(lookup_times_tx_gpu);
            cudaFree(lookup_times_rx_gpu);
            cudaFree(amplitudes_tx_gpu);
            cudaFree(amplitudes_rx_gpu);
            cudaFree(result_gpu);
            cudaDeviceReset();
            Marker=0;
            return Marker;
    }
    
    // Copy any new data
    //==========================================================================    
    
    if (ExpDataUpdate == 1)
    {
        //Transfer ExpData related data to GPU 
        //======================================================================
        cudaMemcpy(scanlines_gpu, scanlines, Nscan*2*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(tx_gpu, tx, numscanlines*sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(rx_gpu, rx, numscanlines*sizeof(unsigned int), cudaMemcpyHostToDevice);  
    }
    
    if (txUpdate == 1)
    {
        //Transfer Transmitter related data to GPU 
        //======================================================================
        cudaMemcpy(lookup_times_tx_gpu, lookup_times_tx, Nlook*sizeof(double), cudaMemcpyHostToDevice); 
        cudaMemcpy(amplitudes_tx_gpu, amplitudes_rx, Nlook*sizeof(double), cudaMemcpyHostToDevice);  
    }
    if (rxUpdate == 1)
    {
        //Transfer Receiver related data to GPU 
        //======================================================================
        cudaMemcpy(lookup_times_rx_gpu, lookup_times_rx, Nlook*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(amplitudes_rx_gpu, amplitudes_rx, Nlook*sizeof(double), cudaMemcpyHostToDevice);         
    }

    // Calculate Block Size for GPU
    int blockSize,minGridSize,gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, c_delay_and_sum_amplitudes_linear_Kernel_DP, 0, numpoints); 
    // Round up according to array size 
    gridSize = (numpoints + blockSize - 1) / blockSize; 

    // Perform Kernel Operations
    //=========================================================================
    c_delay_and_sum_amplitudes_linear_Kernel_DP<<<gridSize, blockSize>>>(scanlines_gpu,tx_gpu,rx_gpu,lookup_times_tx_gpu,lookup_times_rx_gpu,
                 amplitudes_tx_gpu,amplitudes_rx_gpu, invdt, t0, fillvalue,result_gpu,
                numpoints,numsamples,numelements, numscanlines);


    // Return Data from GPU
    //=========================================================================
    cudaMemcpy(result, result_gpu, numpoints*2*sizeof(double), cudaMemcpyDeviceToHost);
    
    Marker++; //Counts Up (for reporting purposes, can see if initialisation in current run (initialisation =2, else >2)
    
    // OPTIONAL USER FORCED CLEANUP
    if (finaliseOpt == 0)
    {
            cudaFree(scanlines_gpu);
            cudaFree(tx_gpu);
            cudaFree(rx_gpu);
            cudaFree(lookup_times_tx_gpu);
            cudaFree(lookup_times_rx_gpu);
            cudaFree(amplitudes_tx_gpu);
            cudaFree(amplitudes_rx_gpu);
            cudaFree(result_gpu);
            cudaDeviceReset();
            Marker=0;
    }

    return Marker;
}

