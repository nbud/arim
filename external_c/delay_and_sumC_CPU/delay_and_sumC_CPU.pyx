#!python
#cython: language_level=3, boundscheck=False, wraparound=False,initializedcheck=False

  
cdef extern int c_delayAndSum_Algorithm_nearest_DP_CPU(const double* scanlines, const unsigned int* tx, const unsigned int* rx, const double* lookup_times_tx, const double* lookup_times_rx,
                 const double* amplitudes_tx, const double* amplitudes_rx, double invdt, double t0, double fillvalue,double* result,int* mask,
                int numpoints,int numsamples,int numelements, int numscanlines,int CHUNKSIZE);
    
cdef extern int c_delayAndSum_Algorithm_nearest_SP_CPU(const float* scanlines, const unsigned int* tx, const unsigned int* rx, const float* lookup_times_tx, const float* lookup_times_rx,
                 const float* amplitudes_tx, const float* amplitudes_rx, float invdt, float t0, float fillvalue,float* result,int* mask,
                int numpoints,int numsamples,int numelements, int numscanlines,int CHUNKSIZE);       

cdef extern int c_delayAndSum_Algorithm_linear_SP_CPU(const float* scanlines, const unsigned int* tx, const unsigned int* rx, const float* lookup_times_tx, const float* lookup_times_rx,
                 const float* amplitudes_tx, const float* amplitudes_rx, float invdt, float t0, float fillvalue,float* result,int* mask,
                int numpoints,int numsamples,int numelements, int numscanlines,int CHUNKSIZE);  

cdef extern int c_delayAndSum_Algorithm_linear_DP_CPU(const float* scanlines, const unsigned int* tx, const unsigned int* rx, const float* lookup_times_tx, const float* lookup_times_rx,
                 const float* amplitudes_tx, const float* amplitudes_rx, float invdt, float t0, float fillvalue,float* result,int* mask,
                int numpoints,int numsamples,int numelements, int numscanlines,int CHUNKSIZE);  
                
#===========================================================================================================================================             
def delayAndSum_Algorithm_nearest_DP_CPU(double[:,:,::1] scanlines, unsigned int[:] tx, unsigned int[:] rx, double[:,::1] lookup_times_tx, double[:,::1] lookup_times_rx, \
                double[:,::1] amplitudes_tx, double[:,::1] amplitudes_rx, double invdt, double t0, double fillvalue,double[:,::1] result,int[:] mask, int CHUNKSIZE): # except? -2:
#===========================================================================================================================================      
    
    cdef int numscanlines = <int>scanlines.shape[0] 
    cdef int numsamples = <int>scanlines.shape[1]
    cdef int numpoints= <int>lookup_times_tx.shape[0]
    cdef int numelements = <int>lookup_times_tx.shape[1]

    cdef int ReportValue    
    ReportValue=c_delayAndSum_Algorithm_nearest_DP_CPU(&scanlines[0,0,0], &tx[0], &rx[0], &lookup_times_tx[0,0], &lookup_times_rx[0,0],&amplitudes_tx[0,0],&amplitudes_rx[0,0], invdt,t0, fillvalue,&result[0,0],&mask[0],numpoints,numsamples,numelements,numscanlines,CHUNKSIZE)
      

#===========================================================================================================================================    
def delayAndSum_Algorithm_nearest_SP_CPU(float[:,:,::1] scanlines, unsigned int[:] tx, unsigned int[:] rx, float[:,::1] lookup_times_tx, float[:,::1] lookup_times_rx, \
                float[:,::1] amplitudes_tx, float[:,::1] amplitudes_rx, float invdt, float t0, float fillvalue,float[:,::1] result,int[:] mask, int CHUNKSIZE): # except? -2:
#===========================================================================================================================================      
    
    cdef int numscanlines = <int>scanlines.shape[0] 
    cdef int numsamples = <int>scanlines.shape[1]
    cdef int numpoints= <int>lookup_times_tx.shape[0]
    cdef int numelements = <int>lookup_times_tx.shape[1]

    cdef int ReportValue    
    ReportValue=c_delayAndSum_Algorithm_nearest_SP_CPU(&scanlines[0,0,0], &tx[0], &rx[0], &lookup_times_tx[0,0], &lookup_times_rx[0,0],&amplitudes_tx[0,0],&amplitudes_rx[0,0], invdt,t0, fillvalue,&result[0,0],&mask[0],numpoints,numsamples,numelements,numscanlines,CHUNKSIZE)
    
#===========================================================================================================================================    
def delayAndSum_Algorithm_linear_SP_CPU(float[:,:,::1] scanlines, unsigned int[:] tx, unsigned int[:] rx, float[:,::1] lookup_times_tx, float[:,::1] lookup_times_rx, \
                float[:,::1] amplitudes_tx, float[:,::1] amplitudes_rx, float invdt, float t0, float fillvalue,float[:,::1] result,int[:] mask, int CHUNKSIZE): # except? -2:
#===========================================================================================================================================      
    
    cdef int numscanlines = <int>scanlines.shape[0] 
    cdef int numsamples = <int>scanlines.shape[1]
    cdef int numpoints= <int>lookup_times_tx.shape[0]
    cdef int numelements = <int>lookup_times_tx.shape[1]

    cdef int ReportValue    
    ReportValue=c_delayAndSum_Algorithm_linear_SP_CPU(&scanlines[0,0,0], &tx[0], &rx[0], &lookup_times_tx[0,0], &lookup_times_rx[0,0],&amplitudes_tx[0,0],&amplitudes_rx[0,0], invdt,t0, fillvalue,&result[0,0],&mask[0],numpoints,numsamples,numelements,numscanlines,CHUNKSIZE)

#===========================================================================================================================================    
def delayAndSum_Algorithm_linear_DP_CPU(float[:,:,::1] scanlines, unsigned int[:] tx, unsigned int[:] rx, float[:,::1] lookup_times_tx, float[:,::1] lookup_times_rx, \
                float[:,::1] amplitudes_tx, float[:,::1] amplitudes_rx, float invdt, float t0, float fillvalue,float[:,::1] result,int[:] mask, int CHUNKSIZE): # except? -2:
#===========================================================================================================================================      
    
    cdef int numscanlines = <int>scanlines.shape[0] 
    cdef int numsamples = <int>scanlines.shape[1]
    cdef int numpoints= <int>lookup_times_tx.shape[0]
    cdef int numelements = <int>lookup_times_tx.shape[1]

    cdef int ReportValue    
    ReportValue=c_delayAndSum_Algorithm_linear_DP_CPU(&scanlines[0,0,0], &tx[0], &rx[0], &lookup_times_tx[0,0], &lookup_times_rx[0,0],&amplitudes_tx[0,0],&amplitudes_rx[0,0], invdt,t0, fillvalue,&result[0,0],&mask[0],numpoints,numsamples,numelements,numscanlines,CHUNKSIZE)
       