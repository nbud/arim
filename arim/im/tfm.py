"""
This module defines classes to perform TFM and TFM-like imaging.
"""

import numpy as np
import warnings
from .. import geometry as g
from .amplitudes import UniformAmplitudes, AmplitudesRemoveExtreme
from .base import delay_and_sum
from .fermat_solver import Path, View, Rays, FermatSolver
from .. import settings as s
from ..core import Frame, FocalLaw
from ..enums import CaptureMethod
from .. import has_cuda_gpu
import numba
import math
__all__ = ['BaseTFM', 'ContactTFM', 'SingleViewTFM','BaseMultiTFMviews']

# Order by length then by lexicographic order
# Remark: independant views for one array (i.e. consider that view AB-CD is the
# same as view DC-BA).
IMAGING_MODES = ["L-L", "L-T", "T-T",
                 "LL-L", "LL-T", "LT-L", "LT-T", "TL-L", "TL-T", "TT-L", "TT-T",
                 "LL-LL", "LL-LT", "LL-TL", "LL-TT",
                 "LT-LT", "LT-TL", "LT-TT",
                 "TL-LT", "TL-TT",
                 "TT-TT"]

class BaseTFM:
    """
    Base object for TFM-like algorithms. Define the general workflow.

    To implement a TFM-like algorithm: create a new class that inherits this object and implement if necessary the
    computation of scanlines amplitudes, weights and the delay laws.


    Parameters
    ----------
    frame : Frame
    grid : Grid
    amplitudes : Amplitudes or str
        An amplitude object. Accepted keywords: 'uniform'.
    dtype : numpy.dtype
    geom_probe_to_grid : GeometryHelper

    Attributes
    ----------
    result
    frame
    grid
    dtype
    amplitudes : Amplitudes
    geom_probe_to_grid
    interpolate_position (Default is 0 = Nearest)
    fillvalue : float
        Value to assign to scanlines outside ``[tmin, tmax]``. Default: nan
    delay_and_sum_kwargs : dict

    """
    
    def __init__(self, frame, grid, amplitudes_tx='uniform', amplitudes_rx='uniform',
                 dtype=None, fillvalue=np.nan, geom_probe_to_grid=None,interpolate_position='nearest',maskThreshold=0.5):
        self.frame = frame
        self.grid = grid

        self.rxname=None
        self.txname=None
        self.scanlines_weighted=None
        self.ExpData=None
        self.tfm_number=None
        
        if dtype is None:
            dtype = s.FLOAT
        self.dtype = dtype

        self.fillvalue = fillvalue
        self.interpolate_position=interpolate_position
        self.maskThreshold=maskThreshold


        self.delay_and_sum_kwargs = {}
        self.res = None
        self.mask = None

        if geom_probe_to_grid is None:
            geom_probe_to_grid = g.GeometryHelper(frame.probe.locations, grid.as_points, frame.probe.pcs)
        else:
            assert geom_probe_to_grid.is_valid(frame.probe, grid.as_points)
        self._geom_probe_to_grid = geom_probe_to_grid

        if amplitudes_tx == 'uniform':
            amplitudes_tx = UniformAmplitudes(frame, grid, fillvalue=fillvalue)
        if amplitudes_rx == 'uniform':
            amplitudes_rx = UniformAmplitudes(frame, grid, fillvalue=fillvalue)
        self.amplitudes_tx = amplitudes_tx
        self.amplitudes_rx = amplitudes_rx


    @property
    def MaskedRes(self):
        return self.res*self.mask.astype(self.res.dtype) 
        


    @staticmethod
    def _MaskedResNaN(res,mask): 
        if mask == 0:
            return np.nan
        else:
            return res

    @property
    def MaskedResNaN(self):
        vfunc=np.vectorize(self._MaskedResNaN,otypes=[self.res.dtype])   
        return vfunc(self.res,self.mask) 

    @staticmethod
    def _MaskRemoveNaNres(res): 
        if np.isnan(res):
            return 0.0
        else:
            return res

    @property
    def MaskRemoveNaNres(self):
        vfunc=np.vectorize(self._MaskRemoveNaNres,otypes=[self.res.dtype])   
        return vfunc(self.res) 

    @staticmethod
    def _MaskRemoveNaNmask(res): 
        if np.isnan(res):
            return 0
        else:
            return 1

    @property
    def MaskRemoveNaNmask(self):
        vfunc=np.vectorize(self._MaskRemoveNaNmask,otypes=[np.int])   
        return vfunc(self.res) 


    @staticmethod
    def _MaskedResThreshold(res,mask,threshold): 
        if mask < threshold:
            return np.nan
        else:
            return res
        
    @property
    def MaskedResThreshold(self):
        vfunc=np.vectorize(self._MaskedResThreshold,otypes=[self.res.dtype])     
        threshold=self.maskThreshold
        return vfunc(self.res,self.mask,threshold)
              

    def run(self,use_cpu=1,previous_rxname=None,previous_txname=None,finaliseOnFinal=1,tfm_counter=None,correction=None,ExpDataUpdate=1):
        """
        Compute TFM: get the lookup times, the amplitudes, and delay and sum
        the scanlines.

        Returns
        -------
        result

        """
        
        #GPU Test
        if use_cpu == 0:
            if has_cuda_gpu.gpu_checked == 1:
                pass
            else:
                warnings.warn("GPU Inspection not undertaken. Attempting now...")
                has_cuda_gpu.test()
            if has_cuda_gpu.has_gpu == 1:
                pass
            else:
                warnings.warn("Invalid Option: No appropriate GPU Detected. Using CPU OpenMP")
                use_cpu=1         
        
        self.hook_start_run()

        lookup_times_tx = self.get_lookup_times_tx()
        lookup_times_rx = self.get_lookup_times_rx()
        amplitudes_tx = self.get_amplitudes_tx()
        amplitudes_rx = self.get_amplitudes_rx()

        focal_law = FocalLaw(lookup_times_tx, lookup_times_rx, amplitudes_tx, amplitudes_rx)

        focal_law = self.hook_focal_law(focal_law)
        self.focal_law = focal_law
        
        numscanlines, numsamples = self.frame.scanlines.shape 
        
        if self.ExpData is None:
            print("Scanlines weighted does not exist. Creating")
            scanlines_weighted=calc_hmc_fmc_weighted_scan_data(self.frame) 
            self.scanlines_weighted=scanlines_weighted
            
            if self.frame.scanlines.dtype.kind == 'c': 
                ExpData=self.scanlines_weighted.view(s.FLOAT)
                ExpData=np.reshape(ExpData,(numscanlines,numsamples,2))
            elif self.frame.scanlines.dtype.kind == 'f':
                warnings.warn("scanlines is FLOAT, not COMPLEX. Will assign zeros to imaginary component and continue. Recode for faster FLOAT implementation.")
                ExpData=np.empty((self.frame.scanlines.shape[0],self.frame.scanlines.shape[1],2),dtype=s.FLOAT)
                ExpData[:,:,0],ExpData[:,:,1]=self.scanlines_weighted,np.zeros_like(self.frame.scanlines)    
                #ExpData=self.scanlines_weighted.view(s.FLOAT)
                #ExpData=np.reshape(ExpData,(numscanlines,numsamples,1))
            else:
                raise NotImplementedError
            
            self.ExpData=ExpData
            
            assert self.ExpData.dtype == s.FLOAT
            assert self.ExpData.flags.c_contiguous   
        
        ExpData=self.ExpData        

        assert self.focal_law.lookup_times_tx.dtype == s.FLOAT
        assert self.frame.tx.dtype == s.UINT
        assert self.frame.time.step.dtype == s.FLOAT or isinstance(self.frame.time.step,float)
        assert self.frame.time.start.dtype == s.FLOAT or isinstance(self.frame.time.start,float)
        
        numelements = self.frame.probe.numelements
        numpoints = self.grid.numpoints
        
        assert self.focal_law.lookup_times_tx.shape == (numpoints, numelements)
        assert self.focal_law.lookup_times_rx.shape == (numpoints, numelements)
        assert self.focal_law.amplitudes_tx.shape == (numpoints, numelements)
        assert self.focal_law.amplitudes_rx.shape == (numpoints, numelements)
        assert self.frame.tx.shape == (numscanlines,)
        assert self.frame.rx.shape == (numscanlines,)

        assert self.focal_law.lookup_times_tx.flags.c_contiguous
        assert self.focal_law.amplitudes_rx.flags.c_contiguous
        
        
        dt=self.frame.time.step
        t0=self.frame.time.start
        invdt=1.0/dt.astype(s.FLOAT)
        resData=np.empty((numpoints,2), dtype=s.FLOAT)
        mask=np.empty((numpoints),dtype=np.int)


        if use_cpu is None:
            delay_and_sum_kwargs = dict(fillvalue=self.fillvalue)
            res = delay_and_sum(self.scanlines_weighted,self.frame, focal_law,interpolate_position=self.interpolate_position,**delay_and_sum_kwargs)
            res = self.hook_result(res)
            self.res = res
            if np.isnan(self.fillvalue):
                self.mask= self.MaskRemoveNaNmask
                self.res = self.MaskRemoveNaNres
            else:
                self.mask= np.ones(self.res.shape,dtype=np.int)    
            

            
            
            return

        if use_cpu==1:
            import delay_and_sumC_CPU  
            # Dynamic Chunksize for OpenMP implementation
            CHUNKSIZE=s.BLOCK_SIZE_DELAY_AND_SUM
        elif use_cpu==0:
            import delay_and_sumC_CUDA
            # NTHREADS -> Currently not used (CUDA contains algorithm to compute block/Grid dispersion)
            NTHREADS=s.NTHREADS_PER_BLOCK
            #Find out what if any changes have occurred to focal_law information
            #To keep GPU passing information to minimum
            #Updating will occur if set to 1, no update if 0
            txUpdate=1 #Transmitter Path Update
            rxUpdate=1 #Receiver Path Update
            finaliseOpt=-1 #Free memory on GPU if (0,1). 0 => Free at end of TFM calculation. 1 = Free only (no TFM calc), use as post-TFMs calc call
    
            if finaliseOnFinal == 1:
                if tfm_counter is None:
                    finaliseOpt=0
                elif tfm_counter - 1 == self.tfm_number:
                    print("Freeing GPU Data after processing TFM data")
                    finaliseOpt=0
            if self.txname is not None:        
                if self.txname == previous_txname:
                    txUpdate=0
            if self.rxname is not None:         
                if self.rxname == previous_rxname:
                    rxUpdate=0 
                    
            if self.tfm_number is not None:
                print("GPU Calculation of TFM ",self.tfm_number,"Updating ",self.txname," TX(",txUpdate,") ",self.rxname," RX(",rxUpdate,"), ExpData(",ExpDataUpdate,")")     
            else:
                print("GPU Calculation of TFM") 
                    
        else:
            raise NotImplementedError


        if correction is not None:
            pass
        else:
            if self.interpolate_position == 'nearest':  
                if s.FLOAT == np.float64:
                    if use_cpu == 1:
                        delay_and_sumC_CPU.delayAndSum_Algorithm_nearest_DP_CPU(ExpData, self.frame.tx, self.frame.rx, self.focal_law.lookup_times_tx, self.focal_law.lookup_times_rx, \
                            self.focal_law.amplitudes_tx, self.focal_law.amplitudes_rx,invdt,t0, self.fillvalue,resData,mask,CHUNKSIZE)
                    else:
                        delay_and_sumC_CUDA.delayAndSum_Algorithm_nearest_DP_CUDA(ExpData, self.frame.tx, self.frame.rx, self.focal_law.lookup_times_tx, self.focal_law.lookup_times_rx, \
                            self.focal_law.amplitudes_tx, self.focal_law.amplitudes_rx,invdt,t0, self.fillvalue,resData,NTHREADS,txUpdate,rxUpdate,ExpDataUpdate,finaliseOpt)
                elif s.FLOAT == np.float32:
                    if use_cpu == 1:
                        delay_and_sumC_CPU.delayAndSum_Algorithm_nearest_SP_CPU(ExpData, self.frame.tx, self.frame.rx, self.focal_law.lookup_times_tx, self.focal_law.lookup_times_rx, \
                            self.focal_law.amplitudes_tx, self.focal_law.amplitudes_rx,invdt,t0, self.fillvalue,resData,mask,CHUNKSIZE)
                    else:
                        delay_and_sumC_CUDA.delayAndSum_Algorithm_nearest_SP_CUDA(ExpData, self.frame.tx, self.frame.rx, self.focal_law.lookup_times_tx, self.focal_law.lookup_times_rx, \
                            self.focal_law.amplitudes_tx, self.focal_law.amplitudes_rx,invdt,t0, self.fillvalue,resData,NTHREADS,txUpdate,rxUpdate,ExpDataUpdate,finaliseOpt)
                else:
                    raise NotImplementedError     
            elif self.interpolate_position == 'linear':  
                if s.FLOAT == np.float64:
                    if use_cpu == 1:
                        delay_and_sumC_CPU.delayAndSum_Algorithm_linear_DP_CPU(ExpData, self.frame.tx, self.frame.rx, self.focal_law.lookup_times_tx, self.focal_law.lookup_times_rx, \
                            self.focal_law.amplitudes_tx, self.focal_law.amplitudes_rx,invdt,t0, self.fillvalue,resData,mask,CHUNKSIZE)
                    else:
                        delay_and_sumC_CUDA.delayAndSum_Algorithm_linear_DP_CUDA(ExpData, self.frame.tx, self.frame.rx, self.focal_law.lookup_times_tx, self.focal_law.lookup_times_rx, \
                            self.focal_law.amplitudes_tx, self.focal_law.amplitudes_rx,invdt,t0, self.fillvalue,resData,NTHREADS,txUpdate,rxUpdate,ExpDataUpdate,finaliseOpt)
                elif s.FLOAT == np.float32:
                    if use_cpu == 1:
                        delay_and_sumC_CPU.delayAndSum_Algorithm_linear_SP_CPU(ExpData, self.frame.tx, self.frame.rx, self.focal_law.lookup_times_tx, self.focal_law.lookup_times_rx, \
                            self.focal_law.amplitudes_tx, self.focal_law.amplitudes_rx,invdt,t0, self.fillvalue,resData,mask,CHUNKSIZE)
                    else:
                        delay_and_sumC_CUDA.delayAndSum_Algorithm_linear_SP_CUDA(ExpData, self.frame.tx, self.frame.rx, self.focal_law.lookup_times_tx, self.focal_law.lookup_times_rx, \
                            self.focal_law.amplitudes_tx, self.focal_law.amplitudes_rx,invdt,t0, self.fillvalue,resData,NTHREADS,txUpdate,rxUpdate,ExpDataUpdate,finaliseOpt)
                else:
                    raise NotImplementedError 
            else:
                raise NotImplementedError
        
        if self.frame.scanlines.dtype.kind == 'c': 
            self.res = resData.view(s.COMPLEX)
        elif self.frame.scanlines.dtype.kind == 'f':
            #self.res = np.empty((numpoints), dtype=s.FLOAT)
            self.res=resData[:,0]  
            print(self.grid.numpoints)
            print(resData.shape)
            print(np.sum(self.MaskedResNaN))
            print(np.sum(resData))
            print(np.sum(self.res))
        else:
            raise NotImplementedError  
            
        #mask=mask.__nonzero__()    
        self.mask=mask 
        self.mask=self.hook_result(self.mask)
        #self.res = resData.view(s.COMPLEX)
        #self.res = np.empty((numpoints), dtype=s.COMPLEX)
        #self.res.real,self.res.imag=resData[:,0],resData[:,1]            
        self.res=self.hook_result(self.res)
        
#        
            
        #return res

    def get_amplitudes_tx(self):
        return self.amplitudes_tx()

    def get_amplitudes_rx(self):
        return self.amplitudes_rx()

    def get_lookup_times_tx(self):
        raise NotImplementedError('must be implemented by child class')

    def get_lookup_times_rx(self):
        raise NotImplementedError('must be implemented by child class')
        
    def get_scanline_weights(self):
        """
        Standard scanline weights. Handle FMC and HMC.

        For FMC: weights 1.0 for all scanlines.
        For HMC: weights 2.0 for scanlines where TX and RX elements are different, 1.0 otherwise.

        """
        capture_method = self.frame.metadata.get('capture_method', None)
        if capture_method is None:
            raise NotImplementedError
        elif capture_method is CaptureMethod.fmc:
            weights = np.ones(self.frame.numscanlines, dtype=self.dtype)
            return weights
        elif capture_method is CaptureMethod.hmc:
            weights = np.full(self.frame.numscanlines, 2.0, dtype=self.dtype)
            same_tx_rx = self.frame.tx == self.frame.rx
            weights[same_tx_rx] = 1.0
            return weights
        else:
            raise NotImplementedError

    def hook_start_run(self):
        """Implement this method in child class if necessary."""
        pass

    def hook_focal_law(self, focal_law):
        """Hooked called after creating the focal law.
        Implement this method in child class if necessary."""
        return focal_law

    def hook_result(self, res):
        """Implement this method in child class if necessary.

        Default behaviour: reshape results (initially 1D array) to 3D array with
        same shape as the grid.
        """
        return res.reshape((self.grid.numx, self.grid.numy, self.grid.numz))

    @property
    def geom_probe_to_grid(self):
        return self._geom_probe_to_grid


class ContactTFM(BaseTFM):
    """
    Contact TFM. The probe is assumed to lay on the surface of the examination
    object.
    """
    def __init__(self, speed, **kwargs):
        # This attribute is attached to the instance AND the class (double underscore):
        self.__lookup_times = None

        self.speed = speed
        super().__init__(**kwargs)

    def get_lookup_times_tx(self):
        """
        Lookup times obtained by dividing Euclidean distances between elements and
        image points by the speed (``self.speed``).
        """
        if self.__lookup_times is None:
            distance = self._geom_probe_to_grid.distance_pairwise()
            # distance = distance_pairwise(
            #     self.grid.as_points, self.frame.probe.locations, **self.distance_pairwise_kwargs)
            distance /= self.speed
            self.__lookup_times = distance
        return self.__lookup_times

    get_lookup_times_rx = get_lookup_times_tx


class SingleViewTFM(BaseTFM):
    """
    Single View TFM. The probe is located in vicinity of examination object, 
    although not assumed in contact, thus multiple potential ray paths exist
    and hence multiple potential views. This is a single view from the specified
    list of views.
    """    
    def __init__(self, frame, grid, view, rays_tx, rays_rx, **kwargs):
        # assert grid is view.tx_path[-1]
        # assert grid is view.rx_path[-1]
        if grid.numpoints != len(view.tx_path[-1]):
            raise ValueError("Inconsistent grid")
        if grid.numpoints != len(view.rx_path[-1]):
            raise ValueError("Inconsistent grid")

        assert view.tx_path == rays_tx.path
        assert view.rx_path == rays_rx.path
        assert isinstance(rays_tx, Rays)
        assert isinstance(rays_rx, Rays)
        assert rays_rx.indices.flags.fortran
        assert rays_tx.indices.flags.fortran
        assert rays_tx.times.flags.fortran
        assert rays_rx.times.flags.fortran
        assert rays_tx.path[0] is frame.probe.locations
        assert rays_rx.path[0] is frame.probe.locations
        assert rays_tx.path[-1] is grid.as_points
        assert rays_rx.path[-1] is grid.as_points
        self.rays_tx = rays_tx
        self.rays_rx = rays_rx
        self.view = view
        
        # used in get_amplitudes
        self.fillvalue_extreme_points = np.nan

        amplitudes_tx = kwargs.get('amplitudes_tx')
        if amplitudes_tx is None:
            amplitudes_tx = AmplitudesRemoveExtreme(frame, grid, rays_tx)
        kwargs['amplitudes_tx'] = amplitudes_tx

        amplitudes_rx = kwargs.get('amplitudes_rx')
        if amplitudes_rx is None:
            amplitudes_rx = AmplitudesRemoveExtreme(frame, grid, rays_rx)
        kwargs['amplitudes_rx'] = amplitudes_rx

        super().__init__(frame, grid, **kwargs)        
        self.txname, self.rxname = view.name.split('-')


    def get_lookup_times_tx(self):
        """Lookup times in transmission, obtained with Fermat solver."""
        return self.rays_tx.times.T

    def get_lookup_times_rx(self):
        """Lookup times in reception, obtained with Fermat solver."""
        return self.rays_rx.times.T


    def __repr__(self):
        return "<{}: {} at {}>".format(
            self.__class__.__name__,
            str(self.view),
            hex(id(self)))

class BaseMultiTFMviews:
    """
    Multi TFM Views. This is a collection of individual TFM views,
    where the probe is located in vicinity of examination object, 
    although not assumed in contact. This is a container for each
    single view from the specified list of views.
    """       
    def __init__(self,frame,probe,frontwall,backwall,grid,v_longi,v_shear,v_couplant,interpolate_position='nearest',AllowExpDataUpdate=0,finaliseOnFinal=1, **kwargs):
        
        #TFM Runner variables
        self.previous_txname=None
        self.previous_rxname=None
        self.AllowExpDataUpdate=AllowExpDataUpdate
        self.finaliseOnFinal=finaliseOnFinal
        
        #Setup for TFM
        scanlines_weighted=calc_hmc_fmc_weighted_scan_data(frame) 
        self.scanlines_weighted=scanlines_weighted
        
        numscanlines, numsamples = self.scanlines_weighted.shape 
        if frame.scanlines.dtype.kind == 'c': 
            ExpData=self.scanlines_weighted.view(s.FLOAT)
            ExpData=np.reshape(ExpData,(numscanlines,numsamples,2))
        elif self.scanlines.dtype.kind == 'f':
            ExpData=self.scanlines_weighted.view(s.FLOAT)
            ExpData=np.reshape(ExpData,(numscanlines,numsamples,1))
        else:
            raise NotImplementedError
            
        self.ExpData=ExpData    
        
        assert ExpData.dtype == s.FLOAT
        assert ExpData.flags.c_contiguous   
                
        views = self.make_views(probe, frontwall, backwall, grid.as_points, v_couplant, v_longi, v_shear)
        self.views=views
        print('Views to show: {}'.format(str(views)))
        
        #%% Setup Fermat solver and compute rays
        fermat_solver = FermatSolver.from_views(views)
        rays = fermat_solver.solve()
        
        self.rays=rays
        
        tfms = []
        tfm_counter=0
        for i, view in enumerate(views):
            rays_tx = rays[view.tx_path]
            rays_rx = rays[view.rx_path]
            amps_tx = UniformAmplitudes(frame, grid)
            amps_rx = UniformAmplitudes(frame, grid)
            #tic = time.clock() 
            
            tfm = SingleViewTFM(frame, grid, view, rays_tx, rays_rx,
                                  amplitudes_tx=amps_tx, amplitudes_rx=amps_rx,interpolate_position=interpolate_position)
            
            tfm.tfm_number=i
            tfm_counter+=1
            #print("Initialised TFM View ",i,tfm.txname,tfm.rxname)
            tfm.ExpData=self.ExpData
            tfm.scanlines_weighted=self.scanlines_weighted
       
            tfms.append(tfm)

        self.tfms=tfms
        self.tfm_counter=tfm_counter


    def BasicRunAll(self,use_cpu=1,correction=None):
        
        #GPU Test
        if use_cpu == 0:
            if has_cuda_gpu.gpu_checked == 1:
                pass
            else:
                warnings.warn("GPU Inspection not undertaken. Attempting now...")
                has_cuda_gpu.test()
            if has_cuda_gpu.has_gpu == 1:
                pass
            else:
                warnings.warn("Invalid Option: No appropriate GPU Detected. Using CPU OpenMP")
                use_cpu=1 
        
        tfm_number=0        
        EDUpdate=1
        for tfm in self.tfms:
            tfm_number+=1
            #Conduct Checks to see what to pass for GPU (if used)                   
            tfm.run(use_cpu=use_cpu,correction=correction,previous_rxname=self.previous_rxname,previous_txname=self.previous_txname,tfm_counter=self.tfm_counter,ExpDataUpdate=EDUpdate) 
            EDUpdate=0    #ExpData only needs to be transferred once per batch of TFMs        
            self.previous_rxname=tfm.rxname
            self.previous_txname=tfm.txname
            

        

    @classmethod
    def make_views(cls, probe, frontwall, backwall, grid, v_couplant, v_longi, v_shear):
        """
        Create direct-direct, skip-direct and skip-skip views.

        Parameters
        ----------
        probe : Points
        frontwall : Points
        backwall : Points
        grid : Points
        v_couplant : float
        v_longi : float
        v_shear : float

        Returns
        -------
        views : List[View]
        """
        views = []
        parse = lambda name: cls._parse_name_view(name, backwall, v_longi, v_shear)
        for name in IMAGING_MODES:
            tx_name, rx_name = name.split('-')
            tx_path = Path((probe, v_couplant, frontwall) + parse(tx_name) + (grid,))
            rx_path = Path((probe, v_couplant, frontwall) + parse(rx_name[::-1]) + (grid,))

            views.append(View(tx_path, rx_path, name))
        return views

    @staticmethod
    def _parse_name_view(name, backwall, v_longi, v_shear):
        name = name.upper()
        if name == 'L':
            return (v_longi,)
        elif name == 'T':
            return (v_shear,)
        elif name == 'LL':
            return (v_longi, backwall, v_longi)
        elif name == 'LT':
            return (v_longi, backwall, v_shear)
        elif name == 'TL':
            return (v_shear, backwall, v_longi)
        elif name == 'TT':
            return (v_shear, backwall, v_shear)
        else:
            raise ValueError("Cannot parse view '{}'".format(name))




        
            
@numba.jit
def _apply_scanline_weights(frame,numscanlines,numsamples):
    """
    Applies the HMC Weighting to scanlines
    """
    scanlines_weighted = 2.0*frame.scanlines
    for scan in range(numscanlines):
        if frame.tx[scan]==frame.rx[scan]:
            for sample in range(numsamples):
                scanlines_weighted[scan,sample]=frame.scanlines[scan,sample]
        
    return scanlines_weighted
    
def calc_hmc_fmc_weighted_scan_data(frame):
    """
    Computes the adjusted scanlines, weighted for FMC (weight = 1 throughout) or HMC (weight =1 if rx=tx else 2)
    """
    numscanlines,numsamples=frame.scanlines.shape
    numelements = frame.probe.numelements
    print("Number of Probe Array elements",numelements)
    print("Number of Scanlines in frame",numscanlines)
    #Determine if hmc or fmc from size of data
    if 2*(numscanlines-numelements)+numelements == numelements*numelements:
        #HMC
        print("Detected HMC, weighting appropriately")
        scanlines_weighted=_apply_scanline_weights(frame,numscanlines,numsamples)
    elif numscanlines == numelements*numelements:
        #FMC (no need to do anything)
        scanlines_weighted=frame.scanlines
    else:
        raise NotImplementedError
        
    return scanlines_weighted          