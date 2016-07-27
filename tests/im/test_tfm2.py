import numpy as np
import pytest

import arim
import arim.im as im
import arim.im.amplitudes
import arim.im.tfm
import arim.io
import arim.settings as s
import arim.geometry as g
from arim.im import fermat_solver as t
import math
from tests.helpers import get_data_filename


@pytest.fixture()
def probe():
    return arim.probes['ima_50_MHz_64_1d']


@pytest.fixture()
def grid():
    return g.Grid(-10.e-3, 60.e-3, 0., 0., 0., 40.e-3, 1.e-3)


@pytest.fixture()
def frame():
    expdata_filename = get_data_filename("brain/exp_data.mat7.mat")
    frame = arim.io.load_expdata(expdata_filename)
    frame.probe = probe()
    return frame


@pytest.fixture()
def tfm(frame, grid):
    speed = 1.0
    return im.ContactTFM(speed, frame=frame, grid=grid)


class TestTFM:
    def test_contact_tfm(self, grid, frame):
        speed = 6300
        assert grid.numpoints == 2911
        tfm = im.ContactTFM(speed, frame=frame, grid=grid)
        tfm.run(use_cpu=None)

        res=tfm.MaskedResNaN
        #res=tfm.res
        assert res.dtype==s.FLOAT
        x2=71
        y2=1
        z2=41
        x1,y1,z1=res.shape
        assert x1 == x2
        assert y1 == y2
        assert z1 == z2  
        print()        
        res_abs=0.0
        res_tot=0.0
        res_nan=0
        for xx in range(x1):
            for yy in range(y1):
                for zz in range(z1):
                    if math.isfinite(res[xx,yy,zz]):
                        res_abs=res_abs+abs(res[xx,yy,zz])
                        res_tot=res_tot+res[xx,yy,zz]                        
                    else:
                        res_nan=res_nan+1
        #Note values changed below to reflect impact within delay_and_sum of changing from round(lookup_time) to (int)(lookup_time+0.5) different behaviour with round(0.5)->0 (round to zero), int(0.5+0.5)->1 (round up)               
        #assert res_nan == 2006
        assert res_nan == 2004
        if s.FLOAT == np.float64:
            assert math.isclose(res_abs,4978.46875,abs_tol=1e-10,rel_tol=1e-10)
            assert math.isclose(res_tot,3749.125,abs_tol=1e-10,rel_tol=1e-10)
        else:
            #Numbers should match exactly if every variable was single precision (numbers from C code), but python is double preci
            assert math.isclose(res_abs,4978.3671875,rel_tol=1e-04)        
            assert math.isclose(res_tot,3748.9921875,rel_tol=1e-04)
        
        
           


class TestAmplitude:
    def test_uniform(self, frame, grid):
        amplitudes = arim.im.amplitudes.UniformAmplitudes(frame, grid)
        res = amplitudes()
        assert np.allclose(res, 1.)

    def test_directivity_finite_width_2d(self, frame, grid):
        amplitudes = arim.im.amplitudes.DirectivityFiniteWidth2D(frame, grid, speed=1.0)
        res = amplitudes()

    def test_multi_amplitudes(self, frame, grid):
        amp1 = arim.im.amplitudes.UniformAmplitudes(frame, grid)
        amp2 = arim.im.amplitudes.UniformAmplitudes(frame, grid)
        multi_amp = arim.im.amplitudes.MultiAmplitudes([amp1, amp2])
        assert np.allclose(multi_amp(), 1.)


def test_make_views():
    probe = g.Points(np.random.rand(10), np.random.rand(10), np.random.rand(10), 'Probe')
    frontwall = g.Points(np.random.rand(10), np.random.rand(10), np.random.rand(10), 'Frontwall')
    backwall = g.Points(np.random.rand(10), np.random.rand(10), np.random.rand(10), 'Backwall')
    grid = g.Points(np.random.rand(10), np.random.rand(10), np.random.rand(10), 'Grid')

    v_couplant = 1.0
    v_longi = 2.0
    v_shear = 3.0

    views = arim.im.tfm.BaseMultiTFMviews.make_views(probe, frontwall, backwall, grid, v_couplant, v_longi, v_shear)

    assert len(views) == 21
    assert len(set([v.name for v in views])) == 21

    view = [v for v in views if v.name == 'LT-TL'][0]
    assert view.tx_path == view.rx_path

    view = [v for v in views if v.name == 'LT-LT'][0]
    assert view.tx_path == (probe, v_couplant, frontwall, v_longi, backwall, v_shear, grid)
    assert view.rx_path == (probe, v_couplant, frontwall, v_shear, backwall, v_longi, grid)