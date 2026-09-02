#!/usr/bin/env python
# -*- coding: utf-8 -*-
import bilby
from bilby.gw.detector import PowerSpectralDensity
from bilby.gw.detector import utils
import gwpy
import gwpy.timeseries as ts
from gwpy.timeseries import TimeSeries
import warnings
import sys
"""
The code for generating noise only frames
"""
jobNumber = int(sys.argv[1])
duration = 2048

st = (jobNumber-1)*duration
et = st + duration
minimum_frequency = 3. 
reference_frequency = 25.
sampling_frequency = 2048.0
psd_CE = PowerSpectralDensity(psd_file="../Data/CE_40_PSD.txt")
interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1'])
for ifo in interferometers:
    ifo.minimum_frequency  = minimum_frequency
    ifo.sampling_frequency = sampling_frequency
    ifo.duration = duration
    ifo.power_spectral_density = psd_CE

# Realization of the detector's noise
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=st)


base_channel_name = 'TEST_INJ'
gps_start_time = ifo.time_array[0]
for ix, ifo in enumerate(interferometers):
    new_channel_names = ifo.name+':' + base_channel_name
    inj_channel = gwpy.detector.Channel(ifo.name+':'+new_channel_names)
    injected_ts = ts.TimeSeries(ifo.time_domain_strain, times=ifo.time_array,
                                name=new_channel_names, channel=inj_channel, dtype=float)
    file_name = ifo.name[0]+'-STRAIN-'+str(st)+'-'+str(duration)+'.gwf'
    injected_ts.write(f'../Frames/Pure_Noise/{file_name}')
