#!/usr/bin/env python3

import BNS_Optimisation_Module as bnso #import module
import bilby
import h5py
import numpy as np
import gwinc
import astropy.units as u
import astropy.constants as c
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from csv import writer
import pandas as pd
import merger_rates_for_mallika
import random
import BNS_Optimisation_Module_Randomised as newbn

df, index = bnso.sim_index(sim_data_name = 'THCsim_name_one_removesmallbig.csv')
duration, sampling_frequency, injection_parameters = bnso.initial_parameters()
metadata = 'metadata.txt'

IFO, name = newbn.IFOmaker('Config_1',duration,sampling_frequency, 0., " "
                               , 'OzHF_NoiseBudget_RadCooling_TotalNoise.txt', model='psd'); 
detratelist = []
for j in range(0,50):
    numbers = list(index)*100
    random.shuffle(numbers)
    time, random_param = newbn.random_param(df,400*u.Mpc,295.7, scalewavno=100);
    SNRlist = []
    for i in range(0,len(numbers)):
        AusIFO = IFO;
        injection_parameters = dict(distance=random_param['distance'][i], phase=random_param['phase'][i], ra=random_param['ra'][i], 
                                dec=random_param['dec'][i], psi=random_param['psi'][i], t0=0., geocent_time=0.)
        SNR = newbn.calc_SNR(numbers[i], duration,sampling_frequency, injection_parameters,AusIFO,df)
        SNRlist.append(SNR)
    detratescaled = sum(i > 5 for i in SNRlist)
    detrate = detratescaled/time
    detratelist.append(detrate)
    datalist = pd.DataFrame(detratelist)
    datalist.to_csv('testingtesting.csv')


print(detratelist)