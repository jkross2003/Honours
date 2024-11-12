import bilby # Requires bilby version 1.1.3! (newer versions will require NS masses in injection parameters)
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

bilby.core.utils.logger.setLevel("ERROR") # Suppress output
# input_path = get.cwd()
input_path='./CoRe_DB_clone/' #path to the numerical waveforms

#list of constants that might want to be changed in the future
fmin = 300
fmax = 10e3 
NEMO_latitude = -31.34
NEMO_longitude = 115.91
NEMO_elevation = 0
NEMO_xarm_azimuth = 2
NEMO_yarm_azimuth = 125
min_cutting_freq = 1000 #minimum peak frequency (mainly to get rid of artefacts from cutting around 0 Hz)
NEMO_length = 4 #arm length

def sim_index(sim_data_name):
    """ 

    Reads in the simulation key which links each simulation name with an index in a csv. 

    Parameters
    ==========
    sim_data_name: str, optional
        File name for the simulation key csv.
        If not provided will fall back to 'THCsim_name.csv'.

    Returns
    =======
    DataFrame: Conversion of given csv file.
    list: List of index numbers to loop over.

    """
    df = pd.read_csv(#input_path + 
        sim_data_name) #read in list of simulations
    index = range(0,len(df))
    return df, index

def make_h5(h5_name, no_of_data_points=7):
    """ 

    Creates .h5 file split by the different EOS provided in the NR simulations.
    Each Dataset falls back to 6 columns, but this number can be changed - this means the function defined for data collection must be changed too.
    The columns given in this code are: simulation number (from key), total mass, mass of A, mass of B, mass ratio, peak frequency, duration.

    Parameters
    ==========
    h5_name: str
        File name for the output file being created.
        
    no_of_data_points: int, optional
       Number of columns in .h5 file outputted. 
        If not provided will fall back to 7. If this is changed, the toappend list in the collect() function will need to be changed, adding/removing the relevant columns.

    """
    names = ['BHBlp', 'DD2', 'LS220', 'SFHo', '2B', '2H', 'ALF2', 'ENG', 'H4', 'MPA1', 'MS1', 'MS1b', 'SLy', 'G2', 'G2k123', 'BLh','BLQ','SLy4']  # UPDATE this with new releases!
    b = h5py.File(h5_name, "w")
    for n in names:
        b.create_dataset(n , (0,no_of_data_points), maxshape = (None, None))
    b.close()
    
def initial_parameters(distance=40, phase=0, ra=0, dec=0, psi=0, t0=0., geocent_time=0., duration=1, sampling_frequency=2*8192): 
    """ 

    Defines and creates a dictionary for the initial fudicial parameters for the waveforms used when creating the NR .h5 file. These parameters are redefined with random values when calculating the SNR.

    Parameters
    ==========
    distance: float, optional
        Distance to system.
        If not provided will fall back to 40.
        
    phase: float, optional
        Binary phase at a reference frequency.
        If not provided will fall back to 0.
        
    ra: float, optional
        Right ascension of system.
        If not provided will fall back to 0.
        
    ra: float, optional
        Declination of system.
        If not provided will fall back to 0.
        
    psi: float, optional
        Polarization angle of the source.
        If not provided will fall back to 0.
        
    t0: float, optional
        Starting time of the time array.
        If not provided will fall back to 0.
        
    geocent_time: float, optional
        GPS reference time at the geocenter, near merger time.
        If not provided will fall back to 0.
        
    duration: float, optional
        Duration of time array.
        If not provided will fall back to 1.
        
    sampling_frequency: float, optional
        The sampling frequency. Should be atleast twice the maximum frequency of the detector sensitivity curve.
        If not provided will fall back to 8*8192.

    Returns
    =======
    int: duration
    int: sampling frequency
    dict: injection parameters

    """
    injection_parameters = dict(distance=distance, phase=phase, ra=ra, dec=dec, psi=psi, t0=t0, geocent_time=geocent_time)
    return duration, sampling_frequency, injection_parameters 

def get_info(simulation_name, txtfile = 'metadata.txt'):
    """ 

    Reads in the metadata for each simulation. 
    For this function to work the metadata and h5 file for each simulation must be placed in a folder with the simulation name alongside the waveform file.

    Parameters
    ==========
    simulation_name: str
        Simulation name - can be found from the simulation key.
        
    txtfile: str, optional
        File name for the metadata.
        If not provided will fall back to 'metadata.txt'.

    Returns
    =======
    float: Mass of A.
    float: Mass of B.
    float: Total mass.
    str: EOS
    float: Mass ratio.

    """
    data = pd.read_fwf(input_path + simulation_name + '/' + txtfile)
    equalise = data[data.iloc[:,0].str.contains('Initial')==True]
    index = equalise.index
    clean = data.iloc[index[0]+1:]
    clean2 = clean.iloc[:, 0].str.split('=', expand=True)
    findinga = data[data.iloc[:,0].str.contains('id_mass_starA')==True]
    indexa = findinga.index
    massA = float(clean2.iloc[indexa[0]-(index[0]+1),1])

    findingb = data[data.iloc[:,0].str.contains('id_mass_starB')==True]
    indexb = findingb.index
    massB = float(clean2.iloc[indexb[0]-(index[0]+1),1])

    findingt = data[data.iloc[:,0].str.contains('id_mass')==True]
    indext = findingt.index
    masst = float(clean2.iloc[indext[0]-(index[0]+1),1])

    findingeos = data[data.iloc[:,0].str.contains('id_eos')==True]
    indexeos = findingeos.index
    EOS = clean2.iloc[indexeos[0]-(index[0]+1),1]
    EOS = EOS.strip()

    findingr = data[data.iloc[:,0].str.contains('id_mass_ratio')==True]
    indexr = findingr.index
    ratio = float(clean2.iloc[indexr[0]-(index[0]+1),1])
    return massA, massB, masst, EOS, ratio

def set_simulation_name(sim_name):
    """
    Sets the global simulation_name variable. 

    Parameters
    ==========
    sim_name: str
        The desired simulation name to be set.

    Returns
    =======
    str: Same as input string.
    """
    global simulation_name
    simulation_name = sim_name
    return simulation_name
## Need to convert from natural units (c=G=Msun=1) to SI

#The time and distance conversions are not correct for every waveform - soem are not scaled by the mass of the Sun, but by the mass of the binary system. It is currently unknown which waveform is scaled by what.
def time_geo_to_s(time_in_geo):
    """ 

    Converts the time given in the simulation to seconds. 

    Parameters
    ==========
    time_in_geo: float
        Time in geo (time given in the simulation data).

    Returns
    =======
    float: Time in seconds.

    """
    constants = c.c.value**3 / (c.M_sun.value * c.G.value)
    return time_in_geo /constants

def dist_geo_to_Mpc(dist_in_geo):
    """ 

    Converts the distance given in the simulation to Mpc. 

    Parameters
    ==========
    dist_in_geo: float
        Distance in geo (distances given in the simulation data).

    Returns
    =======
    float: Distance in Mpc.

    """
    constants = c.c.value**2 / (c.M_sun.value * c.G.value)
    dist_in_m = dist_in_geo / constants
    return dist_in_m / u.Mpc.to('m')

## create injection function for the write form for Bilby
def NR_injection_into_Bilby(time, distance, **waveform_kwargs):
    """ 

    Converts and extracts the waveform for injection from the full simulation data. 

    Parameters
    ==========
    Model to be used in the Bilby waveform generator function. 

    Returns
    =======
    dict: hplus and hcross

    """
    
    NR_directory = input_path + simulation_name + '/'
    NR_filename = NR_directory + 'data.h5'  

    ## open NR file and get data
    ff = h5py.File(NR_filename, 'r')
    # l=m=2 waveforms, evaluated at coordinate radius of 600 Msun
    # Options will differ per waveform (other options here are 450 and 500 Msun)
    # other modes are available as well (22 should always be dominant) 
    # M - since not all the sims have 400, just take the lowest one available
    keys = list(ff['rh_22'].keys())
    rh_22 = ff['rh_22'][keys[0]]  

    sim_time = time_geo_to_s(rh_22[:, -1])

    ### NEED TO CHECK DISTANCES -- I DON'T THINK THIS IS RIGHT
    ## (haven't accounted for the fact this is measured at distance of 600 Msun)

    hp = dist_geo_to_Mpc(rh_22[:, 1] / distance) #*dist_geo_to_Mpc(600))  # hplus #so I guess it's just scaled directly by divideing by the dis
    hc = dist_geo_to_Mpc(rh_22[:, 2] / distance )#*dist_geo_to_Mpc(600))  # hcross $does this mean you just multiply by 600???


    ## figure out where post-merger starts. (just find max strain) and get rid of inspiral
    postmerger_start_idx = np.argmax(np.sqrt(hp**2 + hc**2))

    sim_time = sim_time[postmerger_start_idx:] - sim_time[postmerger_start_idx] # make t = 0 start of postmerger
    hp = hp[postmerger_start_idx:]
    hc = hc[postmerger_start_idx:]

    ## interpolate onto bilby time grid
    hplus_interp_func = interp1d(sim_time, hp, bounds_error=False, fill_value=0)
    hcross_interp_func = interp1d(sim_time, hc, bounds_error=False, fill_value=0)

    hplus = hplus_interp_func(time)
    hcross = hcross_interp_func(time)

    return {'plus': hplus, 'cross': hcross}

def hplot(simulation_name, output_folder_name):
    """ 

    Plots the hplus, hcross, and total amplitude of the waveform from a given simulation. Currently does not save the figure to an output location; the part of the code does this is commented out, but the # can be deleted. The figure can also be saved by just using the plt.savefig() after hplot().

    Parameters
    ==========
    simulation_name: str
        Simulation name - can be found from the simulation key.
    
    output_name: str
        Folder where the figures get saved to.

    """
    filename = input_path + simulation_name + '/'+'data.h5'
    ff = h5py.File(filename, 'r')
    keys = list(ff['rh_22'].keys())
    final = ff['rh_22'][keys[0]] 
    df = final[()]
    timeingeo = df[:,-1]
    hpingeo = df[:,1]
    hcingeo = df[:,2] #get relevant data from the h.5 file
    
    time = time_geo_to_s(timeingeo)
    hp = dist_geo_to_Mpc(hpingeo)
    hc = dist_geo_to_Mpc(hcingeo) #convert to the relevant units
    
    postmerger_start_idx = np.argmax(np.sqrt(hp**2 + hc**2))
    time = time - time[postmerger_start_idx] # make t = 0 start of postmerger
    #hp = hp[postmerger_start_idx:]
    #hc = hc[postmerger_start_idx:]
    hplus_interp_func = interp1d(time, hp, bounds_error=False, fill_value=0)
    hcross_interp_func = interp1d(time, hc, bounds_error=False, fill_value=0)
    hplus = hplus_interp_func(time)
    hcross = hcross_interp_func(time)
    fig, (ax1,ax2, ax3) = plt.subplots(figsize=(20,5), nrows=3, sharex=True)
    axes = [ax1,ax2, ax3]
    #plt.subplots_adjust(hspace=.0)
    ax1.plot(time, np.sqrt(hplus**2 + hcross**2), label = 'Total Amplitude', alpha = 0.8, color = 'royalblue')
    ax2.plot(time, hplus, label = r'$h_+(t)$', alpha = 0.8, color = 'darkorange')
    ax3.plot(time, hcross, label = r'$h_{\times}(t)$', alpha = 0.8, color = 'teal')
    
    for i in axes:
        i.axvline(0, color = 'red')
        i.tick_params(axis="x", labelsize=25)
        i.tick_params(axis="y", labelsize=25)
        i.xaxis.offsetText.set_fontsize(20)
        i.yaxis.offsetText.set_fontsize(0)
        i.legend(fontsize=25, loc = 1)

    ax1.yaxis.offsetText.set_fontsize(20)
    ax3.set_xlabel('Time (s)', fontsize=30)
    ax2.set_ylabel('Strain',fontsize=30)

    #fig.suptitle("NR Waveform Data - Unscaled to Distance")
    #fig.savefig(output_folder_name + '/' + 'EOS_{}'.format(EOS) + '/' + simulation_name + '/' + 'Waveform.png') 
    #plt.close('all')
    
def collect(i,df,duration,sampling_frequency, injection_parameters, hf_name , output_folder_name):
    """ 

    This collects the necessary data and appends it to the .h5 file for a simulation. 
    To get data for all the simulations in the simulation key, loop over the key with a for loop.

    Parameters
    ==========
    i: float
        Index number for the simulation in question (from the key).
        
    df: str
        Name for simulation key.
    
    duration: float
        Duration of time array from initial conditions.
    
    sampling_frequency: float
        The sampling frequency from initial conditions.
        
    injection_parameters: dict
        Injection parameters.
        
    hf_name: str
        Name of the .h5 file created earlier.
        
    output_name: str
        Folder where the figures get saved to. Currently not outputting figures; to do so uncomment the last two lines of the hplot() functions.

    Returns
    =======

    """
    global simulation_name
    simulation_name = df.iloc[i,1]
    simnum = df.iloc[i,0]
    metadata = 'metadata.txt' #define filenames
    
    waveform = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
            time_domain_source_model=NR_injection_into_Bilby,
            parameters=injection_parameters,
            start_time=injection_parameters['geocent_time']);
    
    signalpc = waveform.frequency_domain_strain() #gives frequency on x-axis and strain on y-axis - i.e. the index is the frequency
    signal = ((np.abs(signalpc['plus'])**2+np.abs(signalpc['cross'])**2)/2)**0.5 #is this correct?
    freq_array = waveform.frequency_array
    array = pd.DataFrame({'Freq':freq_array, 'Strain':signal} )    
    cutting_index = int((array.loc[array['Freq'] == min_cutting_freq]).index.values)
    array = array.iloc[cutting_index:]
    max_index = array.idxmax(0)[1]
    freqpeak = array.iloc[max_index-cutting_index,0]
    
    filename = input_path + simulation_name + '/'+'data.h5' #find the duration of the signal
    ff = h5py.File(filename, 'r')
    keys = list(ff['rh_22'].keys())
    final = ff['rh_22'][keys[0]] 
    df = final[()]
    timeingeo = df[:,-1]
    time = time_geo_to_s(timeingeo)
    hpingeo = df[:,1]
    hcingeo = df[:,2]
    hp = dist_geo_to_Mpc(hpingeo)
    hc = dist_geo_to_Mpc(hcingeo)
    postmerger_start_idx = np.argmax(np.sqrt(hp**2 + hc**2))
    time = time - time[postmerger_start_idx] # make t = 0 start of postmerger
    newd = max(time) #last value of the list, the largest, will then be the duration
    
    global EOS
    massa, massb, masst,EOS,ratio  = get_info(simulation_name,metadata)
    #path = output_folder_name + '/' + 'EOS_{}'.format(EOS) + '/' + simulation_name ##create path to save figures if hplot() function is active
    #if not os.path.isdir(path):
    #    os.makedirs(path)
    b = h5py.File(hf_name, "a")
    indexs = b[EOS].shape #this separates out the values by EOS
    b[EOS].resize((indexs[0]+1,indexs[1])) #resize .h5 file
    b[EOS][indexs[0]] = [simnum, masst, massa, massb, ratio, freqpeak, newd] #append columns #information to be appended; corresponds to columns in make_h5().
    b.close()

    # return EOS, simulation_name, masst, massa, massb, ratio, freqpeak, newd

    # return simnum, masst, massa, massb, ratio, freqpeak, newd
    #hplot(simulation_name, output_folder_name)
    # FIND POSSIBLE UNSCALED WAVEFORMS (uncomment below to save plots)
    # if freqpeak > 4000 or freqpeak < 1500:
    #     plt.yscale("log")
    #     plt.plot(freq_array, signal)
    #     plt.axvline(x = freqpeak, color = 'r')
    #     plt.title(f"{simulation_name}, massa={massa}, massb={massb}, masst={masst}, EOS={EOS}, ratio={ratio}, freqpeak={freqpeak}", wrap=True)
    #     plt.ylabel('Signal')
    #     plt.xlabel('Frequency [Hz]')
    #     plt.xlim(0,10000)
    #     plt.savefig(f"susWaveforms/{simulation_name}_{massa}_{massb}_{masst}_{EOS}_{ratio}_{round(freqpeak)}.png")
    #     return True
    #     plt.close()


def IFOmaker(config_name,duration,sampling_frequency, geocent_time, separation, model, filename = None):
    """ 

    This function defines an interferometer given the fiducial conditions and some parameters set in the function. These can be changed manually. This function defines interferometers from a file where the first column is the frequency, and the second column is the power spectral density. 

    Parameters
    ==========  
    config_name: str
        Name of the detector for labels, etc.     
    
    duration: float
        Duration of time array from initial conditions.
    
    sampling_frequency: float
        The sampling frequency from initial conditions.
        
    geocent_time: float, optional
        GPS reference time at the geocenter, near merger time. Can be called from the initial injection_parameters set (set to 0).
        
    separation: str
        Symbol used to separate the values in the configuration file, for example " ".
        
    model: str
        The type of sensitivity curve you are reading in. It must be either 'psd' for the power spectral density, 'asd' for the amplitude spectral density and 'preset' if you are using one of Bilby's preset IFOs, e.g. 'CE2silicon'.
        
    filename: str
        If using the asd or psd model please input the name of the file here, otherwise will default to None.
        
    Returns
    =======
    NemIFO: Interferometer.
    config_name: Configuration name.

    """
    
    if model == 'psd':
        df2 = pd.read_csv(filename, sep=separation,header=None)
        freqarr = np.array(df2[0])
        psd = np.array(df2[1])

    if model == 'asd':
        df2 = pd.read_csv(filename, sep=separation,header=None, skiprows=1)
        freqarr = np.array(df2[0])
        psd = np.array(df2[1]**2)
        
    if model == 'preset':
        N = 100000
        freqarr = np.geomspace(fmin,fmax,N,endpoint=True)
        budget_ap = gwinc.load_budget(named)
        traces = budget_ap.run(freq=freqarr)
        #save the psd 
        psd = traces.psd
    

    NemIFO = bilby.gw.detector.Interferometer(
      power_spectral_density=bilby.gw.detector.PowerSpectralDensity(
       frequency_array=freqarr, psd_array=psd),
      name='NemIFO', length=NEMO_length,
       minimum_frequency=fmin, maximum_frequency=fmax,
       latitude=NEMO_latitude, longitude=NEMO_longitude,
       elevation=NEMO_elevation, xarm_azimuth=NEMO_xarm_azimuth, yarm_azimuth=NEMO_yarm_azimuth)
    NemIFO.set_strain_data_from_power_spectral_density(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=geocent_time - 0.5);
    config_name = config_name
    return NemIFO, config_name

def IFOmakerFromASDArray(config_name,duration,sampling_frequency, geocent_time, fsig, ASDarr):
    """ 

    This function defines an interferometer given the fiducial conditions and some parameters set in the function. These can be changed manually. This function defines interferometers from a frequency array and ASD array. 

    Parameters
    ==========  
    config_name: str
        Name of the detector for labels, etc.     
    
    duration: float
        Duration of time array from initial conditions.
    
    sampling_frequency: float
        The sampling frequency from initial conditions.
        
    geocent_time: float, optional
        GPS reference time at the geocenter, near merger time. Can be called from the initial injection_parameters set (set to 0).
        
    fsig: list
        Arrray of frequencies.
        
    ASDarr : list
        Array of amplitude spectral density values for the interferometer.
        
    Returns
    =======
    NemIFO: Interferometer.
    config_name: Configuration name.

    """
    
    freqarr = np.array(fsig)
    psd = np.array(ASDarr**2)
    
    NemIFO = bilby.gw.detector.Interferometer(
      power_spectral_density=bilby.gw.detector.PowerSpectralDensity(
       frequency_array=freqarr, psd_array=psd),
      name='NemIFO', length=NEMO_length,
       minimum_frequency=fmin, maximum_frequency=fmax,
       latitude=NEMO_latitude, longitude=NEMO_longitude,
       elevation=NEMO_elevation, xarm_azimuth=NEMO_xarm_azimuth, yarm_azimuth=NEMO_yarm_azimuth)
    NemIFO.set_strain_data_from_power_spectral_density(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=geocent_time - 0.5);
    config_name = config_name
    return NemIFO, config_name
    
def random_param(df,maxdist,constant_merger_rate_density_per_redshift, scalewavno=1): 
    """ 
        
    This function oututs a DataFrame of randomised injection parameters and the time used to scale the total number of parameters and waveforms used. First, the number of events is scaled to be equal to the number of waveforms, and then this number can be dubled, tripled, etc. using the scalewavno. Currently, this function randomises the distance, phase, ra, dec, and psi.

    Parameters
    ==========  
    df: str
        DataFrame given by the sim_index() function. Used to scale the total number of waveforms.
    
    maxdist: float + units
        Maximum observing distance of the detectors, needs astropy units - e.g. u.Mpc.
    
    constant_merger_rate_density_per_redshift: float
        The merger event rate in Gpc^-3 yr^-1.
    
    scalewavno: int
        Scaling factor for the time and number of waveforms used. i.e. scalewavno=2 doubles the time and the number of waveforms used to calculate the detection rate.
        If not set will fall back to 1.

    Returns
    =======
    time: Number of years the number of events in each bin were scaled to in order to get a total of 165*scalewavno waveforms.
    random_param: DataFrame of length equivalent to the number of waveforms being injected with each row being a set of randomised injection parameters.

    """
    total = merger_rates_for_mallika.number_in_cmd_bin((1*u.Mpc, maxdist),constant_merger_rate_density_per_redshift) #min. dist. 1 Mpc
        
    
    time = len(df)/(total)*scalewavno
    # print(f"Observing time: {time}")
    priors = bilby.gw.prior.BNSPriorDict()
    priors['luminosity_distance'] = bilby.gw.prior.UniformComovingVolume(1, maxdist/u.Mpc, name='luminosity_distance')
    params = priors.sample(len(df)*scalewavno)
    
    #this is if the Bilby priors distribution is incorrect
    #step_1_dec = np.random.uniform(low=0.0, high=1.0, size=len(df)*scalewavno)
    #decs = np.arcsin(2*step_1_dec-1)
    #params['dec'] = decs
    
    d = {'distance': (params['luminosity_distance']), 'phase': params['phase'],
     'ra': params['ra'], 'dec': params['dec'], 'psi': params['psi']}
    random_param = pd.DataFrame(data = d)
    
    return  time, random_param

def calc_SNR(i, duration,sampling_frequency, injection_parameters,IFO,df):
    """ 

    This function injects the randomised parameters and gives the optimal SNR. This function should be put in a for loop over the total number of simulations multplied by the scalewavno used earlier.

    Parameters
    ==========
    i: float
        Index number for the simulation in question (from the key).
    
    duration: float
        Duration of time array from initial conditions.
    
    sampling_frequency: float
        The sampling frequency from initial conditions.
        
    injection_parameters: dict
        Injection parameters.
        
    IFO: str
        Interferometer name.
    
    df: str
        Name for simulation key.

    Returns
    =======
    SNRO: Optimal signal to noise ratio value of the waveform at the defined injection parameters.

    """
    global simulation_name
    simulation_name = df.iloc[i,1]
    simnum = df.iloc[i,0]
    AusIFO = IFO #you need to redefine the IFO and not just use IFO because then the injections will all be done to the same IFO and stack
    metadata = 'metadata.txt'
    waveform = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            time_domain_source_model=NR_injection_into_Bilby,
            parameters=injection_parameters,
            start_time=injection_parameters['geocent_time']); #define waveform in Bilby
    AusIFO.inject_signal(waveform_generator=waveform, 
            parameters=injection_parameters); #inject signal into AusIFO
    signal = AusIFO.get_detector_response(waveform.frequency_domain_strain(), injection_parameters) #define the signal of the detector
    SNRO = np.sqrt(AusIFO.optimal_snr_squared(signal)) #find the optimal SNR from that signal
    return SNRO