"""James Gardner, April 2022"""
from numpy import pi as PI
from scipy.integrate import quad

# this is the cosmology that I use, you can edit the code to use a different one if you want
from astropy.cosmology import Planck18


def differential_comoving_volume(z):
    """$\frac{\text{d}V}{\text{d}z}(z)$ in B&S2022; 4*pi to convert from Mpc^3 sr^-1 (sr is steradian) to Mpc^3"""
    return 4.0 * PI * Planck18.differential_comoving_volume(z).value


def merger_rate(z, constant_merger_rate_density_per_redshift):
    """$R(z)$ in B&S2022; normalisation of merger rate density $\dot{n}(z)$ in the source frame to GWTC3_MERGER_RATE_BNS in https://arxiv.org/pdf/2111.03606v2.pdf. (GWTC3_MERGER_RATE_BNS, GWTC3_MERGER_RATE_BBH = 105.5, 23.9)
    1e-9 converts Gpc^-3 to Mpc^-3 to match Planck18, in Fig 2 of Ngetal2021: the ndot_F rate is in Gpc^-3 yr^-1, injections.py cites v1 of an arXiv .pdf
    defaults to merger_rate_density_per_redshift in units bns per year per Gpc^3 per redshift"""
    # you can edit this to allow for non-constant merger rates vs z if you find a model that you like
    return (
        constant_merger_rate_density_per_redshift  # bns per year per Gpc^3 per redshift
        * 1e-9  # --> bns per year per Mpc^3 per redshift
        * differential_comoving_volume(z)  # --> bns per year per redshift
    )


def merger_rate_in_obs_frame(z,constant_merger_rate_density_per_redshift, **kwargs):
    """1+z factor of time dilation of merger rate in observer frame z away. kwargs, e.g. normalisation, are passed to merger_rate"""
    # bns per year per redshift in observer's frame
    return merger_rate(z,constant_merger_rate_density_per_redshift, **kwargs) / (1 + z)


def number_of_sources_in_redshift_bin(zmin, zmax,constant_merger_rate_density_per_redshift, **kwargs):
    """$D_R(z, \rho_\ast)|_{\varepsilon=1}$ in B&S2022;i.e. "merger rate" in Fig 2, not R(z) but int R(z)/(1+z), i.e. if perfect efficiency. the redshift bin is (zmin, zmax)"""
    # use this like: number_of_sources_in_redshift_bin(0, 0.01) or number_of_sources_in_redshift_bin(0, 0.01, constant_merger_rate_density_per_redshift=200)
    # you could add a redshift dependent detection efficiency in the integral if you wanted to
    # you can wrap this with some astropy.cosmology to make an integral over luminosity distance
    return quad(
        lambda z: merger_rate_in_obs_frame(z,constant_merger_rate_density_per_redshift, **kwargs),
        zmin,
        zmax,
    )[0]


from astropy.cosmology import z_at_value
import astropy.units as u
from merger_rates_for_mallika import *


def z_for_Dcm(Dcm):
    return z_at_value(Planck18.comoving_distance, Dcm)


def number_in_cmd_bin(comoving_distance_bin, constant_merger_rate_density_per_redshift):
    """Dcm with units"""
    zmin, zmax = map(z_for_Dcm, comoving_distance_bin)
    return number_of_sources_in_redshift_bin(zmin, zmax,constant_merger_rate_density_per_redshift)


# number_in_cmd_bin((93*u.Mpc, 100*u.Mpc))
