# This module contains the functions that will be used to invert from reflectances to water quality parameters
# The dicionary `functions`` will be used by the waterquality module:
# function = {
#   'name_of_the_parameter': {'function': any_function}, 'units': 'units to be displayed in the report',
#   'name_of_the_parameter2': .....
# }

# Any bands can be used to compute the final value. The name of the band must match the internal name used by WaterDetect
# It is enough to put the band name as an argument in the function.
# Available bands in Sentinel2 are:
# Blue, Green, Red, RedEdg1, RedEdg2, RedEdg3, Nir, Nir2, Mir, Mir2

import numpy as np


#### Chlorophyll-a
# Gitelson
def chl_gitelson(Red, RedEdg1, RedEdg2):
    chl = 23.1 + 117.4 * (1 / Red - 1 / RedEdg1) * RedEdg2
    return chl

# Gitelson and Kondratyev, Dall'Olmo et al. (2003)
def chl_gitelson2(Red, RedEdg1, a=61.324, b=-37.94):
    chl = a * (RedEdg1 / Red) + b
    return chl

# OC2
def chl_OC2(Blue, Green, a=0.2389, b=-1.9369, c=1.7627, d=-3.0777, e=-0.1054):
    X = np.log10(Blue / Green)
    chl = 10**(a + b*X + c*X**2 + d*X**3 + e*X**4)
    return chl

# 2-band ratio by Gilerson et al. (2010)
def chl_gilerson2(Red, RedEdg1, a=35.75, b=1.124):
    chl = (a * (RedEdg1 / Red) - 19.30)**b
    return chl

# 2-band semi-analytical by Gons et al. (2003, 2005)
def chl_gons(Red, RedEdg1, RedEdg3, a=1.063, b=0.016, aw664=0.40, aw708=0.70):
    bb = 1.61 * RedEdg3 / (0.082 - 0.6 * RedEdg3)
    chl = ((RedEdg1 / Red) * (aw708 + bb) - aw664 - bb**a)/b
    return chl

# JM Hybride 1
def chl_h1(Red, RedEdge1, RedEdge2):
    res = RedEdge1 - Red  # B5 - B4
    chl = np.zeros_like(res)
    chl[res < 0] = 115.107 * RedEdge2[res < 0] * (1/Red[res < 0] - 1/RedEdge1[res < 0]) + 16.56
    chl[res >= 0] = 115.794 * RedEdge2[res >= 0] * (1/Red[res >= 0] - 1/RedEdge1[res >= 0]) + 20.678
    return chl

# JM Hybride 2
def chl_h2(Red, RedEdge1, RedEdge2):
    res = RedEdge1 - Red  # B5 - B4
    chl = np.zeros_like(res)
    chl[res < 0] = 46.859 * RedEdge1[res < 0]/Red[res < 0] - 29.916
    chl[res >= 0] = 115.794 * RedEdge2[res >= 0] * (1/Red[res >= 0] - 1/RedEdge1[res >= 0]) + 20.678
    return chl

#### SPM
# Nechad et al. (2010)
def nechad(Red, a=610.94, c=0.2324):
    spm = a * Red / (1 - (Red / c))
    return spm

# SEN3R SPM
def _spm_modis(Nir, Red):
    return 759.12 * ((Nir / Red) ** 1.92)

def _power(x, a, b, c):
    return a * (x) ** (b) + c

def spm_s3(Red, Nir2, cutoff_value=0.027, cutoff_delta=0.007, low_params=None, high_params=None):

    b665 = Red
    b865 = Nir2

    if cutoff_delta == 0:
        transition_coef = np.where(b665 <= cutoff_value, 0, 1)

    else:
        transition_range = (cutoff_value - cutoff_delta, cutoff_value + cutoff_delta)
        transition_coef = (b665 - transition_range[0]) / (transition_range[1] - transition_range[0])
        transition_coef = np.clip(transition_coef, 0, 1)

    low_params = [2.79101975e+05, 2.34858344e+00, 4.20023206e+00] if low_params is None else low_params
    high_params = [848.97770516, 1.79293191, 8.2788616] if high_params is None else high_params

    low = _power(b665, *low_params)
    high = _spm_modis(b865, b665)

    spm = (1 - transition_coef) * low + transition_coef * high
    return spm

#### Turbidity (FNU)
# Dogliotti
def turb_dogliotti(Red, Nir2):
    """Switching semi-analytical-algorithm computes turbidity from red and NIR band

    following Dogliotti et al., 2015
    :param water_mask: mask with the water pixels (value=1)
    :param rho_red : surface Reflectances Red  band [dl]
    :param rho_nir: surface Reflectances  NIR band [dl]
    :return: turbidity in FNU
    """

    limit_inf, limit_sup = 0.05, 0.07
    a_low, c_low = 228.1, 0.1641
    a_high, c_high = 3078.9, 0.2112

    t_low = nechad(Red, a_low, c_low)
    t_high = nechad(Nir2, a_high, c_high)
    w = (Red - limit_inf) / (limit_sup - limit_inf)
    t_mixing = (1 - w) * t_low + w * t_high

    t_low[Red >= limit_sup] = t_high[Red >= limit_sup]
    t_low[(Red >= limit_inf) & (Red < limit_sup)] = t_mixing[(Red >= limit_inf) & (Red < limit_sup)]
    t_low[t_low > 4000] = 0
    return t_low

#### CDOM
# Brezonik et al. 2005
def cdom_brezonik(Blue, RedEdg2):


    cdom = np.exp(1.872 - 0.830 * np.log(Blue/RedEdg2))

    return cdom


functions = {
    'SPM_Nechad': {
        'function': nechad,
        'units': 'mg/l'
    },
    
    'SPM_S3': {
        'function': spm_s3,
        'units': 'mg/l'
    },

    'TURB_Dogliotti': {
        'function': turb_dogliotti,
        'units': 'FNU'
    },

    'CHL_Gitelson': {
        'function': chl_gitelson2,
        'units': 'mg/m³'
    },

    'CHL_OC2': {
        'function': chl_OC2,
        'units': 'mg/m³'
    },

    'CHL_Gilerson': {
        'function': chl_gilerson2,
        'units': 'mg/m³'
    },

    'CHL_Gons': {
    'function': chl_gons,
    'units': 'mg/m³'
    },

    'CHL_Hybrid1': {
        'function': chl_h1,
        'units': 'mg/m³'
    },

    'CHL_Hybrid2': {
        'function': chl_h2,
        'units': 'mg/m³'
    },

    'CDOM_Brezonik': {
        'function': cdom_brezonik,
        'units': '',
    }
}