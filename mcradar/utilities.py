# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst


import numpy as np

g = 9.81 # gravitational acceleration [m/s^2]
rho_i = 917.6 # density of ice [kg/m^3]
def lin2db(data):
    """
    Convert from linear to logarithm units
    
    Parameters
    ----------
    data: single value or an array
    
    Returns
    -------
    returns the data converted to dB
    """

    return 10*np.log10(data)

def db2lin(data):
    """
    Convert from logarithm to linear units
    
    Parameters
    ----------
    
    data: single value or an array
    
    Returns
    -------
    returns the data converted to linear
    """
    
    return 10**(data/10.)
    
def fall_velocity_HW(area, mass, D_max, T=273.15,P=1000e2):
    """The Heymsfield-Westbrook fall velocity.

    Args:
        area: Projected area [m^2].
        mass: Particle mass [kg].
        D_max: Particle maximum dimension [m].
        T: Ambient temperature [K].
        P: Ambient pressure [Pa].

    Returns:
        The fall velocity [m/s].
    """
    do_i = 8.0
    co_i = 0.35

    rho_air = air_density(T, P)
    eta = air_dynamic_viscosity(T)

    # modified Best number eq. on p. 2478
    Ar = area / (np.pi/4)
    Xbest = rho_air * 8.0 * mass * g * D_max / (eta**2 * np.pi * 
        np.sqrt(Ar))

    # Re-X eq. on p. 2478
    c1 = 4.0 / ( do_i**2 * np.sqrt(co_i) )
    c2 = 0.25 * do_i**2
    bracket = np.sqrt(1.0 + c1*np.sqrt(Xbest)) - 1.0
    Re = c2*bracket**2

    return eta * Re / (rho_air * D_max)
def air_kinematic_viscosity(T, P):
    """The kinematic viscosity of air.

    Args:
        T: Ambient temperature [K].
        P: Ambient pressure [Pa].

    Returns:
        The kinematic viscosity [m^2/s].
    """
    rho = air_density(T, P)
    mu = air_dynamic_viscosity(T)
    return mu/rho


def air_dynamic_viscosity(T):
    """The kinematic viscosity of air.

    Args:
        T: Ambient temperature [K].

    Returns:
        The kinematic viscosity [Pa/s].
    """
    mu0 = 1.716e-5
    T0 = 273.15
    C = 111.0
    return mu0 * ((T0+C)/(T+C)) * (T/T0)**1.5


def air_density(T, P):
    """The density of air.

    Args:
        T: Ambient temperature [K].
        P: Ambient pressure [Pa].

    Returns:
        The kinematic viscosity [Pa/s].
    """
    R = 28704e-2 # gas constant for air
    return P / (T*R)
