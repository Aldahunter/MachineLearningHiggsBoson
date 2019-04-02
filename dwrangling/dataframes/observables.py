"""DataFrames.Observables - Data and functions associated with reconstructing, \
and manipulating, the observables used for the machine learning."""

import numpy as np
import pandas as pd

import dwrangling as DW
import dwrangling.dataframes as DWDF



### Observables Data ###

#: All possible observables in a :class:`dwrangling.dataframes.ODataFrame`.
all_observables = ['mu-_m', 'mu-_E', 'mu-_px', 'mu-_py',
                  'mu-_pz', 'mu-_pT', 'mu-_rap',
                  'mu+_m', 'mu+_E', 'mu+_px', 'mu+_py',
                  'mu+_pz', 'mu+_pT', 'mu+_rap',
                  'e+_m', 'e+_E', 'e+_px', 'e+_py',
                  'e+_pz', 'e+_pT', 'e+_rap',
                  'e-_m', 'e-_E', 'e-_px', 'e-_py',
                  'e-_pz', 'e-_pT', 'e-_rap',
                  'Z_mu_m', 'Z_mu_E', 'Z_mu_px', 'Z_mu_py',
                  'Z_mu_pz', 'Z_mu_pT', 'Z_mu_rap',
                  'Z_e_m', 'Z_e_E', 'Z_e_px', 'Z_e_py',
                  'Z_e_pz', 'Z_e_pT', 'Z_e_rap',
                  'delR_mu', 'delR_e', 'delR_Z', 'm_H',
                  'signal']


### Hidden Functions ###
def _docstring_parameter(*sub, **kwsub):
    """Formats an objects docstring."""
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub, **kwsub)
        return obj
    return dec



### Observable Functions ###
def calc_rapditiy(p_E, p_pT, p_pz):
    """Calculates the rapidity of a particle with respect to the beam axis.
    
    Parameters:
     - p_E: The :class:`pandas.DataFrame` column for the particle's energy;
     - p_pT: The :class:`pandas.DataFrame` column for the  particle's
             transverse momentum;
     - p_pz: The :class:`pandas.DataFrame` column for the  particle's beam
             axis momentum.
    
    Returns:
     - p_rap: The particle's rapidity as a :class:`pandas.Series` object."""

    # Create zero pd.Series, to ensure p_eff_m2 >= 0
    zero = DWDF.zero_col(p_E)['zeros']

    # Calculate the effective mass squared
    p_eff_m2 = DWDF.pick_row_max(p_E**2 - (p_pT**2 + p_pz**2), zero)['max']

    # Calculate the particle's rapidity relative to beam axis
    E_plus_pz = p_E + np.abs(p_pz)
    p_rap = 0.5 * np.log( (p_pT**2 + p_eff_m2) / E_plus_pz**2 )

    # Fix sign for negative beam axis (z-compontent) momentums 
    coefficents = pd.Series(np.where( p_pz > 0.0 , -1.0, 1.0))
    p_rap = p_rap * coefficents

    # Return the result
    return p_rap

def add_angles(angle1, angle2):
    """Adds two angles and keeps them between -pi and pi."""
    angle_sum = angle1 + angle2
    mask = angle_sum > np.pi
    angle_sum -= mask * 2*np.pi
    angle_sum += ~mask * 2*np.pi
    return angle_sum
    


def azimuth_angle(p_px, p_py):
    """Calculates the azimuth angle in radians of a particle with respect \
    to the x-y plane from the x-axis.
    
    Parameters:
      - p_px: The :class:`pandas.DataFrame` column for the  particle's x
              axis momentum;
      - p_pz: The :class:`pandas.DataFrame` column for the  particle's x
              axis momentum.
    
    Returns:
     - p_azi: The particle's azimuth angle in radians as a 
              :class:`pandas.Series` object."""
    return np.arctan2(p_py, p_px)


def angular_seperation(p1_azi, p1_rap, p2_azi, p2_rap):
    """Calculates the lorentz invarient angular seperation between two \
    particles, p1 and p2.
    
    Parameters:
      - pn_azi: The :class:`pandas.DataFrame` column for particle n's
                azimuth angle (where  n is [1,2]);
      - pn_rap: The :class:`pandas.DataFrame` column for particle n's
                rapidity (where  n is [1,2]).
    
    Returns:
     - p_delR: The angular seperation in radians between the two particles
               as a :class:`pandas.Series` object."""
    delta_azi = add_angles(p2_azi, -1.0 * p1_azi)
    delta_rap = p2_rap - p1_rap
    return np.sqrt(np.square(delta_azi) + np.square(delta_rap))


def add_angular_seperation(dataframe):
    
    # Calculate the angular seperation for the muons
    mu1_azi = azimuth_angle(dataframe['mu-_px'], dataframe['mu-_py'])
    mu2_azi = azimuth_angle(dataframe['mu+_px'], dataframe['mu+_py'])
    mu_delR = angular_seperation(mu1_azi, dataframe['mu-_px'],
                                 mu2_azi, dataframe['mu+_px'])
    
    # Calculate the angular seperation for the electrons
    e1_azi = azimuth_angle(dataframe['e-_px'], dataframe['e-_py'])
    e2_azi = azimuth_angle(dataframe['e+_px'], dataframe['e+_py'])
    e_delR = angular_seperation(e1_azi, dataframe['e-_px'],
                                e2_azi, dataframe['e+_px'])
    
    # Assign signal column to end of the DataFrame
    dataframe['delR_mu'] = mu_delR
    dataframe['delR_e'] = e_delR
    dataframe = DWDF.df_move_to_last_col(dataframe, 'signal')
    
    return dataframe


def add_higgs_mass(dataframe):
    """Adds the Higg's mass observable to the event DataFrame.
    
    Parameters:
     - dataframe: A :class:`pandas.DataFrame`, containing the the columns
                  ('E', 'pT') for each of the four lepton particles.
    
    Returns:
     - dataframe: A :class:`pandas.DataFrame` with the Higg's Mass column
                  ('m_H')."""
    
    # Obtain E and pT values of the combined leptons
    E = (dataframe['mu-_E'] + dataframe['mu+_E']
         + dataframe['e+_E'] + dataframe['e-_E'])
    p_x = (dataframe['mu-_px'] + dataframe['mu+_px']
           + dataframe['e+_px'] + dataframe['e-_px'])
    p_y = (dataframe['mu-_py'] + dataframe['mu+_py']
           + dataframe['e+_py'] + dataframe['e-_py'])
    p_z = (dataframe['mu-_pz'] + dataframe['mu+_pz']
           + dataframe['e+_pz'] + dataframe['e-_pz'])
    
    # Create a zero column, so that negative values are not square-rooted.
    zero = DWDF.zero_col(dataframe)
    # Calculate the Higg's mass for these events
    m_H = np.sqrt( DWDF.pick_row_max( E**2 - (p_x**2
                                              + p_y**2
                                              + p_z**2),
                                     zero) )
    
    # Assign Higg's mass to penultimate column of the DataFrame
    dataframe['m_H'] = m_H
    dataframe = DWDF.df_move_to_last_col(dataframe, 'signal')
    
    # Return the DataFrame
    return dataframe


def add_Z_bosons(dataframe):
    """Adds both Z Bosons observables to the event DataFrame.
    
    Parameters:
     - dataframe: A :class:`pandas.DataFrame`, containing the the
                  observables ('E', 'px', 'py', 'pz') for each of the four
                  lepton particles.
    
    Returns:
     - dataframe: A :class:`pandas.DataFrame` with the two Z Boson values
                  ('Z_pT', 'z_rap') for both bosons ('Z_mu', 'Z_e')."""
    
    # Calculate the four momentum components
    Z_mu_E = dataframe['mu-_E'] + dataframe['mu+_E']
    Z_mu_px = dataframe['mu-_px'] + dataframe['mu+_px']
    Z_mu_py = dataframe['mu-_py'] + dataframe['mu+_py']
    Z_mu_pz = dataframe['mu-_pz'] + dataframe['mu+_pz']
    
    Z_e_E = dataframe['e-_E'] + dataframe['e+_E']
    Z_e_px = dataframe['e-_px'] + dataframe['e+_px']
    Z_e_py = dataframe['e-_py'] + dataframe['e+_py']
    Z_e_pz = dataframe['e-_pz'] + dataframe['e+_pz']
    
    # Calculate Transverse Momentums
    Z_mu_pT = np.sqrt(  Z_mu_px**2 + Z_mu_py**2 )
    Z_e_pT = np.sqrt(  Z_e_px**2 + Z_e_py**2 )
    
    # Calculate Azimuth Angles
    Z_mu_azi = azimuth_angle(Z_mu_px, Z_mu_py)
    Z_e_azi = azimuth_angle(Z_e_px, Z_e_py)
    
    # Calculate Rapidity
    Z_mu_rap = calc_rapditiy(Z_mu_E, Z_mu_pT, Z_mu_pz)
    Z_e_rap = calc_rapditiy(Z_e_E, Z_e_pT, Z_e_pz)
    
    # Calculate Z Boson Angular Seperation
    Z_delR = angular_seperation(Z_mu_azi, Z_mu_rap, Z_e_azi, Z_e_rap)
    
    # Claclulate Z masses
    zero = DWDF.zero_col(dataframe)
    Z_mu_m = np.sqrt(DWDF.pick_row_max(Z_mu_E**2 - (Z_mu_px**2 + Z_mu_py**2
                                                    + Z_mu_pz**2),
                                       zero))
    zero = DWDF.zero_col(dataframe)
    Z_e_m = np.sqrt(DWDF.pick_row_max(Z_e_E**2 - (Z_e_px**2 + Z_e_py**2
                                                  + Z_e_pz**2),
                                      zero))
    
    # Add calculated values to the events DataFrame
    dataframe['Z_mu_E'] = Z_mu_E
    dataframe['Z_mu_px'] = Z_mu_px
    dataframe['Z_mu_py'] = Z_mu_py
    dataframe['Z_mu_pz'] = Z_mu_pz
    dataframe['Z_mu_pT'] = Z_mu_pT
    dataframe['Z_mu_rap'] = Z_mu_rap
    dataframe['Z_mu_m'] = Z_mu_m
    
    dataframe['Z_e_E'] = Z_e_E
    dataframe['Z_e_px'] = Z_e_px
    dataframe['Z_e_py'] = Z_e_py
    dataframe['Z_e_pz'] = Z_e_pz
    dataframe['Z_e_pT'] = Z_e_pT
    dataframe['Z_e_rap'] = Z_e_rap
    dataframe['Z_e_m'] = Z_e_m
    
    dataframe['delR_Z'] = Z_delR
    
    # Bring the 'signal' column to the end of the DataFrame
    dataframe = DWDF.df_move_to_last_col(dataframe, 'signal')
    
    # Return the events DataFrame
    return dataframe


def add_reconstructed_observables(dataframe):
    """Adds both Z Bosons observables and the reconstructed Higgs mass \
    to the DataFrame.
    
    Parameters:
     - dataframe: A :class:`pandas.DataFrame`, containing the the
                  observables 'E', 'px', 'py', 'pz') for each of the four
                  lepton particles.
    
    Returns:
     - dataframe: A :class:`pandas.DataFrame` with the two Z Boson values
                  ('Z_pT', 'z_rap') for both bosons ('Z_mu', 'Z_e') and the
                  Higg's Mass column ('m_H')."""
    
    dataframe = dataframe.copy()
    dataframe = add_angular_seperation(dataframe)
    dataframe = add_Z_bosons(dataframe)
    dataframe = add_higgs_mass(dataframe)
    dataframe = DWDF.df_move_to_last_col(dataframe, 'signal')
    return dataframe


@_docstring_parameter(collisions=DW.collisions)
def get_opt_observables_dataframe(collision, dataframe):
    """For a given collision, returns the dataframe with only the \
    optimal observables.

    Parameters:
     - collision: A :class:`str` from {collisions}; 
     - dataframe: A :class:`pandas.DataFrame`, containing the the
                  observables ('E', 'px', 'py', 'pz') for each of the four
                  lepton particles.
    
    Returns:
     - dataframe: A :class:`dwrangling.dataframes.ODataFrame` with only the
                  optimal observables for the given collision."""
    
    # Add all the reconstructed observables to the DataFrame
    dataframe = add_reconstructed_observables(dataframe)
    
    # Retrieve the optimal observables for this collision
    collision_observables = DW.get_collision_observables(collision)
    if 'signal' not in collision_observables:
        collision_observables += ['signal',]
    
    # Return an ObservablesDataFrame with these observables
    return DWDF.ODataFrame( dataframe[collision_observables] )