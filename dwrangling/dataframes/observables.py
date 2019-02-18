"""DataFrames.Observables - Data and functions associated with reconstructing, \
and manipulating, the observables used for the machine learning."""

import numpy as np
import pandas as pd

import dwrangling as DW
import dwrangling.dataframes as DWDF



### Observables Data ###
df_observables = ['mu-_m', 'mu-_E', 'mu-_px', 'mu-_py', 'mu-_pz', 'mu-_pT', 'mu-_rap',
                  'mu+_m', 'mu+_E', 'mu+_px', 'mu+_py', 'mu+_pz', 'mu+_pT', 'mu+_rap',
                  'e+_m', 'e+_E', 'e+_px', 'e+_py', 'e+_pz', 'e+_pT', 'e+_rap',
                  'e-_m', 'e-_E', 'e-_px', 'e-_py', 'e-_pz', 'e-_pT', 'e-_rap',
                  'Z_mu_pT', 'Z_mu_rap', 'Z_e_pT', 'Z_e_rap', 'm_H', 'signal']



### Observable Functions ###
def calc_rapditiy(p_E, p_pT, p_pz):
    """Calculates the rapidity of a particle with respect to the beam axis.
    
    Parameters:
     - p_E: The particle's energy.
     - p_pT: The particle's transverse momentum.
     - p_pz: The particle's beam axis momentum.
    
    Returns:
     - p_rap: The particle's rapidity as a pandas.Series object."""

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


def add_higgs_mass(event_df):
    """Adds the Higg's mass observable to the event DataFrame (event_df).
    
    Parameters:
     - event_df: An event DataFrame, containing the the observables (E, pT)
                 for each of the four lepton particles.
    
    Returns:
     - event_df: An event DataFrame with the Higg's Mass column (m_H)."""
    
    # Obtain E and pT values of the combined leptons
    E = event_df['mu-_E'] + event_df['mu+_E'] + event_df['e+_E'] + event_df['e-_E']
    p_x = event_df['mu-_px'] + event_df['mu+_px'] + event_df['e+_px'] + event_df['e-_px']
    p_y = event_df['mu-_py'] + event_df['mu+_py'] + event_df['e+_py'] + event_df['e-_py']
    p_z = event_df['mu-_pz'] + event_df['mu+_pz'] + event_df['e+_pz'] + event_df['e-_pz']
    
    # Create a zero column, so that negative values are not square-rooted.
    zero = DWDF.zero_col(event_df)
    # Calculate the Higg's mass for these events
    m_H = np.sqrt( DWDF.pick_row_max( E**2 - (p_x**2 + p_y**2 + p_z**2), zero) )
    
    # Assign Higg's mass to penultimate column of the DataFrame
    event_df['m_H'] = m_H
    event_df = DWDF.df_move_to_last_col(event_df, 'signal')
    
    # Return the DataFrame
    return event_df

def add_Z_bosons(event_df):
    """Adds both Z Bosons observables to the event DataFrame (event_df).
    
    Parameters:
     - event_df: An event DataFrame, containing the the observables 
                 (E, px, py, pz) for each of the four lepton particles.
    
    Returns:
     - event_df: An event DataFrame with the two Z Boson values
                 (Z_pT, z_rap) for both bosons (Z_mu, Z_e)."""
    
    
    # Calculate the transverse momentums
    Z_mu_pT = np.sqrt(  (event_df['mu-_px'] + event_df['mu+_px'])**2
                      + (event_df['mu-_py'] + event_df['mu+_py'])**2)
    
    Z_e_pT = np.sqrt(  (event_df['e-_px'] + event_df['e+_px'])**2
                     + (event_df['e-_py'] + event_df['e+_py'])**2)
    
    # Calculate the energies and beam axis momentum components
    Z_mu_E = event_df['mu-_E'] + event_df['mu+_E']
    Z_mu_pz = event_df['mu-_pz'] + event_df['mu+_pz']
    
    Z_e_E = event_df['e-_E'] + event_df['e+_E']
    Z_e_pz = event_df['e-_pz'] + event_df['e+_pz']
    
    # Calculate Rapidity
    Z_mu_rap = calc_rapditiy(Z_mu_E, Z_mu_pT, Z_mu_pz)
    Z_e_rap = calc_rapditiy(Z_e_E, Z_e_pT, Z_e_pz)
    
    # Add calculated values to the events DataFrame
    event_df['Z_mu_pT'] = Z_mu_pT
    event_df['Z_mu_rap'] = Z_mu_rap
    event_df['Z_e_pT'] = Z_e_pT
    event_df['Z_e_rap'] = Z_e_rap
    
    # Bring the 'signal' column to the end of the DataFrame
    event_df = DWDF.df_move_to_last_col(event_df, 'signal')
    
    # Return the events DataFrame
    return event_df


def add_reconstructed_observables(df):
    """Adds both Z Bosons observables and the reconstructed Higgs mass \
to the event DataFrame (event_df).
    
    Parameters:
     - event_df: An event DataFrame, containing the the observables 
                 (E, px, py, pz) for each of the four lepton particles.
    
    Returns:
     - event_df: An event DataFrame with the two Z Boson values
                 (Z_pT, z_rap) for both bosons (Z_mu, Z_e) and the 
                 Higg's Mass column (m_H)."""
    
    df = add_Z_bosons(df)
    df = add_higgs_mass(df)
    return df


def get_ML_observables_dataframe(collision, dataframe):
    f"""For a given collision, returns the dataframe with only the \
optimal observables.

    Parameters:
     - collision: Must be from {DW.collisions}; 
     - event_df: An event DataFrame, containing the the observables
                 (E, px, py, pz) for each of the four lepton particles.
    
    Returns:
     - event_df: An event 'ObservablesDataFrame' with only the optimal
                 observables for the given collision."""
    
    # Add all the reconstructed observables to the DataFrame
    dataframe = add_reconstructed_observables(dataframe)
    
    # Retrieve the optimal observables for this collision
    collision_observables = DW.get_collision_observables(collision)
    collision_observables += ['signal']
    
    # Return an ObservablesDataFrame with these observables
    return DWDF.ODataFrame( dataframe[collision_observables] )