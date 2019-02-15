### Imports ###
import numpy as np
import os
import pickle
import pandas as pd

import dwrangling.reconstruction as DWR



### Functions ###

## DataFrame Functions ##
def zero_col(DataFrame):
    """Creates a column the same length as DataFrame filled with zeros."""
    return pd.DataFrame(0, index=np.arange(len(DataFrame)), columns=['zeros'])


def pick_row_max(dataframe1, dataframe2):
    """Choses the maximum value between 2 columns.

    Parameters:
     - dataframe1: the first column.
     - dataframe2: the second column.

    Returns:
     - restult: a single column DataFrame, with the max value 
                for each row from dataframe1 or dataframe2."""

    # Join the dataframes side by side
    joined_df = pd.concat([dataframe1, dataframe2], axis=1)

    # Find the maximum of each row
    result = joined_df.max(axis=1)

    # Convert to a DataFrame and return
    return pd.DataFrame(result, columns=['max'])


def df_move_to_last_col(dataframe, column):
    """Moves the specified column to the end of the dataframe.
    
    Parameters:
     - dataframe: The DataFrame you want to reindex.
     - column: The name of the column you want to move to the end.
     
    Results:
     - dataframe: The inputted dataframe, with column at the end."""
    
    # Obtain list of column names
    col_list = list(dataframe.columns)
    
    # Check column is in the DataFrame
    if not column in col_list:
        raise TypeError("column must be the name of a column in the DataFrame." +
                        f"You entered '{column}'.")
    
    # Find the position of this column in the DataFrame
    index = col_list.index(column)
    
    # If this column is the last column, do nothing
    if index == len(col_list) - 1:
        return dataframe
    
    # Change ordering of the columns, so given column is last
    col_list = col_list[:index] + col_list[index+1:] + [column,]
    dataframe = dataframe.reindex(columns=col_list)
    
    # Return the reindex-ed DataFrame
    return dataframe


def load_to_DataFrame(events, is_signal):
    """Loads a list of events into a data frame.
    
    Parameters:
     - events: List of event objects.
     - is_signal: Either {0, 1} for background and signal, respectively.
     
    Returns:
     - events_df: A panda.DataFrame contains the events"""
    
    event_dfs = []
    for event in events:

        lepton_dfs = []
        for lepton in event.leptons:

            lepton_name = str(lepton.get_name())
            observables = [lepton_name + suffix for suffix in obsv_suffixes]

            lepton_values = [[lepton.m, lepton.E, lepton.px, lepton.py, lepton.pz,
                              lepton.perp, lepton.rap],]

            lepton_df = pd.DataFrame(lepton_values, columns=observables)
            lepton_dfs.append(lepton_df)

        event_df = pd.concat(lepton_dfs, axis=1)
        event_df['signal'] = is_signal
        event_dfs.append(event_df)
    
    events_df = pd.concat(event_dfs, ignore_index=True)
    return events_df


df_observables = ['mu-_m', 'mu-_E', 'mu-_px', 'mu-_py', 'mu-_pz', 'mu-_pT', 'mu-_rap',
                  'mu+_m', 'mu+_E', 'mu+_px', 'mu+_py', 'mu+_pz', 'mu+_pT', 'mu+_rap',
                  'e+_m', 'e+_E', 'e+_px', 'e+_py', 'e+_pz', 'e+_pT', 'e+_rap',
                  'e-_m', 'e-_E', 'e-_px', 'e-_py', 'e-_pz', 'e-_pT', 'e-_rap',
                  'Z_mu_pT', 'Z_mu_rap', 'Z_e_pT', 'Z_e_rap', 'm_H', 'signal']

## Observable Functions ##
def calc_rapditiy(p_E, p_pT, p_pz):
    """Calculates the rapidity of a particle with respect to the beam axis.
    
    Parameters:
     - p_E: The particle's energy.
     - p_pT: The particle's transverse momentum.
     - p_pz: The particle's beam axis momentum.
    
    Returns:
     - p_rap: The particle's rapidity as a pandas.Series object."""

    # Create zero pd.Series, to ensure p_eff_m2 >= 0
    zero = zero_col(p_E)['zeros']

    # Calculate the effective mass squared
    p_eff_m2 = pick_row_max(p_E**2 - (p_pT**2 + p_pz**2), zero)['max']

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
    zero = zero_col(event_df)
    # Calculate the Higg's mass for these events
    m_H = np.sqrt( pick_row_max( E**2 - (p_x**2 + p_y**2 + p_z**2), zero) )
    
    # Assign Higg's mass to penultimate column of the DataFrame
    event_df['m_H'] = m_H
    event_df = df_move_to_last_col(event_df, 'signal')
    
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
    event_df = df_move_to_last_col(event_df, 'signal')
    
    # Return the events DataFrame
    return event_df


def add_complex_observables(df):
    df = add_Z_bosons(df)
    df = add_higgs_mass(df)
    return df


def select_observables(bkg_events, sgn_events, lepton_observables):
    """lepton_observables: dict{pid: observable_fn}
        - pid: int,
        - observable_fn: fn(lepton) = return(lepton.m, lepton.rap, ...)"""
    
    bkg_data = []
    for bkg_event in bkg_events:
        observables = (lepton_observables[abs(lepton.pid)](lepton)
                       for lepton in bkg_event.leptons)
        
        datum = tuple()
        for observable in observables:
            datum = datum + observable
        
        bkg_data.append( (datum, False) )
    
    sgn_data = []
    for sgn_event in sgn_events:
        observables = (lepton_observables[abs(lepton.pid)](lepton)
                       for lepton in sgn_event.leptons)
        
        datum = tuple()
        for observable in observables:
            datum = datum + observable
        
        sgn_data.append( (datum, True) )
    
    return bkg_data, sgn_data

## Loading/Saving Functions ##
def save_object(obj, filename):
    FILE = os.path.join(os.getcwd(), "Data", filename + ".pkl")
    with open(FILE, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        
def load_object(filename):
    FILE = os.path.join(os.getcwd(), "Data", filename + ".pkl")
    with open(FILE, 'rb') as _input:
        obj = pickle.load(_input)
    return obj


def load_event_DataFrames(event):
    """Loads the events as a pandas DataFrame object.
    
    Parameters:
     - event: Event must be from ['pp_h_2mu2e', 'pp_2mu2nu'].
    
    Returns:
     - sgn_df: The signal DataFrame for the event.
     - bkg_df: The background DataFrame for the event."""
    
    events = ['pp_h_2mu2e', 'pp_2mu2nu']
    if event not in events:
        raise ValueError(f"You must enter either {events}. You entered {event}")
    
    filename = event + "_signal_DataFrame"
    FILE = os.path.join(os.getcwd(), "Data", filename + ".pkl")
    with open(FILE, 'rb') as _input:
        sgn_df = pickle.load(_input)
    
    filename = event + "_background_DataFrame"
    FILE = os.path.join(os.getcwd(), "Data", filename + ".pkl")
    with open(FILE, 'rb') as _input:
        bkg_df = pickle.load(_input)
    
    return sgn_df, bkg_df


def load_events(file):
    FILE = os.path.join(os.getcwd(), "Data", file + ".pkl")
    with open(FILE, "rb") as pklFILE:
        data = pickle.load(pklFILE)
                
    events = []
    for event in data: #Background signal
        p = []
        pdgs=[]

        init_state=[]
        leptons=[]
        hadrons=[]
        met=DWR.Reco(0,0,0,0)
        muon=DWR.Reco(0,0,0,0)
        antimuon=DWR.Reco(0,0,0,0)
        neutrino=DWR.Reco(0,0,0,0)
        antineutrino=DWR.Reco(0,0,0,0)

        for part in event['particles']:

            # Initial Particles
            if part['status'] == -1:
                tmp = DWR.Reco(part['px'], part['py'], part['pz'], part['e'])
                tmp.setpid(part['id'])
                init_state.append(tmp)

            # Final Particles
            elif part['status'] == 1:

                # Neutrinos
                if abs(part['id']) == 12 or abs(part['id']) == 14 or abs(part['id']) == 16:
                    tmp = DWR.Reco(part['px'], part['py'], part['pz'], part['e'])
                    tmp.setpid(part['id'])
                    met = DWR.addReco(met,tmp)

                    # Muon Neutrino
                    if(part['id'] == 14): 
                        neutrino = DWR.addReco(neutrino,tmp)
                        neutrino.setpid(tmp.pid)

                    # Muon Anti-Neutrino
                    if(part['id'] == -14): 
                        antineutrino = DWR.addReco(antineutrino,tmp)
                        antineutrino.setpid(tmp.pid)

                # Charged leptons
                elif abs(part['id']) == 11 or abs(part['id']) == 13 or abs(part['id']) == 15:
                    tmp = DWR.Reco(part['px'], part['py'], part['pz'], part['e'])
                    tmp.setpid(part['id'])
                    leptons.append(tmp)

                    # Muon
                    if(part['id'] == 13): 
                        muon = DWR.addReco(muon,tmp)
                        muon.setpid(tmp.pid)

                    # Anti Muon
                    if(part['id'] == -13): 
                        antimuon = DWR.addReco(antimuon,tmp)
                        antimuon.setpid(tmp.pid)

                # Hadrons (Composite of 2 or more quarks, held by strong force)
                else:
                    tmp = DWR.Reco(part['px'], part['py'], part['pz'], part['e'])
                    tmp.setpid(part['id'])
                    hadrons.append(tmp)

            # Intermediate Particles
            else:
                continue

        events.append(Event(init_state, leptons, hadrons, met, muon, antimuon, neutrino, antineutrino))
    return events



### Classes ###
class Event(object):
    def __init__(self, init_state, leptons, hadrons, met, muon, antimuon, neutrino, antineutrino):
        self._init_state = init_state
        self._leptons = leptons
        self._hadrons = hadrons
        self._met = met
        self._muon = muon
        self._antimuon = antimuon
        self._neutrino = neutrino
        self._antineutrino = antineutrino
    
    def get_pbeam(self):
        pbeam = np.zeros(3)
        
        for particle in self._init_state:
            ppart = np.array([particle.px, particle.py, particle.pz])
            pbeam += ppart
        
        return pbeam
    
    def get_phat_beam(self):
        p_beam = self.get_pbeam()
        return p_beam / np.linalg.norm(p_beam)
    
    
    def get_pT(self, beam_on_z=True):
        leptons_pT = []
        hadrons_pT = []
        
        # If beam lies on z-axis
        if beam_on_z:
            for particle in self.leptons:
                leptons_pT.append(particle.perp)
            for particle in self.hadrons:
                hadrons_pT.append(particle.perp)
        
        # If beam doesn't lie on z-axis, i.e. has been rotated
        else:
            phat_beam = self.get_phat_beam()
            for particle in self.leptons:
                leptons_pT.append(particle.setpT(phat_beam))
            for particle in self.hadrons:
                hadrons_pT.append(particle.setpT(phat_beam))
        
        return {"leptons":leptons_pT, "hadrons":hadrons_pT}
    
    @property
    def init_state(self): return self._init_state
    @property
    def leptons(self): return self._leptons
    @property
    def hadrons(self): return self._hadrons
    @property
    def met(self): return self._met
    @property
    def muon(self): return self._muon
    @property
    def antimuon(self): return self._antimuon
    @property
    def neutrino(self): return self._neutrino
    @property
    def antineutrino(self): return self._antineutrino