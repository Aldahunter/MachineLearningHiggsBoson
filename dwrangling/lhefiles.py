"""LHE Files - The functions associated with reading the data from the \
.lhe files."""

import csv
import os
import pylhe

import dwrangling as DW
import dwrangling.reconstruction as DWR
import dwrangling.events as DWE
import dwrangling.dataframes as DWDF



### LHE Reading Functions ###
bs = '\\'

def load_lhe_to_generators(collision):
    f"""Loads the LHE file events into a generator.
    
    Parameters:
     - collision: Must be from {DW.collisions};
    
    Returns:
     - generator: A generator containing the LHE file's events."""
    
    # Parse collision and obtain file locations.
    lhe_files = DW.get_data_file_paths(collision, '.lhe', 'lhe')
    
    # Check LHE files exist
    if not os.path.isfile(lhe_files[0]):
        raise IOError(f"No such file or directory: '{lhe_files[0]}'")
    if not os.path.isfile(lhe_files[1]):
        raise IOError(f"No such file or directory: '{lhe_files[1]}'")
    
    # Return LHE files as generators
    return (pylhe.readLHE(lhe_file[0]), pylhe.readLHE(lhe_file[1]))


def lhe_to_csv_files(collision):
    f"""Ports the LHE file to a CSV file.
    
    Parameters:
     - collision: Must be from {DW.collisions};
    
    Returns:
     - None: If successful, 'False' otherwise."""
    
    # Parse collision and obtain file locations.
    csv_files = DW.get_data_file_paths(collision, '.csv', 'csv')
    
    # Check user doesnt accidently overwrite files
    if os.path.isfile(csv_files[0]) or os.path.isfile(csv_files[1]):
        overwrite = DW.y_n_input("The csv files already exist, do you " +
                                 "want to overwrite them? (y/n): ")
        # If user doesnt want to overwrite return False.
        if overwrite is 'n':
            return False
    
    # Load LHE files as generators
    lhe_generators = load_lhe_to_generators(collision)
    
    # Loop through process for signal and background files.
    for lhe_generator, csv_file in zip(lhe_generators, csv_files):
        with open(csv_file, "wb") as csvFILE:
            # Open a csv file (with the same name as an lhe file) for writing

            # Create object to write dictionaries to csv
            writer = csv.DictWriter(csvFILE, ["particles", "eventinfo"])

            # Write dict keys as table headers
            writer.writeheader()

            # Write events from LHE generator to csv files
            writer.writerows(event for event in lhe_generator)

        # Inform user of progress
        print(f"Successfully ported to '{csv_file.split(bs)[-1]}'")


def lhe_to_events(collision):
    f"""Converts LHE files to lists of type 'Events'.
    
    Parameters:
     - collision: Must be from {DW.collisions};
    
    Returns:
     - events: As  (signal_events, background_events), where each is
               a list of type 'Events'."""
    
    # Load LHE files as generators
    lhe_generators = load_lhe_to_generators(collision)
    
    # Loop through process for signal and background files.
    events = [[], [],]
    data_name = ['signal', 'background']
    for s_b, lhe_generator in zip([0, 1], lhe_generators):

        # Iterate and parse each event in the LHE generator
        for event in lhe_generator:
            events[s_b].append( DWE.Event.from_lhe_event(event) )
        
        # Inform user of progress
        print(f"Successfully parsed {data_name[s_b]} to Events.")

    return tuple(events)


def lhe_event_to_observable_list(lhe_event, is_signal):
    """Converts an LHE file's event to a list of observables.
    
    Parameters:
     - lhe_event: An iteration from an LHE file,
     - is_signal: A bool, labeling this event as signal 'True'
                  or background 'False'.
    
    Returns:
     - list: A list of this event's observables."""
    
    if int(is_signal) not in [0, 1]:
        raise ValueError("Parameter 'is_signal' must be boolean, " +
                         f"not '{is_signal}'.")

    values = []
    for particle in lhe_event['particles']:

        # Skip particles which are not final state particles
        if particle['status'] != 1:
            continue

        temp = DWR.Reco.from_dict(particle)

        # Skip particles which cannot be observed
        if temp.is_neutrino():
            continue

        # Obtain particle's observable values
        values += [temp.m, temp.E, temp.px, temp.py,
                   temp.pz, temp.perp, temp.rap]

    # Add signal column to end
    values += [int(is_signal),]

    return values

def lhe_event_to_observable_column_names(lhe_event):
    """Converts an LHE file's event to a list of observable names.
    
    Parameters:
     - lhe_event: An iteration from an LHE file.
    
    Returns:
     - list: A list of str's corresponding to the observable names."""
    
    obsv_suffixes = ['_m', '_E', '_px', '_py', '_pz', '_pT', '_rap']

    columns = []
    for particle in lhe_event['particles']:

        # Skip particles which are not final state particles
        if particle['status'] != 1:
            continue

        temp = DWR.Reco.from_dict(particle)

        # Skip particles which cannot be observed
        if temp.is_neutrino():
            continue

        # Obtain column names for particle's observables
        temp_name = str(temp.get_name())
        columns += [temp_name + suffix for suffix in obsv_suffixes]

    # Add signal column to end
    columns += ['signal',]

    return columns


def lhe_to_dataframes(collision):
    f"""Converts LHE files to an 'ObservablesDataFrame'.
    
    Parameters:
     - collision: Must be from {DW.collisions};
    
    Returns:
     - ObservablesDataFrame: As dataframe containing all the observables
                             of each event from the LHE file."""
    
    # Load LHE files as generators
    lhe_generators = load_lhe_to_generators(collision)

    # Loop through process for signal and background files.
    dfs = {}
    data_name = {'s': 'signal', 'b', 'background'}
    for s_b, is_signal, lhe_generator in zip(['s', 'b'], [True, False], lhe_generators):

        # Get values & column names from first item in LHE generator
        lhe_event = next(lhe_generator)
        data = [lhe_event_to_observable_list(lhe_event, is_signal),]
        columns = lhe_event_to_observable_column_names(lhe_event)
        
        # Iterate and parse rest of the events in the LHE generator
        for lhe_event in lhe_generator:
            data += [lhe_event_to_observable_list(lhe_event, is_signal),]

        # Convert to ObservablesDataFrame
        dfs[s_b] = DWDF.ObservablesDataFrame(data, columns=columns)
        
        # Inform user of progress
        print(f"Successfully parsed {data_name[s_b]} to ObservablesDataFrame.")

    return dfs