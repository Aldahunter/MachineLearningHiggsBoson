"""Data Wrangling - Cleaning, Munging, Mapping and Pipelines."""

### Functions ###
def split_data(data, train_frac = 0.67):
    # The data must be a pandas.DataFrame.
    train_data, test_data = [], []

    # Iterate through each row in data.
    for _, datum in data.iterrows():
        
        # If unform random number between [0, 1] is less than,
        # or equal to, the the train_frac.
        if random() <= train_frac:
            # Add row to train_data.
            train_data.append(datum)
        else:
            # If greater than train_frac, add to test_data.
            test_data.append(datum)
    
    # Return dictionary of train_data and test_data as panda.DataFrames.
    return {'train': pd.DataFrame(train_data).reset_index(drop=True),
            'test': pd.DataFrame(test_data).reset_index(drop=True)}


def df_to_ML_input(df):
    
    # Obtain list of observables.
    input_observables = list(df.columns.values)
   
    # Remove the signal column from observables.
    input_observables.remove('signal')
    
    # Obtain numpy arrays for obervables and signal (labels).
    observables = df[input_observables].values
    labels = df.signal.values
    
    # Return the observables and labels arrays.
    return observables, labels