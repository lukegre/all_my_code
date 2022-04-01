
def guess_coords_from_column_names(
    column_names, 
    match_dict=dict(
        time=["month", "time", "t", "date"],
        depth=["depth", "z"],
        lat=["lat", "latitude", "y"], 
        lon=["lon", "longitude", "x"],)
):
    """
    Takes a list of column names and guesses 
    """
    
    coord_names = {}
    for col in column_names:
        est_name = estimate_name(col, match_dict)
        if est_name != col:
            coord_names[col] = est_name

    coord_names = drop_worst_duplicates_from_rename_dict(coord_names)
    return coord_names


def drop_worst_duplicates_from_rename_dict(rename_dict):
    """
    Will remove the weakest matching key-value pair where the value is duplicate. 
    
    Parameters
    ----------
    rename_dict: dict 
        keys are the original values and keys are the new names. 
        
    Returns
    -------
    dict: the same dictionary with the weakest matching duplicates removed
        
    """
    import pandas as pd
    
    names = pd.Series(rename_dict)
    duplicates = names.duplicated()
    duplicated_coords = names[duplicates].unique()
    
    drop_duplicates = []
    for duplicate_name in duplicated_coords:
        original_columns = names[names == duplicate_name].index.tolist()
        
        ratios = fuzzy_matching(duplicate_name, original_columns).mean(axis=1)
        best_match = ratios.idxmax()
        
        original_columns.remove(best_match)
        drop_duplicates += original_columns

    names_wo_duplicates = names.drop(drop_duplicates)
    
    return dict(names_wo_duplicates)


def estimate_name(name, match_dict):
    """
    Gets the closest match for the name from the match dictionary.
    
    Uses FuzzyWuzzy library to find the nearest match for values in 
    the match_dict. They key of the nearest match will be assigned as
    the new name. 
    
    Parameters
    ----------
    name: str
        A string name that you'd like to find a match with 
    match_dict: dict
        Keys are the new name. Values can be list or string and are
        the possible near matches. 
    threshold: int [75]
        A new name will not be assigned if the ratio does not exceed this
        value
        
    Returns
    -------
    str: 
        either the original name if no strong matches, or a key from the 
        match_dict for the best matching value pair. 
    
    """

    best_ratio = 0
    best_name = ""
    for key in match_dict:
        ratios = fuzzy_matching(name, match_dict[key]).mean(axis=1)
        if ratios.max() > best_ratio:
            best_name = key
            best_ratio = ratios.max()
            
    if best_ratio > 75:
        return best_name
    else:
        return name
    

def fuzzy_matching(s, possible_matches):
    """
    Does fuzzy matching of a string with a list of possible strings
    
    Paramters
    ---------
    s: str
        The string you'd like to find the closest match with
    possible_matches: list
        A list of strings that could be a match for s
        
    Returns
    -------
    pd.dataframe
        fuzzy match ratios with partial_ratios and ratios where
        0 is the min and 100 is the max. The columns are the two
        types of matching algos and the rows are the entries from
        the possible matches. 
        
    Note
    ----
    This is a wrapper around fuzz_ratios that does case insensitive 
    matching. 
    """
    import pandas as pd
    from fuzzywuzzy import fuzz
    ratios = {}
    for func in [fuzz.partial_ratio, fuzz.ratio]:
        name = func.__name__
        ratios[name] = fuzz_ratios(s, possible_matches, func)
    return pd.DataFrame(ratios)   
    
    
def fuzz_ratios(s, possible_matches, func=None):
    """
    Does fuzzy matching of a string with a list of strings
    
    Parameters
    ----------
    s: str
        the string you'd like to match with
    possible_matches: list
        a list of strings that could match with s
    func: [None|callable]
        if None, then will default to fuzzywuzzy.fuzz.partial_ratio
        accepts other fuzzywuzzy functions that return ratioss

    Returns
    -------
    dict: 
        ratios for each of the possible_matches entries
    """
    if func is None:
        from fuzzywuzzy.fuzz import partial_ratio as func
    
    x = s.lower()
    if isinstance(possible_matches, list):
        y = possible_matches
    if isinstance(possible_matches, str):
        y = [possible_matches]
    
    ratios = {m: func(m.lower(), x) for m in y}
    return ratios
