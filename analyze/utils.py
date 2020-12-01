

def assign_dict_recursively(dict_ref, dict_to_update):
    """
    Assign values to non existing dictionary keys recursively.

    Parameters
    ----------
    dict_ref : dict
        Reference dictionary, contains all possible keys with default values.

    dict_to_update : dict
        Dictionary to be updated.
        Exsiting keys' values will not be changed.
        Defaults values will be assign to missing keys.

    Returns
    ----------
        dict_to_update : dict
            Updated dictionary
    """

    # assign default values to missing keys
    for key in dict_ref:
        if key not in dict_to_update:  # if key does not exist in dictToUpdate
            dict_to_update[key] = dict_ref[key]  # assign correponding dictRef value
        elif isinstance(dict_ref[key],
                        dict):  # if key does exist in dictToUpdate - check if corresponding value is also dict
            dict_to_update[key] = assign_dict_recursively(dict_ref[key],
                                                          dict_to_update[key])  # assign nested dict recursively
        # else: key exist and corresponding value is not dict - do nothing

    return dict_to_update