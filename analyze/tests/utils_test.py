import pytest
import numpy
import numpy as np
import os

from analyze.analyzer import Analyzer

# list of all builtin types in python (3)
import builtins
builtin_types = [getattr(builtins, d) for d in dir(builtins) if isinstance(getattr(builtins, d), type)]
builtin_types.append(numpy.ndarray)


def approx_equal_dicts(d, d_ref, keys=None, rel=None, abs=None):
    """
    Wrapper to pytest.assert() function which is able to apply approx to elements in nested dictionaries.
    Note that no type checking is done, and the elements in the dictionary are assumed to be comply with pytest.assert()
    allowed inputs.

    Parameters
    ----------
    d : dict
        Dictionary to be approximated.

    d_ref : dict
        Reference dictionary, to which d will be

    keys : list, optional
        List of dictionary keys to be approximated. If None all keys of d_ref are used.

    rel : float, optional
        Relative tolerance, as in pytest.assert().

    abs: float, optional
        Absolute tolerance, as in pytest.assert().

    Returns
    ----------
    True if all keys in d are approximately equal to those of d_ref, False otherwise.

    """

    if keys is None:
        keys = d_ref.keys()

    conditions = []

    # compare keys
    keys1 = set(d_ref.keys())
    keys2 = set(d.keys())
    keys_dif = keys1.difference(keys2)
    is_equal = True if len(keys_dif) == 0 else False
    conditions.append(is_equal)

    try:

        for key in keys:

            val_ref = d_ref[key]
            val = d[key]

            if isinstance(val_ref, dict):
                conditions.append(approx_equal_dicts(val, val_ref, rel=rel, abs=abs))
            elif isinstance(val_ref, str) or isinstance(val_ref, bool):
                conditions.append(val == val_ref)
            elif isinstance(val_ref, list):  # assume list of non numeric values
                set1 = set(val_ref)
                set2 = set(val)
                set_dif = set1.difference(set2)
                conditions.append(len(set_dif) == 0)
            elif type(val_ref).__module__ == np.__name__:  # numpy variable
                conditions.append(val == pytest.approx(val_ref, rel=rel, abs=abs))
            elif isinstance(val_ref, object) and (type(val_ref) not in builtin_types):  # custom class
                conditions.append(compare_instances(val, val_ref))
            else:  # assume numeric type
                conditions.append(val == pytest.approx(val_ref, rel=rel, abs=abs))

        return all(conditions)

    except: # e.g. if d or d_ref does not contain key

        return False


def compare_instances(instance1, instance2):

    is_equal_list = []

    # if instances are None
    if (instance1 is None) and (instance2 is None):
        return True
    elif ((instance1 is None) and (instance2 is not None)) or \
            ((instance1 is not None) and (instance2 is None)):
        return False

    # compare attribute names
    atts1 = set(instance1.__dict__.keys())
    atts2 = set(instance2.__dict__.keys())
    atts_dif = atts1.difference(atts2)
    is_equal = True if len(atts_dif) == 0 else False
    is_equal_list.append(is_equal)

    # compare representations
    repr1 = str(instance1)
    repr2 = str(instance2)
    is_equal = repr1 == repr2
    is_equal_list.append(is_equal)

    # compare attributes
    for key in atts1:

        val1 = getattr(instance1, key)
        val2 = getattr(instance2, key)

        if isinstance(val1, dict):  # dict
            is_equal = approx_equal_dicts(val1, val2, rel=0.0001)
        elif isinstance(val1, list):  # assume list of non numeric type
            set1 = set(val1)
            set2 = set(val2)
            set_dif = set1.difference(set2)
            is_equal = len(set_dif) == 0
        elif isinstance(val1, numpy.ndarray):
            is_equal = np.all(val1 == pytest.approx(val2))
        elif type(val1) in builtin_types:  # any type of python builtin, excluding dict, list
            is_equal = val1 == val2
        elif (val1 is None) and (val2 is None):  # None
            is_equal = True
        elif isinstance(val1, object) and (type(val1) not in builtin_types):  # custom class
            is_equal = compare_instances(val1, val2)
        else:
            try:
                is_equal = val1 == val2  # try simple comparison
            except:
                is_equal = False

        is_equal_list.append(is_equal)

    is_equal = all(is_equal_list)

    return is_equal


def load_test_analyzer(data_dir='ILSVRC2015_00078000'):
    """
    Load analyzer object from saved test data.

    Parameters
    ----------
    data_dir : str, optional
        Data directory name, should be name of one of the folders inside analyze/tests/data/.

    Returns
    -------
    anaylzer : Analyzer
        Analyzer object.
    analyzer_root_dir : str
        Path of analyzer root directory, such that full images path is the concatenation of analyzer_root_dir and
        analyzer.data image_path.
    """

    # load reference analyzer
    base_dir = os.path.dirname(__file__)
    relative_data_dir = os.path.join('data', data_dir)
    data_dir = os.path.join(base_dir, relative_data_dir)
    analyzer_file = os.path.join(data_dir, 'analyzer.p')
    analyzer = Analyzer.load(analyzer_file, load_images_from_dir=False)
    analyzer_root_dir = os.path.join(base_dir, '..')

    return analyzer, analyzer_root_dir

def load_analyzer(data_dir, load_images_from_dir=False):

    # load reference analyzer
    analyzer_file = os.path.join(data_dir, 'analyzer.p')
    analyzer = Analyzer.load(analyzer_file, load_images_from_dir=False)
    analyzer_root_dir = os.path.dirname(analyzer_file)

    return analyzer, analyzer_root_dir