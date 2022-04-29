from inspect import unwrap

from pkg_resources import DistributionNotFound, get_distribution
from xarray import register_dataarray_accessor as _register_dataarray
from xarray import register_dataset_accessor as _register_dataset

try:
    __version__ = get_distribution("all_my_code").version
except DistributionNotFound:
    __version__ = ""
del get_distribution, DistributionNotFound


def run_parallel(func, args_list, kwargs={}, n_jobs=12, **joblib_kwargs):
    """
    Run a function in parallel.

    Parameters
    ----------
    func : function
        Function that you want to apply to the arguments in args_list
    args_list: list
        List of arguments to apply the function to
    kwargs: dict
        keyword arguments that will be passed to func
    n_jobs: int
        number of parallel jobs

    Returns
    -------
    tuple : the output from each func run
    """
    import joblib

    pool = joblib.Parallel(n_jobs=n_jobs, **joblib_kwargs)
    func = joblib.delayed(func)
    queue = [func(arg, **kwargs) for arg in args_list]
    out = pool(queue)
    return out


def camel_to_snake(name):
    import re

    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def snake_to_camel(snake_str):
    components = snake_str.split("_")
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return "".join([x.title() for x in components])


def xarray_dataset_to_column_input(func):
    """
    Decorator to convert a dataset to a column input for a function.
    Names will also be guessed and reported to the user
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        from inspect import signature
        from xarray import Dataset

        name = func.__name__
        only_one_arg = len(args) == 1
        is_dataset = isinstance(args[0], Dataset)

        if only_one_arg and is_dataset:
            ds = args[0]
            coords = list(ds.coords)
            vars = list(ds.data_vars)
            renames = match_func_args_and_input_vars(func, coords + vars)
            print("The following columns will be used as input for the function: ")
            print(name + str(signature(func)))
            for k, v in renames.items():
                print(f"  {v}=ds.{k}")

            # list only variables that are required from the dataset
            vars = [k for k in renames if k in ds.data_vars]
            # convert only required variables to dataframe columns
            df = ds[vars].to_dataframe()
            # rename the columns to match the function arguments (including coords)
            df_input = df.reset_index().rename(columns=renames)
            # convert the dataframe to a dictionary
            input_args = {k: df_input[k] for k in renames.values()}
            # run actual function
            df[name] = func(**input_args, **kwargs)
            # convert the ouput column to a DataArray
            da = df[name].to_xarray()

            return da
        else:
            return func(*args, **kwargs)

    wrapper.__doc__ = (
        "This function has been wrapped so that array-like inputs can be \n"
        "the variable names of a dataset that will be flattened. \n"
        "Variable names will be approximated. An error will be raised \n"
        "if no best match is found\n"
        f"{'='*60}"
        f"{func.__doc__}"
    )

    return wrapper


def match_func_args_and_input_vars(func, input_vars, threshold=70):
    """
    Tries to create a dictionary of the input arguments of a function that
    match with the variables in a list of variables

    Parameters
    ----------
    func : function
        Function from which the compulsory arguments are extracted
    input_vars : list
        List of variables to match the arguments to
    threshold : int
        Threshold for the similarity between the argument and input_vars

    Returns
    -------
    dict : dictionary of the matched variables
    """
    from .munging.name_matching import fuzzy_matching
    from pandas import Series

    def matching(func_args, keys):
        keys = list(keys)
        matches = {}
        for key in func_args:
            match_ratios = fuzzy_matching(key, keys)
            best_match_ratios = match_ratios.max().values
            best_match_keys = match_ratios.idxmax().values
            best_match = Series(best_match_ratios, index=best_match_keys)
            matches[key] = best_match
        return matches

    func_args = get_compulsory_args(func)
    matches = matching(func_args, input_vars)

    renames = {}
    for func_key in matches:
        ser = matches[func_key].rename(f"Matching Ratios (> {threshold} is good)")
        good_match = ser > threshold
        assert good_match.any(), f"No good match was found for {func_key}\n{ser}"
        var_names = ser[good_match].index.unique()
        assert var_names.size == 1, (
            f"Two matches found for {func_key}\n{ser}\n"
            f"Please rename your columns to match `{func_key}`"
        )
        var_key = var_names[0]
        renames[var_key] = func_key

    return renames


def get_compulsory_args(func):
    """
    Get the arguments that are compulsory for a function.

    Compulsory arguments are those that are required to be passed
    to the function that do not have default values (i.e. key=value).

    Parameters
    ----------
    func : function
        Function to get the compulsory arguments for

    Returns
    -------
    list : list of compulsory argument names
    """
    from inspect import signature, _empty as empty

    sig = signature(func)
    args = []
    for p in sig.parameters.values():
        positional = p.kind == p.POSITIONAL_OR_KEYWORD
        compulsory = p.default == empty
        if positional and compulsory:
            args.append(p.name)
    return args


class add_docs_line1_to_attribute_history(object):
    def __init__(self, func):
        self.func = unwrap(func)
        self.name = func.__name__
        docs = func.__doc__
        self.msg = docs.strip().split("\n")[0] if isinstance(docs, str) else ""

    def __call__(self, *args, **kwargs):
        from xarray import DataArray, Dataset

        try:
            out = self.func(*args, **kwargs)
            # if the output is not a data array then we dont add history
            if not isinstance(out, (DataArray, Dataset)):
                return out
            # if input and output are the same, then don't add history
            elif out.equals(args[0]):
                return out
            # in all other situations we add history
            else:
                return append_attr(out, self.msg, old=args[0])
            return out
        except Exception as e:
            raise e

    def __caller__(self, ds):
        return self._add_history(self.func(ds, *self.args[1:], **self.kwargs))

    def _add_history(self, new, old=None, key="history"):
        return append_attr(ds=new, msg=self.msg, key=key, old=old)


def make_xarray_accessor(
    class_name, func_list, accessor_type="dataarray", add_docs_line_to_history=False
):
    """
    Turns a list of functions into an xarray accessor.

    Parameters
    ----------
    class_name : str
        Name of the class that the accessor will be attached to.
        Will be converted to snake_case format for the accessor name.
        The class name will be converted to CamelCase.
    func_list : list
        List of functions to be attached to the accessor. Note that
        the first input of each function must be a dataset/dataarray.
    accessor_type : str
        Type of accessor to be created. Cane be 'dataarray' or 'dataset' or 'both'
    add_docs_line_to_history : bool
        If True, the first line of the docstring of each function will be
        added to the history attribute.

    Returns
    -------
    None
    """
    from xarray import register_dataarray_accessor, register_dataset_accessor

    def construct(self, da):
        self._obj = da

    def wrapped_function(func):
        from functools import wraps

        if add_docs_line_to_history:
            func = add_docs_line1_to_attribute_history(func)
        og_func = unwrap(func)

        @wraps(og_func)
        def dynamic_function(self, *args, **kwargs):
            da = self._obj
            return func(da, *args, **kwargs)

        return dynamic_function

    func_dict = {"__init__": construct}
    for func in func_list:
        unwrapped = unwrap(func)
        name = unwrapped.__name__
        func_dict[name] = wrapped_function(func)

    # creating class dynamically
    class_name_camel = snake_to_camel(class_name)
    class_name_snake = camel_to_snake(class_name)
    Accessor = type(class_name_camel, (object,), func_dict)

    # creating objects
    if "array" in accessor_type:
        register_dataarray_accessor(class_name_snake)(Accessor)
    elif "set" in accessor_type:
        register_dataset_accessor(class_name_snake)(Accessor)
    elif "both" in accessor_type:
        register_dataarray_accessor(class_name_snake)(Accessor)
        register_dataset_accessor(class_name_snake)(Accessor)


def apply_to_dataset(func):
    from functools import wraps
    from xarray import DataArray

    @wraps(func)
    def wrapper(ds, *args, **kwargs):
        if isinstance(ds, DataArray):
            return func(ds, *args, **kwargs)
        else:
            return ds.map(func, args=args, **kwargs)

    return wrapper


def append_attr(ds, msg, key="history", add_meta=True, old=None, func=None):
    from pandas import Timestamp
    import inspect

    if func is None:
        func = inspect.stack()[1][3]
        func = f": {func}" if func != "__call__" else ""
    if callable(func):
        func = func.__name__

    version = f".{__version__}" if __version__ else ""
    version = version.split("+")[0]

    now = Timestamp.today().strftime("%Y-%m-%d")
    if add_meta:
        msg = f"[all_my_code{version}@{now}{func}] {msg}"

    if old is None:
        old = ds
    hist = old.attrs.get(key, "")
    if hist != "":
        hist = hist.split(";")
        hist = [h.strip() for h in hist]
        msg = "; ".join(hist + [msg])

    new = ds.assign_attrs({key: msg})

    return new


def get_ncfile_if_openable(sname):
    import os
    import xarray as xr

    if sname is None:
        return None
    if os.path.isfile(sname):
        try:
            out = xr.open_dataset(sname)
            print("EXISTS:", sname)
            return out
        except:
            pass


@_register_dataset("append_attrs")
@_register_dataarray("append_attrs")
class AttrAdder(object):
    def __init__(self, ds):
        self._obj = ds

    def __call__(self, func=None, **kwargs):
        import inspect

        if func is None:
            func = inspect.stack()[1][3]
            func = f": {func}" if func != "__call__" else ""
        if len(kwargs) > 1:
            key = list(kwargs)[0]
            msg = kwargs[key]
            return append_attr(self._obj, msg, key=key, func=func)
        else:
            ds = self._obj
            for key in kwargs:
                msg = kwargs[key]
                ds = append_attr(ds, msg, key=key, func=func)
            return ds
