from pkg_resources import DistributionNotFound, get_distribution
from xarray import (
    register_dataset_accessor as _register_dataset, 
    register_dataarray_accessor as _register_dataarray)
try:
    __version__ = get_distribution("all_my_code").version
except DistributionNotFound:
    __version__ = ""
del get_distribution, DistributionNotFound


def get_unwrapped(func):
    def is_wrapped(func):
        if getattr(func, 'func', False):
            return True
        else:
            return False
        
    while is_wrapped(func):
        func = func.func
        
    return func


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
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def snake_to_camel(snake_str):
    components = snake_str.split('_')
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return ''.join([x.title() for x in components])


class add_docs_line1_to_attribute_history(object):
    def __init__(self, func):
        self.func = get_unwrapped(func)
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

    def _add_history(self, new, old=None, key='history'):
        return append_attr(
            ds=new, 
            msg=self.msg, 
            key=key, 
            old=old)


def make_xarray_accessor(class_name, func_list, accessor_type='dataarray', add_docs_line_to_history=False):
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
        og_func = get_unwrapped(func)
        @wraps(og_func)
        def dynamic_function(self, *args, **kwargs):
            da = self._obj
            return func(da, *args, **kwargs)
        return dynamic_function
    
    func_dict = {"__init__": construct}
    for func in func_list:
        unwrapped = get_unwrapped(func)
        name = unwrapped.__name__
        func_dict[name] = wrapped_function(func)
        
    # creating class dynamically
    class_name_camel = snake_to_camel(class_name)
    class_name_snake = camel_to_snake(class_name)
    Accessor = type(class_name_camel, (object,), func_dict)

    # creating objects
    if 'array' in accessor_type:
        register_dataarray_accessor(class_name_snake)(Accessor)
    elif 'set' in accessor_type:
        register_dataset_accessor(class_name_snake)(Accessor)
    elif 'both' in accessor_type:
        register_dataarray_accessor(class_name_snake)(Accessor)
        register_dataset_accessor(class_name_snake)(Accessor)


def append_attr(ds, msg, key='history', old=None, func=None):
    from pandas import Timestamp
    import inspect

    if func is None:
        func = inspect.stack()[1][3]
        func = f": {func}" if func != "__call__" else ""
    if callable(func):
        func = func.__name__

    version = f".{__version__}" if __version__ else ""
    version = version.split('+')[0]
    
    now = Timestamp.today().strftime("%Y-%m-%d")
    msg = f"[all_my_code{version}@{now}{func}] {msg}"

    if old is None:
        old = ds
    hist = old.attrs.get(key, '')
    if hist != '':
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
            print('EXISTS:', sname)
            return out
        except:
            pass


@_register_dataset('append_attrs')
@_register_dataarray('append_attrs')
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