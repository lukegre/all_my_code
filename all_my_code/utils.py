from pkg_resources import DistributionNotFound, get_distribution
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


class add_docs_line1_to_attribute_history(object):
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        docs = func.__doc__
        self.msg = docs.strip().split("\n")[0] if isinstance(docs, str) else ""

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            try:
                out = self._add_history(self.func(*args, **kwargs), args[0], kwargs)
                return out
            except AttributeError as e:
                print(e)
                return self.func(*args, **kwargs)
            except Exception as e:
                raise e
                return args[0]

        self.kwargs = kwargs
        return self.__caller__

    def __caller__(self, ds):
        return self._add_history(self.func(ds, **self.kwargs))

    def _add_history(self, ds, old, kwargs, key='history'):
        from pandas import Timestamp

        version = f".{__version__}" if __version__ else ""
        version = version.split('+')[0]
        
        now = Timestamp.today().strftime("%y%m%d")
        prefix = f"[all_my_code{version}@{now}] "

        dim = kwargs['dim'] if 'dim' in kwargs else 'time' 
        msg = prefix + self.msg.format(dim=dim)
        
        hist = old.attrs.get(key, '')
        if hist != '':
            hist = hist.split(";")
            hist = [h.strip() for h in hist]
            msg = "; ".join(hist + [msg])
            
        ds = ds.assign_attrs({key: msg})

        return ds