from pkg_resources import DistributionNotFound, get_distribution
try:
    __version__ = get_distribution("all_my_code").version
except DistributionNotFound:
    __version__ = ""
del get_distribution, DistributionNotFound


class add_docs_line1_to_attribute_history(object):
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        docs = func.__doc__
        self.msg = docs.strip().split("\n")[0] if isinstance(docs, str) else ""

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            try:
                out = self._add_history(self.func(*args, **kwargs))
                return out
            except Exception as e:
                raise e
                return args[0]

        self.kwargs = kwargs
        return self.__caller__

    def __caller__(self, ds):
        return self._add_history(self.func(ds, **self.kwargs))

    def _add_history(self, ds, key='history'):
        from pandas import Timestamp

        version = ".{__version__}" if __version__ else ""
        
        now = Timestamp.today().strftime("%y%m%d")
        prefix = f"[amc{version}@{now}] "
        msg = prefix + self.msg
        
        hist = ds.attrs.get(key, '')
        if hist != '':
            hist = hist.split(";")
            hist = [h.strip() for h in hist]
            msg = "; ".join(hist + [msg])
            
        ds = ds.assign_attrs({key: msg})

        return ds