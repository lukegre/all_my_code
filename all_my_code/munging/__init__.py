from . import conform
from . import date_utils
from . import colocation
from . import units
from . import sparse
from functools import wraps as _wraps
from ..utils import (
    make_xarray_accessor as _make_xarray_accessor, 
    get_unwrapped, 
    add_docs_line1_to_attribute_history)

from xarray import (
    register_dataset_accessor as _register_dataset, 
    register_dataarray_accessor as _register_dataarray)


from .colocation import colocate_dataarray
from .grid import (
    lon_180W_180E, 
    lon_0E_360E, 
    coord_05_offset, 
    interp_bilinear,
    interp)
from .conform import (
    transpose_dims,
    correct_coord_names,
    time_center_monthly,
    drop_0d_coords,
    rename_vars_snake_case,
    apply_process_pipeline,
)
    

_make_xarray_accessor(
    "grid",
    [
        lon_180W_180E,
        lon_0E_360E,
        coord_05_offset,
        colocate_dataarray,
        interp,
        interp_bilinear,
    ],
    accessor_type='both'
)


_func_registry = [
    lon_0E_360E,
    lon_180W_180E,
    coord_05_offset,
    transpose_dims,
    correct_coord_names,
    rename_vars_snake_case,
    time_center_monthly,
    drop_0d_coords,
]

_default_conform = [
    correct_coord_names,
    transpose_dims,
    lon_180W_180E,
    time_center_monthly,
    rename_vars_snake_case,
]


@_register_dataset("conform")
@_register_dataarray("conform")
class DataConform(object):
    """
    A class to conform a dataset/dataarray to a the desired conventions

    Modules (subfunctions) can be used to conform the dataset/dataarray 
    individually, or you can call this function to apply a set of standard 
    functions. 
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

        for func in _func_registry:
            func = add_docs_line1_to_attribute_history(func)
            setattr(self, get_unwrapped(func).__name__, self._make_accessor_func(func))

    def _make_accessor_func(self, func):
        @_wraps(get_unwrapped(func))
        def run_func(*args, **kwargs):
            return func(self._obj, *args, **kwargs)

        return run_func

    def __call__(
        self, 
        coord_names=True,
        time_centered=False,
        squeeze=True,
        transpose=True,
        lon_180W=True,
        standardize_var_names=False,
    ):
        da = self._obj

        funclist = []
        if coord_names:
            funclist.append(correct_coord_names)
        if time_centered:
            funclist.append(time_center_monthly)
        if squeeze:
            funclist.append(drop_0d_coords)
        if transpose:
            funclist.append(transpose_dims)
        if lon_180W:
            funclist.append(lon_180W_180E)
        if standardize_var_names:
            funclist.append(rename_vars_snake_case)

        funclist = [add_docs_line1_to_attribute_history(f) for f in funclist]
        out = apply_process_pipeline(da, *funclist)

        return out

