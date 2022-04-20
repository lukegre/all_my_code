import all_my_code as amc


def test_get_unwrapped():
    def wrap_func(func):
        from functools import wraps

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapped_func

    func = amc.utils.camel_to_snake
    wrapped_func = wrap_func(func)
    unwrapped_func = amc.utils.unwrap(wrapped_func)
    assert func == unwrapped_func, "func and unwrapped_func should be the same"
    return None


def test_snake_to_camel():
    snake = "this_is_a_snake_case_string"
    camel = "ThisIsASnakeCaseString"
    assert amc.utils.snake_to_camel(snake) == camel


def test_camel_to_snake():
    camel = "ThisIsASnakeCaseString"
    snake = "this_is_a_snake_case_string"
    assert amc.utils.camel_to_snake(camel) == snake


def test_get_compulsory_args():
    def func(a, b, c=None, d=None):
        return None

    args = amc.utils.get_compulsory_args(func)
    assert args == ["a", "b"]


def test_append_attr():
    import xarray as xr

    ds = xr.Dataset({"a": [1, 2, 3]}, attrs={"info": "some info"})

    ds_out = amc.utils.append_attr(ds, "more info", key="info", add_meta=False)

    assert ds_out.attrs["info"] == "some info; more info", f"{ds_out.attrs['info']}"


def test_append_attr_accessor():
    import xarray as xr

    ds = xr.Dataset({"a": [1, 2, 3]})
    ds = ds.append_attrs(info="some info")
    assert ds.attrs.get("info", None) is not None


def test_get_ncfile_if_openable():
    import xarray as xr
    import tempfile

    f = tempfile.NamedTemporaryFile(suffix=".nc")
    ds = xr.Dataset({"a": [1, 2, 3]})
    ds.to_netcdf(f.name)
    loaded = amc.utils.get_ncfile_if_openable(f.name)
    assert loaded is not None
