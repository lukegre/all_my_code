from ..utils import make_xarray_accessor as _make_xarray_accessor
from . import detect, stats

_make_xarray_accessor(
    "extreme",
    [
        detect.poly_baseline,
        detect.fixed_baseline,
        stats.event_based_stats_2d_agg,
        stats.duration,
        stats.severity,
        stats.n_events,
    ],
)
