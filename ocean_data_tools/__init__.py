from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution('glidertools').version
except DistributionNotFound:
    __version__ = 'version_undefined'
del get_distribution, DistributionNotFound
