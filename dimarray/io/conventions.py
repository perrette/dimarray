"""Adapted from xray.convention, but stripped from pandas call and other xray objects
"""
import functools
import numpy as np
import warnings
from collections import defaultdict
from datetime import datetime

# standard calendars recognized by netcdftime
_STANDARD_CALENDARS = set(['standard', 'gregorian', 'proleptic_gregorian'])


def _infer_time_units_from_diff(unique_timedeltas):
    for time_unit, delta in [('days', 86400), ('hours', 3600),
                             ('minutes', 60), ('seconds', 1)]:
        unit_delta = np.timedelta64(10 ** 9 * delta, 'ns')
        diffs = unique_timedeltas / unit_delta
        if np.all(diffs == diffs.astype(int)):
            return time_unit
    raise ValueError('could not automatically determine time units')


def infer_datetime_units(dates):
    """Given an array of datetimes, returns a CF compatible time-unit string of
    the form "{time_unit} since {date[0]}", where `time_unit` is 'days',
    'hours', 'minutes' or 'seconds' (the first one that can evenly divide all
    unique time deltas in `dates`)
    """
    # dates = pd.to_datetime(dates, box=False)
    unique_timedeltas = np.unique(np.diff(dates[dates != np.datetime64('NaT')]))
    units = _infer_time_units_from_diff(unique_timedeltas)
    return '%s since %s' % (units, str(dates[0]))

def infer_timedelta_units(deltas):
    """Given an array of timedeltas, returns a CF compatible time-unit from
    {'days', 'hours', 'minutes' 'seconds'} (the first one that can evenly
    divide all unique time deltas in `deltas`)
    """
    unique_timedeltas = np.unique(deltas[deltas != np.timedeltas('NaT')])
    units = _infer_time_units_from_diff(unique_timedeltas)
    return units

def decode_cf_datetime(num_dates, units, calendar=None):
    """Given an array of numeric dates in netCDF format, convert it into a
    numpy array of date time objects.

    For standard (Gregorian) calendars, this function uses vectorized
    operations, which makes it much faster than netCDF4.num2date. In such a
    case, the returned array will be of type np.datetime64.

    See also
    --------
    netCDF4.num2date
    """
    import netCDF4 as nc4
    num_dates = np.asarray(num_dates).astype(float)
    if calendar is None:
        calendar = 'standard'

    def nan_safe_num2date(num):
        return np.datetime64('NaT') if np.isnan(num) else nc4.num2date(num, units, calendar)

    min_num = np.nanmin(num_dates)
    max_num = np.nanmax(num_dates)
    min_date = nan_safe_num2date(min_num)
    if num_dates.size > 1:
        max_date = nan_safe_num2date(max_num)
    else:
        max_date = min_date

    if ((calendar not in _STANDARD_CALENDARS
            or min_date.year < 1678 or max_date.year >= 2262)
            and min_date != np.datetime64('NaT')):

        dates = nc4.num2date(num_dates, units, calendar)

        if min_date.year >= 1678 and max_date.year < 2262:
            try:
                dates = nctime_to_nptime(dates)
            except ValueError as e:
                warnings.warn('Unable to decode time axis into full '
                              'numpy.datetime64 objects, continuing using '
                              'dummy netCDF4.datetime objects instead, reason:'
                              '{0}'.format(e), RuntimeWarning, stacklevel=2)
                dates = np.asarray(dates)
        else:
            warnings.warn('Unable to decode time axis into full '
                          'numpy.datetime64 objects, continuing using dummy '
                          'netCDF4.datetime objects instead, reason: dates out'
                          ' of range', RuntimeWarning, stacklevel=2)
            dates = np.asarray(dates)

    else:
        if min_num == np.datetime64('NaT'):
            dates = np.repeat(np.datetime64('NaT'), num_dates.size)
        elif min_num == max_num:
            # we can't safely divide by max_num - min_num
            dates = np.repeat(np.datetime64(min_date), num_dates.size)
            if dates.size > 1:
                # don't bother with one element, since it will be fixed at
                # min_date and isn't indexable anyways
                dates[np.isnan(num_dates)] = np.datetime64('NaT')
        else:
            # Calculate the date as a np.datetime64 array from linear scaling
            # of the max and min dates calculated via num2date.
            flat_num_dates = num_dates.reshape(-1)
            # Use second precision for the timedelta to decrease the chance of
            # a numeric overflow
            time_delta = np.timedelta64(max_date - min_date).astype('m8[s]')
            if time_delta != max_date - min_date:
                raise ValueError('unable to exactly represent max_date minus'
                                 'min_date with second precision')
            # apply the numerator and denominator separately so we don't need
            # to cast to floating point numbers under the assumption that all
            # dates can be given exactly with ns precision
            numerator = flat_num_dates - min_num
            denominator = max_num - min_num
            dates = (time_delta * numerator / denominator
                     + np.datetime64(min_date))
        # restore original shape and ensure dates are given in ns
        dates = dates.reshape(num_dates.shape).astype('M8[ns]')

    return dates


# TIME_UNITS = set(['days', 'hours', 'minutes', 'seconds'])

def nctime_to_nptime(times):
    """Given an array of netCDF4.datetime objects, return an array of
    numpy.datetime64 objects of the same size"""
    times = np.asarray(times)
    new = np.empty(times.shape, dtype='M8[ns]')
    for i, t in np.ndenumerate(times):
        new[i] = np.datetime64(datetime(*t.timetuple()[:6]))
    return new


def encode_cf_datetime(dates, units=None, calendar=None):
    """Given an array of datetime objects, returns the tuple `(num, units,
    calendar)` suitable for a CF complient time variable.

    Unlike encode_cf_datetime, this function does not (yet) speedup encoding
    of datetime64 arrays. However, unlike `date2num`, it can handle datetime64
    arrays.

    See also
    --------
    netCDF4.date2num
    """
    import netCDF4 as nc4

    dates = np.asarray(dates)

    if units is None:
        units = infer_datetime_units(dates)
    if calendar is None:
        calendar = 'proleptic_gregorian'

    if np.issubdtype(dates.dtype, np.datetime64):
        # for now, don't bother doing any trickery like decode_cf_datetime to
        # convert dates to numbers faster
        # note: numpy's broken datetime conversion only works for us precision
        dates = dates.astype('M8[us]').astype(datetime)

    def encode_datetime(d):
        return np.nan if d is None else nc4.date2num(d, units, calendar)

    num = np.array([encode_datetime(d) for d in dates.flat])
    num = num.reshape(dates.shape)
    return (num, units, calendar)


def encode_cf_timedelta(timedeltas, units=None):
    if units is None:
        units = infer_timedelta_units(timedeltas)

    np_unit = {'seconds': 's', 'minutes': 'm', 'hours': 'h', 'days': 'D'}[units]
    num = timedeltas.astype('timedelta64[%s]' % np_unit).view(np.int64)

    missing = timedeltas == np.datetime64('NaT')
    if np.any(missing):
        num = num.astype(float)
        num[missing] = np.nan

    return (num, units)
