import glob
import dimarray as da

#
# Add or update metadata
#
for nm in glob.glob('cmip5.*.nc'):

    f = da.read_nc(nm)
    f.reset_axis(False, axis='time', units='years since J.C', inplace=True)
    f.reset_axis(False, axis='scenario', units='', long_name='RCP scenarios', inplace=True)

    f['tsl'].long_name = 'global mean thermosteric sea level'
    f['tsl'].units = 'm'

    f['temp'].long_name = 'global mean surface temperature'
    f['temp'].units = 'deg. C'

    if 'ohu' in f.keys(): del f['ohu']  # remove it

    model = nm.split('.')[1] # retrieve model name
    f.model = model
    f.description = "Example data derived from the CMIP5 archive."
    f.disclaimer = "This is an example dataset and may contain errors. Please refer to the PCMDI data portal."
    f.source = "http://cmip-pcmdi.llnl.gov/cmip5/data_portal.html"

    f.write_nc(nm, mode='w')
