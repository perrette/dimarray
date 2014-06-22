import zipfile,os.path
import warnings
from os.path import dirname, abspath, join, exists

# current directory
DIREC=dirname(abspath(__file__))

# unzip data the first time this module is imported
def get_datadir():
    """ Return directory name for the datasets
    """
    return join(DIREC, 'unzipped')

def get_ncfile(fname='cmip5.CSIRO-Mk3-6-0.nc'):
    """ Return one netCDF file
    """
    return join(get_datadir(),fname)

def unzip(source_filename, dest_dir):
    """ unzip data files 
    
    from: http://stackoverflow.com/questions/12886768/simple-way-to-unzip-file-in-python-on-all-oses 
    """
    with zipfile.ZipFile(source_filename) as zf:
        for member in zf.infolist():
            # Path traversal defense copied from
            # http://hg.python.org/cpython/file/tip/Lib/http/server.py#l789
            words = member.filename.split('/')
            path = dest_dir
            for word in words[:-1]:
                drive, word = os.path.splitdrive(word)
                head, word = os.path.split(word)
                if word in (os.curdir, os.pardir, ''): continue
                #path = os.path.join(path, word)
            print member.filename, path
            zf.extract(member, path)

def unzip_data():
    try:
        unzip(join(DIREC, 'data.zip'), DIREC)
    except Exception, msg:
        warnings.warn('could not unzip data')
        print msg.message

if not exists(get_datadir()):
    unzip_data()
