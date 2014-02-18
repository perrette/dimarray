import transform
import stats
from dimarray.testing import testmod

def main(**kwargs):

    testmod(transform, **kwargs)
    testmod(stats, **kwargs)

if __name__ == "__main__":
    main()
