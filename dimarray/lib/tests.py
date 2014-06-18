import transform
import stats
from dimarray.testing import testmod, MyTestResults

def main(**kwargs):

    test = MyTestResults(0,0)
    test += testmod(transform, **kwargs)
    test += testmod(stats, **kwargs)
    return test

if __name__ == "__main__":
    main()
