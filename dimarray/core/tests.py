from dimarray.testing import testmod
#from ..testing import testmod

def main(**kwargs):

    import metadata as metadata
    import dimarraycls
    import axes as axes
    import _indexing as indexing
    import _reshape as reshape
    import _transform as transform
    import missingvalues as missingvalues 
    import _operation as operation
    import align as align

    testmod(metadata, **kwargs)
    testmod(dimarraycls, **kwargs)
    testmod(axes, **kwargs)
    testmod(indexing, **kwargs)
    testmod(transform, **kwargs)
    testmod(reshape, **kwargs)
    testmod(missingvalues, **kwargs)
    testmod(operation, **kwargs)
    testmod(align, **kwargs)

if __name__ == "__main__":
    main()
