from dimarray.testing import testmod

def main(**kwargs):

    import metadata as metadata
    import dimarraycls
    import axes as axes
    import _indexing as indexing
    import _reshape as reshape
    import _transform as transform

    testmod(metadata, **kwargs)

    testmod(dimarraycls, **kwargs)

    testmod(axes, **kwargs)

    testmod(indexing, **kwargs)

    testmod(transform, **kwargs)

    testmod(reshape, **kwargs)

if __name__ == "__main__":
    main()