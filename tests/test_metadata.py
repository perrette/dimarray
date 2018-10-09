""" Test metadata
"""
from __future__ import print_function
import pytest
import dimarray as da
from dimarray.testing import create_metadata


class AbstractTestGetSetDel(object):
    """
    """

    #
    # attrs accessible via __getattr__, __setattr__, __delattr__
    #
    def test_setter(self):
        assert self.obj.attrs == {} # clean at first
        self.obj.some_metadata = 'm' # setter
        assert 'some_metadata' in self.obj.attrs and self.obj.some_metadata == 'm'

    def test_getter(self):
        self.obj.attrs['other_metadata'] = 'bla'
        assert hasattr(self.obj, 'other_metadata') and self.obj.other_metadata == 'bla'

    def test_deleter(self):
        self.obj.attrs['other_metadata'] = 'bla'
        del self.obj.other_metadata
        assert 'other_metadata' not in self.obj.attrs

    #
    # unless we deal with private attributes
    #
    def test_setter_private(self):
        assert self.obj.attrs == {} # clean at first
        self.obj._private_meta = 'private' 
        assert '_private_meta' not in self.obj.attrs

    def test_getter_private(self):
        self.obj.attrs['_private_meta'] = 'private'
        assert not hasattr(self.obj, '_private_meta')

    def test_deleter_private(self):
        self.obj.attrs['_private_meta'] = 'private'
        try:
            del self.obj._private_meta
        except AttributeError:
            pass
        assert '_private_meta' in self.obj.attrs

    #
    # methods, properties, and in general class attributes should not be affected
    # by the '.' syntax either
    #
    _special_values = []

    def test_setter_prop(self):
        for i, k in enumerate(self._special_values):
            try:
                setattr(self.obj, k, i) # dummy integer value i
            except:
                pass
            assert k not in self.obj.attrs

    def test_deleter_prop(self):
        for i, k in enumerate(self._special_values):
            self.obj.attrs[k] = i
            try:
                delattr(self.obj, k) # dummy integer value i
            except:
                pass
            print(self.obj.attrs)
            assert k in self.obj.attrs

class TestDimArray(AbstractTestGetSetDel):
    _special_values = ['values', 'axes', 'dims', 'labels', 'ndim', 'shape', 'dtype', 'take']
    def setup_method(self, method):
        self.obj = da.DimArray([1], dims=['dim0'])

class TestDataset(AbstractTestGetSetDel):
    _special_values = ['values', 'axes', 'dims', 'labels', 'update','keys']
    def setup_method(self, method):
        a = da.DimArray([1], dims=['dim0'])
        self.obj = da.Dataset({'a':a})

class TestAxis(AbstractTestGetSetDel):
    _special_values = ['values', 'name', 'dtype', 'take', '_monotonic']
    def setup_method(self, method):
        self.obj = da.Axis([1], name = "dim0")


class TestPropagation():

    def setup_method(self, method):
        self.obj = da.DimArray([[1,2],[3,4]])
        self.obj.attrs['some'] = 'more'
        self.obj.axes['x0'].attrs['me'] = 'too'

    def test_slicing(self):
        assert self.obj.attrs['some'] == 'more'
        new = self.obj[:1,:]
        assert 'some' in new.attrs and new.attrs['some'] == 'more'
        assert 'me' in new.axes['x0'].attrs and new.axes['x0'].attrs['me'] == 'too'

    def test_reindex(self):
        assert self.obj.attrs['some'] == 'more'
        new = self.obj.reindex_axis([1,2,3,4])
        assert 'some' in new.attrs and new.attrs['some'] == 'more'
        assert 'me' in new.axes['x0'].attrs and new.axes['x0'].attrs['me'] == 'too'

    def test_op(self):
        assert 'some_meta' not in (self.obj*2).attrs
        assert 'some_meta' not in (self.obj+2).attrs
        assert 'some_meta' not in (self.obj**2).attrs

    def test_reduce(self):
        # assert 'some' not in (self.obj.mean(axis=0)).attrs
        assert 'some' in (self.obj.mean(axis=0)).attrs
