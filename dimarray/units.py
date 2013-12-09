""" small module to handle unit operations

>>> import dimarray.units as units
>>> import dimarray.metadata as md
>>> u = md.Units.loads("m^2 m kg")
>>> v = md.Units.loads("m^4 kg^-1 K")
>>> u*v
[K, m^7]
"""
import json

#
# play with units
#

class BaseUnits(object):
    """ represent one unit
    """
    sep = "^"
    def __init__(self, base, power=1):
	assert type(base) is str, "base must be string"
	self.base = base
	self.power = power

    @classmethod
    def loads(cls, rep):
	""" load units from string rep 
	"""
	base_power = rep.split(cls.sep)
	if len(base_power) > 1:
	    base, power = base_power
	    power = json.loads(power) # str to int or float
	else:
	    base = base_power[0]
	    power = 1

	return BaseUnits(base, power)

    def dumps(self):
	if self.power == 1:
	    return self.base

	powers = json.dumps(self.power)
	return self.sep.join([self.base, powers])

    def __pow__(self, n):
	""" power
	"""
	power = self.power + n
	return BaseUnits(self.base, power)

    def __repr__(self):
	return self.dumps()

    def __eq__(self, other):
	return self.base == other.base and self.power == other.power

class Units(list):
    """ list of units: handle operations

    TO DO : check out http://pint.readthedocs.org/
    """
    sep = " "

    @classmethod
    def loads(cls, rep):
	""" load units from string rep 
	"""
	u = cls()
	for urep in rep.split(cls.sep):
	    u.append(BaseUnits.loads(urep))
	return u

    def dumps(self):
	return self.sep.join([u.dumps() for u in self])

    def bases(self):
	""" return a `set` of bases strings
	"""
	return {u.base for u in self}

    def compress(self):
	""" join all bases and power
	"""
	units = Units()
	for b in self.bases():
	    n = sum(u.power for u in self if u.base == b) # sum-up the power
	    units.append(BaseUnits(b, n))
	return units

    def power(self, n):
	""" power of a unit

	n: int or float
	"""
	u = Units()
	for bu in self.compress():
	    u.append(bu.power(n))
	return u

#    def __eq__(self, other):
#	return self.compress() == other.compress()

    def multiply(self, other):
	"""
	"""
	common = self.bases().intersection(other.bases())

	self = self.compress()
	other = other.compress()

	# units not in common
	u_self = Units([b for b in self if b.base not in common])
	u_other = Units([b for b in other if b.base not in common])

	u = u_self + u_other

	for c in self:
	    if c.base not in other.bases(): continue
	    o = [oo for oo in other if oo.base == c.base]
	    assert len(o) == 1, "pb"
	    o = o[0]
	    u_common = BaseUnits(c.base, c.power+o.power)
	    if u_common.power != 0:
		u.append(u_common)

	return Units(u)

    def divide(self, other):
	return self.multiply(other.power(-1))

    def __mul__(self, other):
	return self.multiply(other)

    def __divide__(self, other):
	return self.divide(other)

    def __power__(self, n):
	return self.power(n)

