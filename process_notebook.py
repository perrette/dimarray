#!/usr/bin/python

import json

def read_nb(fname):
    f = open(fname)
    nb = json.load(f)
    f.close()
    return nb

def write_nb(nb, fname):
    f = open(fname, 'w')
    json.dump(nb, f)
    f.close()

def filter_cells(cells=None, nb=None, worksheet=0, cell_type=None, custom=None, strip=True, **kwargs):
    """ filter notebook cells

    cells: list of cells (e.g. nb['worksheets'][0]['cells'])
    nb: notebook, if cells is not provided
    worksheet: worksheet number, if notebook if provided (default 0)
    cell_type
    custom: a function on cell returning True or False
    **kwargs: keyword arguements to test various cell fields
    """
    if cells is None:
	cells = nb['worksheets'][worksheet]['cells']

    # filtering function
    def filt(x):
	res = True
	if res and cell_type is not None:
	    res = x['cell_type'] == cell_type
	for k in kwargs:
	    if not res: break
	    if strip:
		res = x[k].strip() == kwargs[k]
	    else:
		res = x[k] == kwargs[k]

	if res and custom:
	    res = custom(x)

	return res

    return filter(filt, cells)

def get_headings(nb, minlev=0, maxlev=100):
    #heading_cells = filter(lambda x: x['cell_type'] == 'heading', nb['worksheets'][0]['cells'])
    heading_cells = filter_cells(nb=nb, cell_type='heading', custom=lambda x: x['level'] >= minlev and x['level'] <= maxlev)
    headings = [cell['source'] for cell in heading_cells]
    levels = [cell['level'] for cell in heading_cells]
    return headings, levels

def get_group(nb, heading, worksheets=0):
    """ get all cells under a given heading
    """
    start = end = level = None
    for i, cell in enumerate(nb['worksheets'][worksheet]['cells']):

	# find the heading?
	if cell['cell_type'] =='heading' and cell['source'].strip() == heading:
	    start = i
	    level = cell['level']
	    continue

	# stop when a heading of level equal or superior is found
	if start is not None and cell['cell_type'] =='heading' and cell['level'] >= level:
	    end = i
	    break

    if end is None: end = -1
    assert start is not None, "Header not found"

    cells = nb['worksheets'][worksheet]['cells'][start:end]
    return cells

def create_toc(headings, levels, minlev = 0, maxlev = 6):
    toc = []
    for i, hd in enumerate(headings):
	lev = int(levels[i])
	if lev > maxlev or lev < minlev:
	    continue

	url = "#"+hd[0].replace(' ','-').replace('`','')
	entry = "    "*(lev-minlev) + "- [{}]({})".format(hd[0], url)
	toc.append(entry)
    return toc

def get_toc_cell(nb, FLAG, worksheet=0):
    """ return index of TOC cell
    """
    for i, cell in enumerate(nb['worksheets'][worksheet]['cells']):
	if cell['cell_type'] == 'markdown' and cell['source'][0].strip() == FLAG:
	    break

    return i, cell
    
def replace_toc(nb, toc, FLAG):
    """ replace TABLE of content
    """
    i, cell = get_toc_cell(nb, FLAG)
    cell['source'] = "\n".join([FLAG]+toc)

def main():

    nb = read_nb(fname='dimarray.ipynb')

    # Create a table of content
    headings, levels = get_headings(nb)
    toc = create_toc(headings, levels, minlev=2)
    print "\n".join(toc)

    # Update TOC cell
    replace_toc(nb, toc, FLAG='### Table of Content')
    write_nb(nb, fname='dimarray_test.ipynb')

if __name__ == '__main__':
    main()
