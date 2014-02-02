#!/usr/bin/python

import json

def get_nb(fname):
    f = open(fname)
    nb = json.load(f)
    f.close()
    return nb

def get_headings(fname='dimarray.ipynb'):
    nb = get_nb(fname)
    heading_cells = filter(lambda x: x['cell_type'] == 'heading', nb['worksheets'][0]['cells'])
    headings = [cell['source'] for cell in heading_cells]
    levels = [cell['level'] for cell in heading_cells]
    return headings, levels

def create_toc(headings, levels, minlev = 2, maxlev = 3):
    toc = []
    for i, hd in enumerate(headings):
	lev = int(levels[i])
	if lev > maxlev or lev < minlev:
	    continue

	url = "#"+hd[0].replace(' ','-').replace('`','')
	entry = "    "*(lev-minlev) + "- [{}]({})".format(hd[0], url)
	toc.append(entry)
    return toc

def main():
    headings, levels = get_headings(fname='dimarray.ipynb')
    toc = create_toc(headings, levels)
    print "\n".join(toc)

if __name__ == '__main__':
    main()
