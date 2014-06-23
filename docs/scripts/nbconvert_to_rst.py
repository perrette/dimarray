""" Convert a notebook to docstring examples (rst-like)

Usage:
    nbconvert_docstring.py <notebook.ipynb> [ <text.rst> ]

Options:
    -h --help   : display this help
"""
#!/usr/bin/python
from IPython.utils import py3compat
import base64
import sys, os
#import pypandoc
from os.path import basename, splitext, join, commonprefix
from process_notebook import filter_cells, read_nb
import docopt
from IPython.display import Image

def make_label(title):
    """ label for future reference
    """
    return '..  _'+title.replace(' ','_').replace(':','_')+':'

def main():

    args = docopt.docopt(__doc__)

    nm = args['<notebook.ipynb>']
    nm2 = args['<text.rst>']

    filename = splitext(basename(nm))[0] # all but the extension

    if nm2 is None:
         nm2 = basename(nm.replace('.ipynb','.rst'))

    figdir = splitext(nm2)[0]+'_figures' # store figures

    nb = read_nb(nm)

    label =  '..  _page_'+filename+':'  # page label

    header = """
.. This file was generated automatically from the ipython notebook:
.. {notebook}
.. To modify this file, edit the source notebook and execute "make rst" 
    """.format(notebook=nm).strip()

    text = [header + '\n\n'] 
    text += [label + '\n'] # add a label to allow hyperlinks

    for icell, cell in enumerate(nb['worksheets'][0]['cells']):
        text.append('\n\n') # new line

        # code cells
        if cell['cell_type'] == 'code':

            code_lines = []
            for code in cell['input']:
                # do not add '>>>' if indent
                if code.startswith('%'): # magic commadn
                    code = code.strip() + ' # doctest: +SKIP \n' # don't bother with doctets
                if not code.startswith('  '):
                    code = '>>> '+code
                else:
                    code = '... '+code
                code_lines.append(code)

            output_lines = []
            for ioutput, output in enumerate(cell['outputs']):

                # replace with blankline for doctest
                for i, line in enumerate(output['text']):
                    if line == '\n':
                        output['text'][i] = "<BLANKLINE>\n" 

                    elif "<matplotlib.axes.AxesSubplot" in line:
                        code_lines[-1] += ' # doctest: +SKIP' # don't bother with doctets

                # figure
                if output['output_type'] == 'display_data':
                    
                    # save figure
                    figname = 'figure_{}-{}.png'.format(icell, ioutput)
                    figfull = join(figdir, figname)
                    if not os.path.exists(figdir):
                        os.makedirs(figdir) # create fig directory

                    #image = Image(output['png'])
                    # make some conversions (after: https://github.com/ipython/ipython/blob/master/IPython/nbconvert/preprocessors/extractoutput.py)
                    data = output['png']
                    data = py3compat.cast_bytes(data)
                    data = base64.decodestring(data)
                    # write to file
                    with open(figfull, mode='w') as fig:
                        print "Save figure to", figfull
                        fig.write(data) # write figure
                        #fig.write(output['png']) # write figure

                    # include figure
                    figinc = join(basename(figdir), figname)
                    output_lines.append('\n\n')
                    output_lines.append('.. image:: '+figinc)
                    output_lines.append('\n\n')
                    #output_lines.extend(output['text'])

                else:
                    output_lines.extend(output['text'])


            # Here should include figures

            text.extend(code_lines) # add code lines
            text.append('\n')
            text.extend(output_lines) # add code lines

        # add markdown cells
        elif cell['cell_type'] == 'markdown':

            md = cell['source']
            text.extend(md)

            # convert to markdornw to rst? long and make things worse
            #rst = [pypandoc.convert(m, 'rst', format='markdown') for m in md]
            #text.extend(rst)

        # parse headers 
        elif cell['cell_type'] == 'heading': 

            title = cell['source'][0] # assume one line only
            lev = cell['level']
            symbols = ['=', '-', '~', '_', '+', '#'] # title symbols?
            offset = 1 # start at heading level 1
            sym = symbols[lev-offset]

            #add a label to the title, with file name and title separated by '_', 
            # and whitespace also replaced by '_'
            # do not put a label on level 1 (top) title since this would interfere with page label
            if lev != 1:
                label = make_label(title)+'\n'
                text.append(label)
                text.append('\n')

            text.append(title)
            text.append('\n')
            text.append(sym*len(title))

    # write note rst to file
    with open(nm2, 'w') as f:
        f.writelines(text)

    print "output written to", nm2

if __name__ == '__main__':
    main()
