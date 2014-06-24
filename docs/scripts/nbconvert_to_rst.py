""" Convert a notebook to rst file

This include html outputs and png figures.

Usage:
    nbconvert_docstring.py <notebook.ipynb> [ <text.rst> ] [ --toc]

Options:
    -h --help   : display this help
    --toc       : create a Table of Content
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
    return '.. _'+title.replace(' ','_').replace(':','_')+':'

def main():

    args = docopt.docopt(__doc__)

    nm = args['<notebook.ipynb>']
    nm2 = args['<text.rst>']

    filename = splitext(basename(nm))[0] # all but the extension

    if nm2 is None:
         nm2 = basename(nm.replace('.ipynb','.rst'))

    filesdir = splitext(nm2)[0]+'_files' # store figures

    nb = read_nb(nm)

    label =  '.. _page_'+filename+':'  # page label

    header = """
.. This file was generated automatically from the ipython notebook:
.. {notebook}
.. To modify this file, edit the source notebook and execute "make rst" 
    """.format(notebook=nm).strip()

    # to include a TOC under the first header
    toc = """
.. contents:: 
    :local: 

"""
    insert_toc = args['--toc'] # insert toc?

    text = []

    text += [header + '\n\n'] 
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

                    #elif "<matplotlib.axes.AxesSubplot" in line:
                    #elif "<matplotlib.contour.QuadContourSet" in line:
                    elif "<matplotlib." in line and not "+SKIP" in code_lines[-1]: # whatever that is a figure
                        code_lines[-1] += ' # doctest: +SKIP' # don't bother with doctets

                # figure
                if output['output_type'] == 'display_data':
                    
                    # save figure
                    figname = 'figure_{}-{}.png'.format(icell, ioutput)
                    figfull = join(filesdir, figname)
                    if not os.path.exists(filesdir):
                        os.makedirs(filesdir) # create fig directory

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
                    figinc = join(basename(filesdir), figname)
                    output_lines.append('\n\n')
                    output_lines.append('.. image:: '+figinc)
                    output_lines.append('\n\n')
                    #output_lines.extend(output['text'])

                elif output['output_type'] == 'pyout':

                    # write text output
                    output_lines.extend(output['text'])

                    # if html, display after the text
                    if 'html' in output.keys():
        
                        # does not seem to work
                        # print "html found"
                        # output_lines.append('\n\n')
                        # output_lines.append('.. raw:: html')
                        # output_lines.extend(output['html'])
                        # output_lines.append('\n\n')
                        # save html
                        filename = 'output_{}-{}.html'.format(icell, ioutput)
                        filefull = join(filesdir, filename)
                        if not os.path.exists(filesdir):
                            os.makedirs(filesdir) # create files directory
                        # write to file
                        f = open(filefull, mode='w')
                        #with open(filefull, mode='w') as f:
                        print "Save html to", filefull
                        try:
                            f.writelines(output['html']) # write html
                        except:
                            import ipdb
                            ipdb.set_trace()

                        # include html
                        fileinc = join(basename(filesdir), filename)
                        output_lines.append('\n\n')
                        #output_lines.append('.. html:: '+fileinc)
                        output_lines.append('.. raw:: html\n')
                        output_lines.append('     :file: '+fileinc)
                        #output_lines.append('.. include:: '+fileinc)
                        #output_lines.append('.. image:: '+fileinc)
                        output_lines.append('\n\n')
                        #output_lines.extend(output['text'])

                else:
                    print "Warning: unknown output type"
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


            # append TOC
            if insert_toc:
                text.append(toc)
                insert_toc = False

    # write note rst to file
    with open(nm2, 'w') as f:
        f.writelines(text)

    print "output written to", nm2

if __name__ == '__main__':
    main()
