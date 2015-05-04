__doc__=""" Convert a notebook to rst file

It inserts ">>>" in front of code lines to produce a nice formatting in sphinx.
It also handles html outputs and inline figures, can insert a Table of 
Content or insert a download link to the notebook. 

Usage:
    nbconvert_to_rst.py <notebook.ipynb> [ <text.rst> ] [ --toc] [ --download] [ --headers SYMBOLS]

Options:
    -h --help   : display this help
    --toc       : create a Table of Content under the first heading
    --download  : create a link to download the notebook
    --headers   : sequence of symbols to underline headers. The first symbol will 
                  be used for a notebook heading of level 1.
                  Default is: {symbols}
"""
#!/usr/bin/python
import sys, os
from os.path import basename, splitext, join, commonprefix

# to read the notebook 
import json

# to convert images
from IPython.utils.py3compat import cast_bytes
# from ipython_genutils.py3compat import cast_bytes  # ipython 3
import base64

# command-line args
import docopt

# update doc with default symbols
SYMBOLS = '=-^"~+_#'
__doc__ = __doc__.format(symbols=SYMBOLS )
    

def read_nb(fname):
    f = open(fname)
    nb = json.load(f)
    f.close()
    return nb

def get_cells(nb):
    """ worksheets disappeared in v4
    """
    if nb['nbformat'] < 4:
        cells = nb['worksheets'][0]['cells']
    else:
        cells = nb['cells']
    return cells

def get_cell_source(cell):
    try:
        src = cell['source']
    except:
        src = cell['input'] # nbformat < 4
    return src

def make_heading(title, lev, symbols=SYMBOLS):
    lines = []
    offset = 1 # start at heading level 1
    sym = symbols[lev-offset]

    #add a label to the title, with file name and title separated by '_', 
    # and whitespace also replaced by '_'
    # do not put a label on level 1 (top) title since this would interfere with page label
    if lev != 1:
        label = '.. _'+title.replace(' ','_').replace(':','_')+':' +'\n'
        lines.append(label)
        lines.append('\n')

    lines.append(title)
    lines.append('\n')
    lines.append(sym*len(title))

    return lines

def main():

    args = docopt.docopt(__doc__)

    nm = args['<notebook.ipynb>']
    nm2 = args['<text.rst>']

    filename = splitext(basename(nm))[0] # all but the extension

    if nm2 is None:
         nm2 = basename(nm.replace('.ipynb','.rst'))

    filesdir = splitext(nm2)[0]+'_files' # store figures

    nb = read_nb(nm)

    # to underline titles
    if args['--headers']:
        symbols = args['SYMBOLS']
    else:
        symbols = SYMBOLS

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

    # create a link to download the notebook
    path_to_nb = '/'+nm
    download = """
:download:`Download notebook <{}>` 
""".format(path_to_nb)

    insert_toc = args['--toc'] # insert toc?
    insert_download = args['--download'] # insert download link?

    text = []

    text += [header + '\n\n'] 
    text += [label + '\n'] # add a label to allow hyperlinks
    
#    if args['--download']:
#        text.append(download)

    for icell, cell in enumerate(get_cells(nb)):
        text.append('\n\n') # new line

        # code cells
        if cell['cell_type'] == 'code':

            code_lines = []
            for code in get_cell_source(cell):
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

                if nb['nbformat'] < 4:
                    output_text = output['text'] # nbformat < 4
                else:
                    output_text = output['data']["text/plain"] # nbformat >= 4

                # replace with blankline for doctest
                for i, line in enumerate(output_text):
                    if line == '\n':
                        output_text[i] = "<BLANKLINE>\n" 

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

                    # make some conversions (after: https://github.com/ipython/ipython/blob/master/IPython/nbconvert/preprocessors/extractoutput.py)
                    if nb['nbformat'] < 4:
                        data = output['png']
                    else:
                        data = "".join(output['data']['image/png'])
                    data = cast_bytes(data)
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
                    #output_lines.extend(output_text)

                elif output['output_type'] in ('execute_result','pyout','stream'):

                    # write text output
                    output_lines.extend(output_text)

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
                        except Exception as msg:
                            print "Failed !!!"
                            print(msg)
                            raise
                            #import ipdb
                            #ipdb.set_trace()

                        # include html
                        fileinc = join(basename(filesdir), filename)
                        output_lines.append('\n\n')
                        #output_lines.append('.. html:: '+fileinc)
                        output_lines.append('.. raw:: html\n')
                        output_lines.append('     :file: '+fileinc)
                        #output_lines.append('.. include:: '+fileinc)
                        #output_lines.append('.. image:: '+fileinc)
                        output_lines.append('\n\n')
                        #output_lines.extend(output_text)

                else:
                    print "Warning: unknown output type:",output['output_type']
                    output_lines.extend(output_text)

            # Here should include figures

            text.extend(code_lines) # add code lines
            text.append('\n')
            text.extend(output_lines) # add code lines

        # add markdown cells
        elif cell['cell_type'] == 'markdown':

            md = cell['source']

            for line in md:

                if line.startswith('#'):
                    # parse title in markdown mode
                    title = line.lstrip('#').strip()
                    lev = line.count('#') - title.count('#') # on the right side
                    print 'MAKE TITLE level',lev
                    lines = make_heading(title, lev, symbols)
                    text.extend(lines)

                    # append TOC
                    if insert_toc:
                        text.append(toc)
                        insert_toc = False

                    # insert download link
                    if insert_download:
                        text.append(download)
                        insert_download = False
                else:
                    text.append(line)

            # convert to markdornw to rst? long and make things worse
            #rst = [pypandoc.convert(m, 'rst', format='markdown') for m in md]
            #text.extend(rst)

        # parse headers 
        elif cell['cell_type'] == 'heading': 

            title = cell['source'][0] # assume one line only
            lev = cell['level']

            lines = make_heading(title, lev, symbols)

            text.extend(lines)

            # append TOC
            if insert_toc:
                text.append(toc)
                insert_toc = False

            # insert download link
            if insert_download:
                text.append(download)
                insert_download = False

    # write note rst to file
    with open(nm2, 'w') as f:
        f.writelines(text)

    print "output written to", nm2

if __name__ == '__main__':
    main()
