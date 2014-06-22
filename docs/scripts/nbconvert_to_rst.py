""" Convert a notebook to docstring examples (rst-like)

Usage:
    nbconvert_docstring.py <notebook.ipynb> [ <text.rst> ]

Options:
    -h --help   : display this help
"""
#!/usr/bin/python
import sys
#import pypandoc
from os.path import basename, splitext
from process_notebook import filter_cells, read_nb
import docopt

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

    nb = read_nb(nm)

    label =  '..  _page_'+filename+':'  # page label

    header = """
.. This file was generated automatically from the ipython notebook:
.. {notebook}
.. To modify this file, edit the source notebook and execute "make rst" 
    """.format(notebook=nm).strip()

    text = [header + '\n\n'] 
    text += [label + '\n'] # add a label to allow hyperlinks

    for cell in nb['worksheets'][0]['cells']:
        text.append('\n\n') # new line

        # code cells
        if cell['cell_type'] == 'code':

            code_lines = []
            for code in cell['input']:
                # do not add '>>>' if indent
                if not code.startswith('  '):
                    code = '>>> '+code
                else:
                    code = '... '+code
                code_lines.append(code)

            output_lines = []
            for output in cell['outputs']:
                
                # replace with blankline for doctest
                for i, line in enumerate(output['text']):
                    if line == '\n':
                        output['text'][i] = "<BLANKLINE>\n" 
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
