""" A custom script to convert a notebook to docstring examples
"""
import sys
from os.path import basename
from process_notebook import filter_cells, read_nb

def main():
    if len(sys.argv) == 1:
        raise Exception('must provide notebook name as argument')
    nm = sys.argv[1]
    if len(sys.argv) >= 3:
        nm2 = sys.argv[2]
    else:
        nm2 = basename(nm.replace('.ipynb','.rst'))

    nb = read_nb(nm)

    text = []
    for cell in nb['worksheets'][0]['cells']:
        text.append('\n\n') # new line

        # code cells
        if cell['cell_type'] == 'code':

            code_lines = ['>>> ' + code for code in cell['input']]
            output_lines = []
            for output in cell['outputs']:
                output_lines.extend(output['text'])

            text.extend(code_lines) # add code lines
            text.extend(output_lines) # add code lines

        # add markdown cells
        elif cell['cell_type'] == 'markdown':
            text.extend(cell['source'])

    # write note rst to file
    with open(nm2, 'w') as f:
        f.writelines(text)

    print "output written to", nm2

if __name__ == '__main__':
    main()
