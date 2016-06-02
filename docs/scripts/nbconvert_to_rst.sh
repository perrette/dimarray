#!/bin/bash
# convert all jupyter notebook to rst format

NOTEBOOKDIR=$1
RSTDIR=$2

#NOTEBOOKDIR=notebooks
#RSTDIR=_rst
indexfile=$RSTDIR/index.rst
TOC="" # --toc to include TOC
DOWN="" # --download to include download link
DOWN="--download" # --download to include download link

# start the index
echo "Documentation" > $indexfile
echo "-------------" >> $indexfile
echo "             " >> $indexfile
echo ".. toctree::"   >> $indexfile
echo "   :maxdepth: 3" >> $indexfile
echo "             " >> $indexfile


for file in $NOTEBOOKDIR/*ipynb ; do
    # skip files starting with _
    firstletter=`basename $file | cut -c1`
    if [ $firstletter == '_' ] ; then
	continue
    fi

    rstfile=$RSTDIR/`basename ${file%.ipynb}.rst`
    fmt=`bash scripts/nbformat.sh $file`

    if [[ $fmt != 3 ]] ; then
        # convert to nbformat 3 prior to parsing...
        jupyter nbconvert --to notebook --nbformat 3 $file
        filetmp=`basename $file`
        filetmp=${filetmp%.ipynb}.v3.ipynb
        filetmp2=$NOTEBOOKDIR/$filetmp
        mv $filetmp $filetmp2
        python scripts/nbconvert_to_rst.py $filetmp2 $rstfile $TOC $DOWN
        rm $filetmp2
        sed -i "s/.v3//" $rstfile
    # if [[ $fmt != 4 ]] ; then
    #     # convert to nbformat 4 prior to parsing...
    #     jupyter nbconvert --to notebook --nbformat 4 $file
    #     filetmp=`basename $file`
    #     filetmp=${filetmp%.ipynb}.nbconvert.ipynb
    #     filetmp2=$NOTEBOOKDIR/$filetmp
    #     mv $filetmp $filetmp2
    #     python scripts/nbconvert_to_rst.py $filetmp2 $rstfile $TOC $DOWN
    #     rm $filetmp2
    #     sed -i "s/.nbconvert//" $rstfile
    else
        python scripts/nbconvert_to_rst.py $file $rstfile $TOC $DOWN
    fi

    # generate the index
    echo "   `basename $rstfile`" >> $indexfile
done
 

echo "index written to $indexfile" 
