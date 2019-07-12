#!/bin/bash

_DEBUG="on"
function DEBUG()
{
 [ "$_DEBUG" == "on" ] &&  $@
}
 
DEBUG echo 'downloadFiles'
input='/vulcan/scratch/mtang/datasets/ABIDE/min_process/abide_summary.csv'

while IFS=',' read -r f1 f2 f3 f4 f5 f6 f7 f8
do
	DEBUG set -x
	echo "Subject is:  $f7"
	curl -o $f7.nii.gz [download link]
	DEBUG set +x
	
done < "$input"
