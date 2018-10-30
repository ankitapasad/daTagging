#!/bin/bash
: '
argument1: number of files 
these many will be used to create the shuffled data
same files for all kinds of experiments
'

noFiles=${1?Error: no number given}
## rename the 'sample' directory later
mkdir -p humanSubjectEval/sample

## get the filenames being used
str="dataprep"
filelist=($(python getDataHumanSubjectEval.py $noFiles $str))
# shuf -n $noFiles dataset/daFiles/trainFiles.txt > humanSubjectEval/temp.txt

## make directories for saving the audio data
for j in "${filelist[@]}"; do
   mkdir -p humanSubjectEval/sample/audio/shuffled/$j
done

for j in "${filelist[@]}"; do
   mkdir -p humanSubjectEval/sample/audio/sorted/$j
done

# generate data in the respective formats
str="gendata"
filename=$(python getDataHumanSubjectEval.py $noFiles $str) 
# mv humanSubjectEval/sample humanSubjectEval/$filename

# # rename audio data directories
# for i in `seq 1 $noFiles`; do
# 	mkdir -p humanSubjectEval/sample/$i
# done

# declare -i var=1
# for j in "${filelist[@]}"; do
#    mv humanSubjectEval/$filename/$var humanSubjectEval/$filename/$j
#    var=$var+1
# done