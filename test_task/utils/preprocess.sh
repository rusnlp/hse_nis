#! /bin/bash

for i in ${1}/*.txt
do
    echo ${i}
    cat ${i} | ./udpipe --tokenize --tag english-lines-ud-2.3.udpipe > ${i}.conllu
done