#!/bin/bash
if [ "$#" == 1 ] && [ "$1" == "--help" ]; then
    echo "Example Usage:"
    echo "    sh gatherData.sh (gmail id) (gmail pw) (file to read, e.g. sampleTest/simple_single.txt)"
    exit 0
fi

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    exit -1
fi

if [ "$#" == 3 ]; then
  python collectData.py $1 $2 $3
else
  exit -1
fi
