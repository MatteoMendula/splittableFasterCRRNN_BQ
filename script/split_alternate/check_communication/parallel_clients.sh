#!/bin/bash
echo "---- Starting parallel execution ----"
echo "file name: "$1
echo "n clients: "$2
echo "compression: "$3

for ((j = 1; j <= $2; j++)); do
	if [[ "$1" == *"jpeg"* ]]; then
		echo "jpeg client"
		python "$1" -c=$3 -f=True -s=0.1 &
	else
		echo "split client"
		python "$1" -f=True -s=0.1 &
	fi
done

echo "---- Parallel execution done ----"
