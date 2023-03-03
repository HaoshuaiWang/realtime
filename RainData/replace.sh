#!/bin/bash

for i in {0..180..6}
do
    sed -i 's/^/S&/g' rain_$i.txt
    sed -i 's/,/ /g' rain_$i.txt
done
