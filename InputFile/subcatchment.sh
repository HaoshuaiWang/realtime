#!/bin/bash

var1=`awk '$1~/\[SUBCATCHMENTS\]/ {print NR}' tty.inp`
let var1=$var1+3
echo "$var1"
var2=`awk '$1~/\[SUBAREAS\]/ {print NR}' tty.inp`
let var2=$var2-2
echo "$var2"

awk 'NR>=var1 && NR<=var2 {$2=R$1; print $1,$2,$3,$4,$5,$6,$7,$8}' var1="$var1" var2="$var2" tty.inp > subcatchments.dat

echo "[RAINGAGES]" >> RainGage.dat
awk '{print $2}' subcatchments.dat > gageName.dat
awk '{print $0,"INTENSITY","0:05","1.0","TIMESERIES","T"$0}' gageName.dat >> RainGage.dat
echo "[TIMESERIES]" >> TimeSeries.dat
awk '{print "T"$0,"0:00","0.0"}' gageName.dat >> TimeSeries.dat
sed -i '1i\[SUBCATCHMENTS]' subcatchments.dat 
