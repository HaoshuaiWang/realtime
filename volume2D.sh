#!/bin/bash
awk -v FS=", " 'NR>=4 {sum+=($3*4)} END {print sum}' CELL_U.DAT 
