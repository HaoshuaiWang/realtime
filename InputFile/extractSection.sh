#!/bin/bash

# Set the input file and output directory
input_file=$1
output_dir="output/"

# Use awk to split the file into sections
awk -F '[][]' -v output_dir="$output_dir" '{ if (NF == 3) { filename=$2; print $0 > (output_dir filename) } else { print $0 > (output_dir filename) } }' $input_file
#the '[][]' 表示正则表达式，表示分隔符是"["或者"]"，即，遇到这两个符号都可表示分隔符。
