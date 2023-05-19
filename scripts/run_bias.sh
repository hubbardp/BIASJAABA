#!/bin/bash

#output_dir='Y:\hantman_data\jab_experiments\STA14\STA14\20230503\'
exp_list_file=$1
echo "Experiment list file $exp_list_file"
i=1
while read line; do
    i=$((i+1))
    exp_name=$line
    echo "file name: $exp_name"
    #C:/Users/27rut/BIAS/build/Release/test_gui.exe -o $exp_name -i 1 -c 1 -f 2798 -k 0 -w 5000000 -s 1
    C:/Users/27rut/BIAS/build/Release/generate_hoghof.exe -o $exp_name
    C:/Users/27rut/Downloads/ffmpeg-6.0-essentials_build/ffmpeg-6.0-essentials_build/bin/ffmpeg  -i $exp_name'/movie_sde.avi' -i $exp_name'/movie_frt.avi'  -filter_complex hstack -c:v mjpeg -q:v 2 -huffman optimal $exp_name'/movie_comb.avi'

done < $exp_list_file
