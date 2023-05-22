#!/bin/bash

#output_dir='Y:\hantman_data\jab_experiments\STA14\STA14\20230503\'
exp_list_file=$1
echo "Experiment list file $exp_list_file"
#i=1

compute_feat=$2
combine_movie=$3
echo "Compute Features: $compute_feat"
echo "Combine movie: $combine_movie"

while read -r line; do
    #i=$((i+1))
    exp_name=$line
    echo "file name: $exp_name"
    
    if [ "$compute_feat" = 1 ]; then
       feat_avg_side=$exp_name'hoghof_avg_side_biasjaaba.csv'
       feat_avg_front=$exp_name'hoghof_avg_front_biasjaaba.csv'
       feat_side=$exp_name'hoghof_side_biasjaaba.csv'
       feat_front=$exp_name'hoghof_front_biasjaaba.csv'
       if [ -f $feat_avg_side ]; then
           rm $feat_avg_side
       fi
       if [ -f $feat_avg_front ]; then
	   rm $feat_avg_front
       fi
       if [ -f $feat_side ]; then
	   rm $feat_side
       fi
       if [ -f $feat_front ]; then
	   rm $feat_front
       fi
       C:/Users/27rut/BIAS/build/Release/generate_hoghof.exe -o $exp_name
    fi
   
    ## generate movie_comb.avi
    if [ "$combine_movie" = 1 ];then
       movie_comb_file=$exp_name'movie_comb.avi'
       if [ -f "$movie_comb_file" ]; 
       then

            # if file exist the it will be removed
            rm $movie_comb_file
	    echo "$movie_comb_file deleted"
       else
            echo "movie_comb.avi does not exist"
       fi
       C:/Users/27rut/Downloads/ffmpeg-6.0-essentials_build/ffmpeg-6.0-essentials_build/bin/ffmpeg  -i $exp_name'/movie_sde.avi' -i $exp_name'/movie_frt.avi'  -filter_complex hstack -c:v mjpeg -q:v 2 -huffman optimal $exp_name'/movie_comb.avi'
    fi   

done < $exp_list_file
