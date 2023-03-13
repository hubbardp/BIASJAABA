#include "utils.hpp"
#include <stdio.h>
#include <assert.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>


// write out the ouput as a 1d array
void write_histoutput(std::string file,float* out_img, unsigned w, unsigned h,unsigned nbins){

     std::ofstream x_out;
     x_out.open(file.c_str());

     // write hist output to csv file
     for(unsigned k=0;k < nbins;k++){
         for(unsigned i = 0;i < h;i++){
             for(unsigned j = 0; j < w;j++){
                 x_out << out_img[k*w*h +i*w + j]; // i*h +j
                 if(j != w-1 || i != h-1 || k != nbins -1)
                     x_out << ",";
             }
         }
     }
}


void readfromcsv(std::string filename, std::vector<int>& interest_pnts){

    std::string lin;
    std::string result;
    std::ifstream x_in(filename);
    int count_row=0;

    if(x_in){
        while(getline(x_in, lin)){ 
            std::stringstream iss(lin);
            if(std::getline(iss,result,','))
                interest_pnts[count_row]=atoi(result.c_str());
            if(std::getline(iss,result,','))
                interest_pnts[count_row+1]=atoi(result.c_str());
            count_row = count_row + 2; // read in x and y coordiante for each point
        } 
    }else{

        std::cout << "File not present.Enter a valid filename." << std::endl;
        exit(1);  
    } 
}


void write_output(std::string file,float* out_img, unsigned w, unsigned h) {

    std::ofstream x_out;
    x_out.open(file.c_str());

    // write hist output to csv file
    for(unsigned i = 0;i < h; i++) {

        for(unsigned j = 0; j < w;j++) {
 
            x_out << out_img[i*w + j];

                 if(j != w-1 || i != h-1)
                     x_out << ",";
        }
    }     
}


void clear_stringstream(std::stringstream& idx ,std::stringstream& dx ,std::stringstream& dy, 
                        std::stringstream& mag ,std::stringstream& th ,std::stringstream& hist){

    idx.str(std::string());
    hist.str(std::string());
    dx.str(std::string());
    dy.str(std::string());
    th.str(std::string());
    mag.str(std::string());

}


// function to check for mandatory arguements in getopt
void IsSubset(std::set<char> A, std::set<char> B) {

    std::set<char>::iterator it;
    for(auto f : A) {

       it = B.find(f);

       if (it == B.end()) {

           printf("Required Field %c .Please enter a valid value.\n",f);
           exit(1);
       }          
 
    } 
     
}


void copy_features1d(int frame_num, int num_elements, float* out_feat, float* in_feat) {
    // row to add is num_elements * frame_num
    int start_idx = frame_num * num_elements;
    for(int i = 0; i < num_elements; i++) {
        out_feat[i + start_idx] = in_feat[i];
    }
}

// vectors don't work well with hdf5 
//https://stackoverflow.com/questions/32447628/how-to-write-2d-std-vector-of-floats-to-hdf5-file-and-then-read-it-in-python
void create_dataset(H5::H5File& file, std::string key,
        float* features, int num_frames, int num_elements) {

    hsize_t dims[2];
    dims[0] = num_frames;
    dims[1] = num_elements;
    H5::DataSpace dataspace(2, dims); 
    H5::DataSet dataset = file.createDataSet(key, H5::PredType::IEEE_F32LE, dataspace);

    dataset.write(features, H5::PredType::IEEE_F32LE);

    dataset.close();
    dataspace.close();
}


int createh5( std::string exp_path, std::string exp_name,
               int num_frames, int dim_view1, int dim_view2,
               float* hog1, float* hog2,
               float* hof1, float* hof2) {
  
    // Test h5 creation
    std::string out_file = exp_path + exp_name;
    H5::H5File file(out_file.c_str(), H5F_ACC_TRUNC);

    // Create 4 datasets.
    create_dataset(file, "hog_side", hog1, num_frames, dim_view1);
    create_dataset(file, "hog_front", hog2, num_frames, dim_view2);
    create_dataset(file, "hof_side", hof1, num_frames, dim_view1);
    create_dataset(file, "hof_front", hof2, num_frames, dim_view2);

    file.close();

    return 0;
}

