#ifndef UTILS_HPP
#define UTILS_HPP 

#include <string>
//#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <set>
#include "H5Cpp.h"


void read_image(std::string filename, float* img, int w, int h);

void write_histoutput(std::string file, float* out_img, unsigned w, unsigned h, unsigned nbins);

//void read_structfromcsv(std::string filename, struct interest_pnts* pts);

void readfromcsv(std::string filename,std::vector<int>& interest_pnts);

void write_output(std::string file,float* out_img, unsigned w, unsigned h);

void clear_stringstream(std::stringstream& idx, std::stringstream& dx, std::stringstream& dy,
                        std::stringstream& mag, std::stringstream& th, std::stringstream& hist);

void IsSubset(std::set<char> A, std::set<char> B);

void copy_features1d(int frame_num, int num_elements, std::vector<float> &vec_feat, float* array_feat);

//https://support.hdfgroup.org/HDF5/doc/cpplus_RM/readdata_8cpp-example.html
template<typename T>
void readh5(std::string filename, std::string dataset_name, T* data_out) {

    H5::H5File file(filename, H5F_ACC_RDONLY);
    int rank, ndims;
    hsize_t dims_out[2];

    H5::DataSet dataset = file.openDataSet(dataset_name);
    H5::DataSpace dataspace = dataset.getSpace();
    rank = dataspace.getSimpleExtentNdims();
    ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
    H5::DataSpace memspace(rank, dims_out);
    dataset.read(data_out, H5::PredType::IEEE_F32LE, memspace, dataspace);
    file.close();

}

void create_dataset(H5::H5File& file, std::string key, std::vector<float> features, 
                    int num_frames, int num_elements) ;

int createh5( std::string exp_path, std::string exp_name,
               int num_frames, int hog_elements1, int hof_elements1,
               int hog_elements2, int hof_elements2,
               std::vector<float> hog1, std::vector<float> hog2,
               std::vector<float> hof1, std::vector<float> hof2) ;

#endif