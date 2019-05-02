#include <string>
#include <unistd.h>
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

void readh5(std::string filename, std::string dataset_name, float* data_out);

void create_dataset(H5::H5File& file, std::string key, std::vector<float> features, 
                    int num_frames, int num_elements) ;

int createh5( std::string exp_path, std::string exp_name,
               int num_frames, int hog_elements1, int hof_elements1,
               int hog_elements2, int hof_elements2,
               std::vector<float> hog1, std::vector<float> hog2,
               std::vector<float> hof1, std::vector<float> hof2) ;
