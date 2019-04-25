#include "hog.h"
#include "hof.h"
#include "lk.h"
//#include "../../src/crop.h"
#include <string>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <set>
#include "H5Cpp.h"


void parse_input_HOG(int nargs, char** arg_inputs, struct HOGParameters& params,
                 struct HOGImage& img, int& num_frames, int& verbose, std::string& view,
                 std::string& mv_path, struct CropParams& crp_params);

void parse_input_HOF(int nargs, char** arg_inputs, struct HOFParameters& params,
                     int& num_frames, int& verbose, std::string& view, 
                     std::string& mv_path, struct CropParams& crp_params);

void read_image(std::string filename, float* img,int w,int h);

void write_histoutput(std::string file,float* out_img, unsigned w, unsigned h,unsigned nbins);

//void read_structfromcsv(std::string filename, struct interest_pnts* pts);

void readfromcsv(std::string filename,std::vector<int>& interest_pnts);

void write_output(std::string file,float* out_img, unsigned w, unsigned h);

void clear_stringstream(std::stringstream& idx, std::stringstream& dx, std::stringstream& dy,
                        std::stringstream& mag, std::stringstream& th, std::stringstream& hist);

void IsSubset(std::set<char> A, std::set<char> B);

