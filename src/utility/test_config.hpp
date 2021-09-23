#ifndef TEST_CONFIG_HPP
#define TEST_CONFIG_HPP

#include <fstream>
#include <string>
#include <vector>
#include "rtn_status.hpp"

using namespace std;

struct TestConfig {

    int numCameras;
    std::vector<string>cam_suffix;
    int dir_len;
    std::vector<string>dir_list;
    int numFrames;
    int no_of_trials;
    int framerate;
    float latency_threshold;
    string nidaq_prefix;
    string f2f_prefix;
    string queue_prefix;
    string plugin_prefix;
    string logging_prefix;
    string imagegrab_prefix;

    string fieldnames[14] = { "numCameras", "cam_suffix", "dir_len", "dir_list",
                             "numFrames", "no_of_trials","framerate","latency_threshold","nidaq_prefix", "f2f_prefix",
                             "queue_prefix", "plugin_prefix", "logging_prefix", "framegrab_prefix"};
    

};

int convertValtoInt(string val);
float convertToFloat(string val);
void copyConfigField(std::shared_ptr<TestConfig> test_config, int fieldname_index, string val);
int read_testConfig(std::shared_ptr<TestConfig> test_config, ifstream& input_testconfig);

#endif