#ifndef PARSER_HPP
#define PARSER_HPP

#include <stdio.h>
#include <string>
#include <iostream>

using namespace std;

namespace bias {

    struct CmdLineParams {

        string output_dir = "";
        string jaaba_config_path = ""; //added to specify jaaba config path for offline classifier
                                       // where option to add plugin config from bias config is absent.
        string movie_name_suffix = "";
        bool isVideo = false;
        bool saveFeat = false;
        bool compute_jaaba = false;
        bool classify_scores = false;
        bool visualize = false;
        bool isSkip = false;
        int wait_thres = 1500;
        int window_size = 5;
        bool debug = false;
        string comport = "";
        unsigned long frameGrabAvgTime = 2500; // avg time to 
        unsigned long framerate = 400;
        int skip_latency = 4000; // time in us
        double ts_match_thres = 1.0e-4; // ts match threshold
    };
    
    void print(CmdLineParams& cmdlineparams);

    /*class ParseCmdLine
    {
        public:
            ParseCmdLine();
            void parser(int argc, char *argv[], CmdLineParams& cmdlineparams);

    };*/
}
#endif