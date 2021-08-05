#ifndef SPIN_UTILS_HPP
#define SPIN_UTILS_HPP

#include "SpinnakerC.h"
#include "video_utils.hpp"
#include "NIDAQUtils.hpp"
#include "win_time.hpp"

using namespace bias;

typedef enum _chunkDataType
{
    IMAGE,
    NODEMAP
} chunkDataType;

typedef enum _triggerType
{
    SOFTWARE,
    HARDWARE
} triggerType;

const chunkDataType chosenChunkData = IMAGE;
const triggerType chosenTrigger = HARDWARE;

class SpinUtils {

    public:

       
        SpinUtils();
        ~SpinUtils();
        bias::GetTime* gettime;
        //std::vector<std::vector<float>> timeStamps; 
        
        void PrintRetrieveNodeFailure(char node[], char name[]);
        bool8_t IsAvailableAndWritable(spinNodeHandle hNode, char nodeName[]);
        bool8_t IsAvailableAndReadable(spinNodeHandle hNode, char nodeName[]);
        spinError ConfigureTrigger(spinNodeMapHandle hNodeMap);
        spinError ConfigureChunkData(spinNodeMapHandle hNodeMap);
        spinError DisableChunkData(spinNodeMapHandle hNodeMap);
        spinError PrintDeviceInfo(spinNodeMapHandle hNodeMap);
        spinError ReleaseSystem(spinSystem& hSystem, spinCameraList& hCameraList);
        spinError setupSystem(spinSystem& hSystem, spinCameraList& hCameraList);
        
        spinError getFrame_camera(spinCamera& hCam, spinImage& hImage);
        spinError initialize_camera(spinCamera& hCam, spinNodeMapHandle& hNodeMap,
        spinNodeMapHandle& hNodeMapTLDevice);
        spinError deInitialize_camera(spinCamera& hCam, spinNodeMapHandle& hNodeMap);

        

};
#endif