#ifndef BIAS_CAMERA_FINDER_HPP
#define BIAS_CAMERA_FINDER_HPP

#include "basic_types.hpp"
#include "guid_fwd.hpp"
#include "camera_fwd.hpp"
#include <string>

#ifdef WITH_FC2
#include "FlyCapture2_C.h"
#endif
#ifdef WITH_DC1394
#include <dc1394/dc1394.h> 
#endif
#ifdef WITH_SPIN
#include "SpinnakerC.h"
#include "node_map_spin.hpp"
#include "string_node_spin.hpp"
#endif

namespace bias {

    class CameraFinder
    {
        public:
            CameraFinder();
            ~CameraFinder();

            size_t numberOfCameras();
            Guid getGuidByIndex(unsigned int index);

            GuidSet getGuidSet();
            GuidList getGuidList();

            CameraPtrSet createCameraPtrSet();
            CameraPtrList createCameraPtrList();

            std::string getGuidListAsString();
            void printGuid();

        private:
            GuidSet guidSet_;
            
            void createQueryContext_fc2();
            void destroyQueryContext_fc2();

            void createQueryContext_dc1394();
            void destroyQueryContext_dc1394();

            void createQueryContext_spin();
            void destroyQueryContext_spin();

            void update();
            void update_fc2();
            void update_dc1394();
            void update_spin();

#ifdef WITH_FC2
        private:
            fc2Context queryContext_fc2_ = nullptr;
#endif
#ifdef WITH_DC1394
        private:
            dc1394_t *queryContext_dc1394_ = nullptr;
#endif
#ifdef WITH_SPIN
        private:
            spinSystem queryContext_spin_ = nullptr;

#endif
    };

} // namespace bias

#endif // #ifndef BIAS_CAMERA_FINDER_HPP
