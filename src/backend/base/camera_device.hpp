#ifndef BIAS_CAMERA_DEVICE_HPP
#define BIAS_CAMERA_DEVICE_HPP

#ifdef WIN32
#include "win_time.hpp"
#endif

#include <string>
#include <memory>
#include <opencv2/core/core.hpp>
#include "basic_types.hpp"
#include "property.hpp"
#include "guid.hpp"
#include "format7.hpp"

#include "lockable.hpp"

#include <numeric>
#include "NIDAQUtils.hpp"
#include "test_config.hpp"

namespace bias 
{
    // Base class for camera device objects.

    class CameraDevice
    {
        public:
            CameraDevice();
            explicit CameraDevice(Guid guid); 

            Guid getGuid(); 
            virtual ~CameraDevice() {};
            virtual CameraLib getCameraLib();

            virtual void connect() {};
            virtual void disconnect() {};

            virtual void startCapture() {};
            virtual void stopCapture() {};
            virtual cv::Mat grabImage();
            virtual void grabImage(cv::Mat &image) {};

            virtual bool isConnected(); 
            virtual bool isCapturing();
            virtual bool isColor(); 
            virtual bool isSupported(VideoMode vidMode, FrameRate frmRate);
            virtual bool isSupported(ImageMode imgMode);
            virtual size_t getNumberOfImageMode();

            virtual VideoMode getVideoMode(); 
            virtual FrameRate getFrameRate(); 
            virtual ImageMode getImageMode(); 

            virtual VideoModeList getAllowedVideoModes(); 
            virtual FrameRateList getAllowedFrameRates(VideoMode vidMode); 
            virtual ImageModeList getAllowedImageModes();

            virtual Property getProperty(PropertyType propType);
            virtual PropertyInfo getPropertyInfo(PropertyType propType);
            virtual ImageInfo getImageInfo();

            virtual void setProperty(Property prop) {};
            virtual void setVideoMode(VideoMode vidMode, FrameRate frmRate) {};
            virtual void setFormat7ImageMode(ImageMode imgMode) {};

            virtual Format7Settings getFormat7Settings();
            virtual Format7Info getFormat7Info(ImageMode imgMode);

            virtual bool validateFormat7Settings(Format7Settings settings);
            virtual void setFormat7Configuration(Format7Settings settings, float percentSpeed);

            virtual PixelFormatList getListOfSupportedPixelFormats(ImageMode imgMode);

            virtual void setTriggerInternal() {};
            virtual void setTriggerExternal() {};
            virtual TriggerType getTriggerType();

            virtual TimeStamp getImageTimeStamp();

            virtual std::string getVendorName();
            virtual std::string getModelName();

            virtual std::string toString(); 
            virtual void printInfo() {};
            virtual void printGuid() {};

            virtual TimeStamp getDeviceTimeStamp();
            virtual void initCounter();

            TimeStamp cam_ofs = { 0,0 };
            TimeStamp cpu_time = { 0,0 };
            vector<int64_t>skipFrames;
            virtual TimeStamp cameraOffsetTime();
            virtual TimeStamp getCPUtime();	

            //Test
            virtual void setupNIDAQ(std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task,
                                    bool testConfigEnabled, string trial_info,
                                    std::shared_ptr<TestConfig> testConfig,
                                    std::shared_ptr<Lockable<GetTime>> gettime,
                                    unsigned int cameraNumber);
            virtual void skipDetected(StampedImage& stampedImage);

        protected:
            Guid guid_;
            bool connected_;
            bool capturing_;
    };

    typedef std::shared_ptr<CameraDevice> CameraDevicePtr;
}


#endif // #ifndef BIAS_CAMERA_DEVICE_HPP
