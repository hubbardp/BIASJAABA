#ifdef WITH_SPIN
#include "camera_device_spin.hpp"
#include "utils_spin.hpp"
#include "exception.hpp"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <bitset>
#include <fstream>

#ifdef linux
#include <sys/time.h>
#endif

/*#include "base_node_spin.hpp"
#include "string_node_spin.hpp"
#include "enum_node_spin.hpp"
#include "entry_node_spin.hpp"
#include "number_node_spin.hpp"
#include "float_node_spin.hpp"
#include "integer_node_spin.hpp"
#include "bool_node_spin.hpp"
#include "enum_node_spin.hpp"
#include "command_node_spin.hpp"
*/

namespace bias {

    constexpr double CameraDevice_spin::MinAllowedShutterUs;
    constexpr double CameraDevice_spin::MaxAllowedShutterUs;

    CameraDevice_spin::CameraDevice_spin() : CameraDevice() {}

    CameraDevice_spin::CameraDevice_spin(Guid guid) : CameraDevice(guid)
    {
        spinError err = spinSystemGetInstance(&hSystem_);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": unable to create Spinnaker context, error = " << err;
            throw RuntimeError(ERROR_SPIN_CREATE_CONTEXT, ssError.str());
        }
        gettime = new GetTime(0, 0);
        
        time_stamp3.resize(100000, std::vector<uInt32>(2, 0));
        
    }

    CameraDevice_spin::~CameraDevice_spin()
    {

        if (capturing_)
        {
            stopCapture();
        }

        if (connected_)
        {
            disconnect();
        }

        spinError err = spinSystemReleaseInstance(hSystem_);
        if ( err != SPINNAKER_ERR_SUCCESS )
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": unable to destroy Spinnaker context, error = " << err;
            throw RuntimeError(ERROR_SPIN_DESTROY_CONTEXT, ssError.str());
        }

        if (gettime != nullptr)
            delete gettime;
    }


    CameraLib CameraDevice_spin::getCameraLib()
    {
        return guid_.getCameraLib();
    }


    void CameraDevice_spin::connect()
    {
        if (!connected_)
        {
            spinError err = SPINNAKER_ERR_SUCCESS;
            spinCameraList hCameraList = nullptr;

            // Create empty camera list
            err = spinCameraListCreateEmpty(&hCameraList);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                std::stringstream ssError;
                ssError << __FUNCTION__;
                ssError << ": unable to create Spinnaker empty camera list, error=" << err;
                throw RuntimeError(ERROR_SPIN_CREATE_CAMERA_LIST, ssError.str());
            }

            // Retrieve list of cameras from system
            err = spinSystemGetCameras(hSystem_, hCameraList);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                std::stringstream ssError;
                ssError << __FUNCTION__;
                ssError << ": unable to enumerate Spinnaker cameras, error=" << err;
                throw RuntimeError(ERROR_SPIN_ENUMERATE_CAMERAS, ssError.str());
            }

            err = spinCameraListGetBySerial(hCameraList, guid_.toString().c_str() , &hCamera_);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                std::stringstream ssError;
                ssError << __FUNCTION__;
                ssError << ": unable to get Spinnaker camera from list, error = " << err;
                throw RuntimeError(ERROR_SPIN_GET_CAMERA, ssError.str());
            }

            // Clear Spinnaker camera list
            err = spinCameraListClear(hCameraList);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                std::stringstream ssError;
                ssError << __FUNCTION__;
                ssError << ": unable to clear Spinnaker camera list, error=" << err;
                throw RuntimeError(ERROR_SPIN_CLEAR_CAMERA_LIST, ssError.str());
            }

            // Destroy Spinnaker camera list
            err = spinCameraListDestroy(hCameraList);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                std::stringstream ssError;
                ssError << __FUNCTION__;
                ssError << ": unable to destroy Spinnaker camera list, error=" << err;
                throw RuntimeError(ERROR_SPIN_DESTROY_CAMERA_LIST, ssError.str());
            }

            // Initialize camera
            err = spinCameraInit(hCamera_);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                hCamera_ = nullptr;
                std::stringstream ssError;
                ssError << __FUNCTION__;
                ssError << ": unable to initialize Spinnaker camera, error=" << err;
                throw RuntimeError(ERROR_SPIN_GET_TLDEVICE_NODE_MAP, ssError.str());
            }
            connected_ = true;

            // Setup node maps for TLDevice and camera and get camera info
            nodeMapTLDevice_ = NodeMapTLDevice_spin(hCamera_);
            nodeMapCamera_ = NodeMapCamera_spin(hCamera_);
            nodeMapTLStream_ = NodeMapTLStream_spin(hCamera_);

            // Get Camera info
            cameraInfo_ = nodeMapTLDevice_.cameraInfo();
            cameraInfo_.print();


            // Default settings - may want to test for availability before setting.
            // ----------------------------------------------------------------------------------------------

            // Exposure defaults
            EnumNode_spin exposureModeNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("ExposureMode");
            if (exposureModeNode.isAvailable() && exposureModeNode.isWritable())
            {
                exposureModeNode.setEntryBySymbolic("Timed");
            }

            EnumNode_spin exposureAutoNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("ExposureAuto");
            if (exposureAutoNode.isAvailable() && exposureAutoNode.isWritable())
            {
                exposureAutoNode.setEntryBySymbolic("Off");
            }

            // Blacklevel defaults
            EnumNode_spin blackLevelSelectorNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("BlackLevelSelector");
            if (blackLevelSelectorNode.isAvailable() && blackLevelSelectorNode.isWritable())
            {
                blackLevelSelectorNode.setEntryBySymbolic("All");
            }

            // Gain defualts
            EnumNode_spin gainSelectorNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("GainSelector");
            if (gainSelectorNode.isAvailable() && gainSelectorNode.isWritable())
            {
                gainSelectorNode.setEntryBySymbolic("All");
            }

            EnumNode_spin gainAutoNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("GainAuto");
            if (gainAutoNode.isAvailable() && gainAutoNode.isWritable())
            {

                gainAutoNode.setEntryBySymbolic("Off");
            }


            // Trigger defaults
            EnumNode_spin triggerSelectorNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("TriggerSelector");
            if (triggerSelectorNode.isAvailable() && triggerSelectorNode.isWritable())
            {
                triggerSelectorNode.setEntryBySymbolic("FrameStart");
            }

            EnumNode_spin triggerOverlapNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("TriggerOverlap");
            if (triggerOverlapNode.isAvailable() && triggerOverlapNode.isWritable())
            {
                triggerOverlapNode.setEntryBySymbolic("Off");
            }

            triggerType_ = getTriggerType();
            setTriggerInternal();

            // Framerate defaults
            BoolNode_spin frameRateEnableNode = nodeMapCamera_.getNodeByName<BoolNode_spin>("AcquisitionFrameRateEnable");
            if (frameRateEnableNode.isAvailable() && frameRateEnableNode.isWritable())
            {
                frameRateEnableNode.setValue(true);
            }


            /*EnumNode_spin transferMode = nodeMapTLStream_.getNodeByName<EnumNode_spin>("StreamBufferHandlingMode");
            EntryNode_spin transferModeVal;

            if (transferMode.isAvailable() && transferMode.isWritable())
            {
                        transferMode.setEnumEntry("NewestOnly");

            } else {

                std::stringstream ssError;
                ssError << __FUNCTION__;
                ssError << ": ";
                throw RuntimeError(ERROR_SPIN_GET_ENUM_ENTRY_INT_VALUE, ssError.str());

            }*/

            // system Level defaults
            /*IntegerNode_spin transferModeVal = nodeMapTLStream_.getNodeByName<IntegerNode_spin>("ManualStreamBufferCount");
            if (transferModeVal.isAvailable() && transferModeVal.isWritable())
            {
                transferModeVal.setValue(1);
            }*/


            // DEVEL
            // ----------------------------------------------------------------------------------------------
            std::cout << "# TLDevice nodes: " << (nodeMapTLDevice_.numberOfNodes()) << std::endl;
            std::cout << "# Camera nodes:   " << (nodeMapCamera_.numberOfNodes()) << std::endl;
            std::cout << std::endl;

            // ----------------------------------------------------------------------------------------------
            // TODO: - setup strobe output on GPIO pin?? Is this possible?
            // ----------------------------------------------------------------------------------------------

            //develExpProps();

        }
    }


    void CameraDevice_spin::disconnect()
    {
        if (capturing_)
        {
            stopCapture();
        }

        if (connected_)
        {

            // Deinitialize camera
            spinError err = spinCameraDeInit(hCamera_);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                std::stringstream ssError;
                ssError << __FUNCTION__;
                ssError << ": unable to deinitialize Spinnaker camera, error=" << err;
                throw RuntimeError(ERROR_SPIN_RELEASE_CAMERA, ssError.str());
            }

            // Release Camera
            err = spinCameraRelease(hCamera_);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                std::stringstream ssError;
                ssError << __FUNCTION__;
                ssError << ": unable to get Spinnaker camera, error=" << err;
                throw RuntimeError(ERROR_SPIN_RELEASE_CAMERA, ssError.str());
            }

            connected_ = false;
        }
    }


    void CameraDevice_spin::startCapture()
    {

        std::cout << "DEBUG: " << __FUNCTION__ << " begin" << std::endl;

        if (!connected_)
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": unable to start Spinnaker capture - not connected";
            throw RuntimeError(ERROR_SPIN_START_CAPTURE, ssError.str());
        }

        // DEBUG
        // ----------------------------
        //printFormat7Configuration();
        // ----------------------------


        if (!capturing_)
        {

            // Set acquisition mode
            //std::cout << "DEBUG: set AcquisitionMode begin" << std::endl;
            EnumNode_spin acqModeNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("AcquisitionMode");
            if (acqModeNode.isAvailable())
            {
                acqModeNode.setEntryBySymbolic("Continuous");
            }
            //std::cout << "DEBUG: set AcquisitionMode end" << std::endl;

            ///////////////////////////////////////
            // WBD DEBUG
            ///////////////////////////////////////
            setupTimeStamping();

            // Begin acquisition
            spinError err = spinCameraBeginAcquisition(hCamera_);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                std::stringstream ssError;
                ssError << __FUNCTION__;
                ssError << ": unable to begin camera acquisition, error = " << err;
                throw RuntimeError(ERROR_SPIN_START_CAPTURE, ssError.str());
            }

            capturing_ = true;
            //isFirst_ = true;
            //
        }

        std::cout << "DEBUG: " << __FUNCTION__ << " end" << std::endl;

    }


    void CameraDevice_spin::stopCapture()
    {
        if (capturing_)
        {
            destroySpinImage(hSpinImage_);

            if (!releaseSpinImage(hSpinImage_))
            {
                std::stringstream ssError;
                ssError << __FUNCTION__;
                ssError << ": unable to release spinImage";
                throw RuntimeError(ERROR_SPIN_RELEASE_SPIN_IMAGE, ssError.str());
            }

            spinError err = spinCameraEndAcquisition(hCamera_);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                std::stringstream ssError;
                ssError << __FUNCTION__;
                ssError <<": unable to stop Spinnaker capture";
                throw RuntimeError(ERROR_SPIN_STOP_CAPTURE, ssError.str());
            }
            capturing_ = false;
        }
    }


    cv::Mat CameraDevice_spin::grabImage()
    {

        cv::Mat image;
        grabImage(image);
        return image;
    }


    void CameraDevice_spin::grabImage(cv::Mat &image)
    {

        //bool resize = false;

        std::string errMsg;

        bool ok = grabImageCommon(errMsg);
        
        if (!ok)
        {
            image.release(); 
            //std::cout << "Release" << std::endl;
            return;
        }

        spinError err = SPINNAKER_ERR_SUCCESS;
        spinImage hSpinImageConv = nullptr;

        err = spinImageCreateEmpty(&hSpinImageConv);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": unable to create empty spinImage, error = " << err;
            throw RuntimeError(ERROR_SPIN_IMAGE_CREATE_EMPTY, ssError.str());
        }

        spinPixelFormatEnums origPixelFormat = getImagePixelFormat_spin(hSpinImage_);
        spinPixelFormatEnums convPixelFormat = getSuitablePixelFormat(origPixelFormat);

        err = spinImageConvert(hSpinImage_, convPixelFormat, hSpinImageConv);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": unable to convert spinImage, error = " << err;
            throw RuntimeError(ERROR_SPIN_IMAGE_CONVERT, ssError.str());
        }

        ImageInfo_spin imageInfo = getImageInfo_spin(hSpinImageConv);

        int opencvPixelFormat = getCompatibleOpencvFormat(convPixelFormat);

        cv::Mat imageTmp = cv::Mat(
                imageInfo.rows+imageInfo.ypad,
                imageInfo.cols+imageInfo.xpad,
                opencvPixelFormat,
                imageInfo.dataPtr,
                imageInfo.stride
                );

        imageTmp.copyTo(image);


        // ----------------------------------------------------------------------------

        if (!destroySpinImage(hSpinImageConv))
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": unable to release spinImage";
            throw RuntimeError(ERROR_SPIN_RELEASE_SPIN_IMAGE, ssError.str());
        }
    }


    bool CameraDevice_spin::isColor()
    {
        std::vector<spinPixelFormatEnums> cameraPixelFormats = getSupportedPixelFormats_spin();
        std::vector<spinPixelFormatEnums> colorFormats = getAllowedColorPixelFormats_spin();

        bool test = false;
        for (auto format : cameraPixelFormats)
        {
            test = std::find(colorFormats.begin(), colorFormats.end(), format) != colorFormats.end();
            if (test)
            {
                break;
            }
        }
        return test;
    }


    VideoMode CameraDevice_spin::getVideoMode()
    {
        VideoModeList allowedVideoModes = getAllowedVideoModes();
        if (allowedVideoModes.size() != 1)
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": more than one video mode supported (DEVEL)";
            throw RuntimeError(ERROR_SPIN_VIDEOMODE_SUPPORT, ssError.str());
        }
        VideoMode videoMode = allowedVideoModes.front();
        return videoMode;
    }


    FrameRate CameraDevice_spin::getFrameRate()
    {
        VideoMode videoMode = getVideoMode();
        FrameRateList allowedFrameRates = getAllowedFrameRates(videoMode);
        if (allowedFrameRates.size() != 1)
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": more than one framerate supported (DEVEL)";
            throw RuntimeError(ERROR_SPIN_FRAMERATE_SUPPORT, ssError.str());
        }
        FrameRate frameRate = allowedFrameRates.front();
        return frameRate;
    }


    ImageMode CameraDevice_spin::getImageMode()
    {
        ImageModeList allowedModeList = getAllowedImageModes();
        if (allowedModeList.size() != 1)
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": more than one imagemode supported (DEVEL)";
            throw RuntimeError(ERROR_SPIN_FRAMERATE_SUPPORT, ssError.str());
        }
        ImageMode mode = allowedModeList.front();
        return mode;
    }


    VideoModeList CameraDevice_spin::getAllowedVideoModes()
    {
        // Note:
        // --------------------------------------------------------------------
        // Spinnaker SDK doesn't really have the same concept of VideoModes as
        // FlyCapture2 and libdc1394 so we fake it.
        // --------------------------------------------------------------------

        VideoModeList allowedVideoModes = {VIDEOMODE_FORMAT7};
        return allowedVideoModes;

    }


    FrameRateList CameraDevice_spin::getAllowedFrameRates(VideoMode vidMode)
    {
        FrameRateList allowedFrameRates = {};
        if (vidMode == VIDEOMODE_FORMAT7)
        {
            allowedFrameRates.push_back(FRAMERATE_FORMAT7);
        }
        return allowedFrameRates;

    }


    ImageModeList CameraDevice_spin::getAllowedImageModes()
    {
        // Note:
        // -------------------------------------------------------------------
        // Spinnaker SDK doesn't really have ImageModes like FlyCapture2 and
        // libdc1394 so we fake it. Current only support IMAGEMODE_0, but we
        // can add synthetic image modes using binning.
        // -------------------------------------------------------------------
        ImageModeList allImageModes = {IMAGEMODE_0};
        return allImageModes;
    }


    PropertyInfo CameraDevice_spin::getPropertyInfo(PropertyType propType)
    {
        PropertyInfo propInfo;
        propInfo.type = propType;

        if (isSpinSupportedPropertyType(propType))
        {
            if (getPropertyInfoDispatchMap_.count(propType) > 0)
            {
                propInfo = getPropertyInfoDispatchMap_[propType](this);
            }
        }

        return propInfo;
    }


    Property CameraDevice_spin::getProperty(PropertyType propType)
    {
        Property prop;
        prop.type = propType;

        if (isSpinSupportedPropertyType(propType))
        {
            if (getPropertyDispatchMap_.count(propType) > 0)
            {
                prop = getPropertyDispatchMap_[propType](this);
            }
        }

        return prop;
    }


    void CameraDevice_spin::setProperty(Property prop)
    {
        std::string settableMsg("");
        bool isSettable = isPropertySettable(prop.type, settableMsg);

        if (!isSettable)
        {
            throw RuntimeError(ERROR_SPIN_PROPERTY_NOT_SETTABLE, settableMsg);
        }

        setPropertyDispatchMap_[prop.type](this,prop);
    }



    bool CameraDevice_spin::isPropertySettable(PropertyType propType, std::string &msg)
    {
        if (!isSpinSupportedPropertyType(propType))
        {
            msg = std::string("PropertyType is not supported by Spinnaker Backend");
            return false;
        }


        if (setPropertyDispatchMap_.count(propType) <= 0)
        {
            msg = std::string("PropertyType is not is setter dispatch map");
            return false;
        }

        PropertyInfo propInfo = getPropertyInfo(propType);
        // -------------------------------------
        // DEBUG
        // -------------------------------------
        // propInfo.present = true;
        // -------------------------------------
        if (!propInfo.present)
        {
            msg = std::string("PropertyType is not present");
            return false;
        }
        return true;
    }

    Format7Settings CameraDevice_spin::getFormat7Settings()
    {
        Format7Settings settings;

        settings.mode = getImageMode();

        IntegerNode_spin offsetXNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("OffsetX");
        settings.offsetX = (unsigned int)(offsetXNode.value());

        IntegerNode_spin offsetYNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("OffsetY");
        settings.offsetY = (unsigned int)(offsetYNode.value());

        IntegerNode_spin widthNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("Width");
        settings.width = (unsigned int)(widthNode.value());

        IntegerNode_spin heightNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("Height");
        settings.height = (unsigned int)(heightNode.value());

        settings.pixelFormat = getPixelFormat();

        //std::cout << __PRETTY_FUNCTION__ << std::endl;
        //settings.print();

        return settings;
    }



    Format7Info CameraDevice_spin::getFormat7Info(ImageMode imgMode)
    {
        Format7Info format7Info;

        format7Info.mode = imgMode;
        format7Info.supported = isSupported(imgMode);

        if (format7Info.supported)
        {
            IntegerNode_spin widthNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("Width");
            format7Info.maxWidth = (unsigned int)(widthNode.maxValue());
            format7Info.imageHStepSize = (unsigned int)(widthNode.increment());

            IntegerNode_spin heightNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("Height");
            format7Info.maxHeight = (unsigned int)(heightNode.maxValue());
            format7Info.imageVStepSize = (unsigned int)(heightNode.increment());

            IntegerNode_spin offsetXNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("OffsetX");
            format7Info.offsetHStepSize = (unsigned int)(offsetXNode.increment());

            IntegerNode_spin offsetYNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("OffsetY");
            format7Info.offsetVStepSize = (unsigned int)(offsetYNode.increment());

        }

        std::cout << __FUNCTION__ << std::endl;
        //format7Info.print();

        return format7Info;
    }


    bool CameraDevice_spin::validateFormat7Settings(Format7Settings settings)
    {
        bool ok = true;

        if (!isSupported(settings.mode))
        {
            ok = false;
        }

        Format7Info info = getFormat7Info(settings.mode);

        if ((settings.width + settings.offsetX) > info.maxWidth)
        {
            ok = false;
        }
        if ((settings.height + settings.offsetY) > info.maxHeight)
        {
            ok = false;
        }
        if (settings.width%info.imageHStepSize != 0)
        {
            ok = false;
        }
        if (settings.height%info.imageVStepSize != 0)
        {
            ok = false;
        }
        if (settings.offsetX%info.offsetHStepSize != 0)
        {
            ok = false;
        }
        if (settings.offsetY%info.offsetVStepSize != 0)
        {
            ok = false;
        }
        if (!isSupportedPixelFormat(settings.pixelFormat, settings.mode))
        {
            ok = false;
        }

        //std::cout << "validate ok = " << ok << std::endl;

        return ok;
    }


    void CameraDevice_spin::setFormat7Configuration(Format7Settings settings, float percentSpeed)
    {
        bool ok = validateFormat7Settings(settings);
        if (!ok)
        {
            return;
        }

        // -----------------------------
        // NOT IMPLEMENTED set ImageMode
        // -----------------------------

        EnumNode_spin pixelFormatNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("PixelFormat");
        spinPixelFormatEnums pixelFormat_spin = convertPixelFormat_to_spin(settings.pixelFormat);
        pixelFormatNode.setEntryByValue(pixelFormat_spin);

        IntegerNode_spin widthNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("Width");
        widthNode.setValue(int64_t(settings.width));

        IntegerNode_spin heightNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("Height");
        heightNode.setValue(int64_t(settings.height));

        IntegerNode_spin offsetXNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("OffsetX");
        offsetXNode.setValue(int64_t(settings.offsetX));

        IntegerNode_spin offsetYNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("OffsetY");
        offsetYNode.setValue(int64_t(settings.offsetY));

        std::cout << __FUNCTION__ << std::endl;
        //settings.print();

    }


    PixelFormatList CameraDevice_spin::getListOfSupportedPixelFormats(ImageMode imgMode)
    {
        PixelFormatList pixelFormatList;
        if (isSupported(imgMode))
        {
            std::vector<spinPixelFormatEnums> pixelFormatVec_spin = getSupportedPixelFormats_spin();
            for (auto format_spin : pixelFormatVec_spin)
            {
                if (isAllowedPixelFormat_spin(format_spin))
                {
                    pixelFormatList.push_back(convertPixelFormat_from_spin(format_spin));
                }
            }
        }
        return pixelFormatList;
    }


    bool CameraDevice_spin::isSupported(VideoMode vidMode, FrameRate frmRate)
    {
        VideoModeList allowedVideoModes = getAllowedVideoModes();
        FrameRateList allowedFrameRates = getAllowedFrameRates(vidMode);
        bool videoModeFound = (std::find(allowedVideoModes.begin(), allowedVideoModes.end(), vidMode) != allowedVideoModes.end());
        bool frameRateFound = (std::find(allowedFrameRates.begin(), allowedFrameRates.end(), frmRate) != allowedFrameRates.end());
        return (videoModeFound && frameRateFound);
    }


    bool CameraDevice_spin::isSupported(ImageMode imgMode)
    {
        ImageModeList allowedModes = getAllowedImageModes();
        bool found = (std::find(allowedModes.begin(), allowedModes.end(), imgMode) != allowedModes.end());
        return found;
    }


    size_t CameraDevice_spin::getNumberOfImageMode()
    {
        return getAllowedImageModes().size();
    }


    //void CameraDevice_spin::setFormat7ImageMode(ImageMode imgMode)
    //{
    //    // -------------------------------------------------
    //    // TO DO ...
    //    // -------------------------------------------------
    //}


    void CameraDevice_spin::setTriggerInternal()
    {

        EnumNode_spin triggerModeNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("TriggerMode");
        if (triggerModeNode.isAvailable() && triggerModeNode.isWritable())
        {
            triggerModeNode.setEntryBySymbolic("Off");
        }
        else
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": triggerModeNode is not available or writable";
            throw RuntimeError(ERROR_SPIN_SET_TRIGGER_INTERNAL, ssError.str());
        }

        EnumNode_spin triggerSourceNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("TriggerSource");
        if (triggerSourceNode.isAvailable() && triggerSourceNode.isWritable())
        {
            triggerSourceNode.setEntryBySymbolic("Software");
            triggerType_ = TRIGGER_INTERNAL;
        }
        else
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": triggerSourceNode is not available or writable";
            throw RuntimeError(ERROR_SPIN_SET_TRIGGER_INTERNAL, ssError.str());
        }
    }


    void CameraDevice_spin::setTriggerExternal()
    {

        EnumNode_spin triggerModeNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("TriggerMode");
        if (triggerModeNode.isAvailable() && triggerModeNode.isWritable())
        {
            triggerModeNode.setEntryBySymbolic("Off");
        }
        else
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": triggerModeNode is not available or writable";
            throw RuntimeError(ERROR_SPIN_SET_TRIGGER_EXTERNAL, ssError.str());
        }

        EnumNode_spin triggerSourceNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("TriggerSource");
        if (triggerSourceNode.isAvailable() && triggerSourceNode.isWritable())
        {
            triggerSourceNode.setEntryBySymbolic("Line0");
        }
        else
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": triggerSourceNode is not available or writable";
            throw RuntimeError(ERROR_SPIN_SET_TRIGGER_EXTERNAL, ssError.str());
        }

        EnumNode_spin triggerActivationNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("TriggerActivation");
        if (triggerActivationNode.isAvailable() && triggerActivationNode.isWritable())
        {
            triggerActivationNode.setEntryBySymbolic("RisingEdge");
        }
        else
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": triggerActivationNode is not available or writable";
            throw RuntimeError(ERROR_SPIN_SET_TRIGGER_EXTERNAL, ssError.str());
        }


        if (triggerModeNode.isAvailable() && triggerModeNode.isWritable())
        {
            triggerModeNode.setEntryBySymbolic("On");
            triggerType_ = TRIGGER_EXTERNAL;
        }
        else
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": triggerModeNode is not available or writable";
            throw RuntimeError(ERROR_SPIN_SET_TRIGGER_EXTERNAL, ssError.str());
        }
    }


    TriggerType CameraDevice_spin::getTriggerType()
    {
        std::string modeSymb;
        EnumNode_spin triggerModeNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("TriggerMode");
        if (triggerModeNode.isAvailable() && triggerModeNode.isReadable())
        {
            EntryNode_spin entryNode = triggerModeNode.currentEntry();
            if (entryNode.isAvailable() && entryNode.isReadable())
            {
                modeSymb = entryNode.symbolic();
            }
            else
            {
                std::stringstream ssError;
                ssError << __FUNCTION__;
                ssError << ": triggerModeNode current entryNode is not available or readable";
                throw RuntimeError(ERROR_SPIN_GET_TRIGGER_TYPE, ssError.str());
            }
        }
        else
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": triggerModeNode is not available or readable";
            throw RuntimeError(ERROR_SPIN_GET_TRIGGER_TYPE, ssError.str());
        }

        return (modeSymb == std::string("On")) ? TRIGGER_EXTERNAL : TRIGGER_INTERNAL;
    }


    TimeStamp CameraDevice_spin::getImageTimeStamp()
    {
        return timeStamp_;
    }


    std::string CameraDevice_spin::getVendorName()
    {
        return cameraInfo_.vendorName();
    }


    std::string CameraDevice_spin::getModelName()
    {
       return cameraInfo_.modelName();
    }


    std::string CameraDevice_spin::toString()
    {
        return cameraInfo_.toString();
    }


    void CameraDevice_spin::printGuid()
    {
        guid_.printValue();
    }


    void CameraDevice_spin::printInfo()
    {
        std::cout << toString();
    }


    //
    //// Private methods
    //// -------------------------------------------------------------------------


    bool CameraDevice_spin::grabImageCommon(std::string &errMsg)
    {

        spinError err = SPINNAKER_ERR_SUCCESS;
        uInt32 read_buffer = 0, read_ondemand = 0;
        TimeStamp pc_ts;
        int64_t pc_ts1;

        if (!capturing_)
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": unable to grab Image - not capturing";
            errMsg = ssError.str();
            return false;
        }

        if (!releaseSpinImage(hSpinImage_))
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": unable to release existing spinImage";
            errMsg = ssError.str();
            return false;
        }

        imageOK_ = false;

        // Get next image from camera
        if (triggerType_ == TRIGGER_INTERNAL)
        {
            err = spinCameraGetNextImage(hCamera_, &hSpinImage_); // This fixes memory leak ??? why??
        }
        else
        {           
            // Note, 2nd arg > 0 to help reduce effect of slow memory leak
            err = spinCameraGetNextImageEx(hCamera_, 10, &hSpinImage_);       
        }

        if (err != SPINNAKER_ERR_SUCCESS)
        {
            
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": unable to get next image";
            errMsg = ssError.str();
            //std::cout << "image error" << std::endl;
            return false;
        }

        // Check to see if image is incomplete
        bool8_t isIncomplete = False;
        err = spinImageIsIncomplete(hSpinImage_, &isIncomplete);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": unable to determine if image is complete";
            errMsg = ssError.str();
            std::cout << "image incomplete" << std::endl;
            return false;
        }

        if (isIncomplete==True)
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": image is incomplete";
            errMsg = ssError.str();
            std::cout << "incomp" << std::endl;
            return false;
        }

        imageOK_ = true;
        numFrameskip++;
        
        
        if (nidaq_task_ != nullptr && cameraNumber_ == 0 
            && numFrameskip <= 100001) {
            DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_trigger_in, 10.0, &read_buffer, NULL));
        }

        if(nidaq_task_ != nullptr && numFrameskip <= 100001){
            
            DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_grab_in, 10.0, &read_ondemand, NULL));
            if (numFrameskip > 2) {
                if (cameraNumber_ == 0)
                    time_stamp3[numFrameskip - 3][0] = read_buffer;
                else
                    time_stamp3[numFrameskip - 3][0] = 0;
                time_stamp3[numFrameskip - 3][1] = read_ondemand;
            }
        }

        
        if (numFrameskip == 100001)
        {
            std::string filename = "imagegrab_cam2sys" + std::to_string(cameraNumber_) + ".csv";
            gettime->write_time_2d<uInt32>(filename, 100000, time_stamp3);
            
        }
        //pc_ts = gettime->getPCtime();
        //pc_ts1 = pc_ts.seconds*1000000 + pc_ts.microSeconds;
        updateTimeStamp();
        //std::cout << "timeStamp_ns_           = " << timeStamp_ns_ << std::endl;
        //std::cout << "timeStamp_.seconds      = " << timeStamp_.seconds << std::endl;
        //std::cout << "timeStamp_.microSeconds = " << timeStamp_.microSeconds << std::endl;
        return true;
    }


    bool CameraDevice_spin::releaseSpinImage(spinImage &hImage)
    {
        bool rval = true;
        if (hImage != nullptr)
        {
            //std::cout << "release" << std::endl;
            spinError err = spinImageRelease(hImage);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                rval = false;
            }
            else
            {
                hImage = nullptr;
            }
        }
        return rval;
    }

    bool CameraDevice_spin::destroySpinImage(spinImage &hImage)
    {
        bool rval = true;
        if (hImage != nullptr)
        {
            //std::cout << "destroy" << std::endl;
            spinError err = spinImageDestroy(hImage);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                rval = false;
            }
            //else
            //{
            //    hImage = nullptr;
            //}
        }
        return rval;
    }



    void CameraDevice_spin::setupTimeStamping()
    {

        std::cout << "DEBUG: " << __FUNCTION__ << std::endl;

        // Enable chunk mode
        BoolNode_spin chunkModeActiveNode = nodeMapCamera_.getNodeByName<BoolNode_spin>("ChunkModeActive");
        if (chunkModeActiveNode.isAvailable())
        {
            chunkModeActiveNode.setValue(true);
        }

        // Get chunk mode selector and  set entry to Timestamp
        std::cout << "DEBUG: set ChunkSelector begin " << std::endl;

        std::cout << "DEBUG: get ChunkSelector node " << std::endl;

        EnumNode_spin chunkSelectorNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("ChunkSelector");

        std::cout << "DEBUG: have ChunkSelector node " << std::endl;

        if (chunkSelectorNode.isAvailable())
        {
            std::cout << "DEBUG: ChunkSelector available " << std::endl;
            /*std::ofstream entries_file;
            entries_file.open("chuckselector_entries.txt");
            entries_file << "DEBUG: ChunkSelector entries begin " << std::endl;
            for (auto entry : chunkSelectorNode.entries())
            {
                entries_file << entry.toString();
            }
            entries_file << "DEBUG: ChunkSelector entries end " << std::endl;
            entries_file.close();*/
            chunkSelectorNode.setEntryBySymbolic("Timestamp");
        }
        else
        {
            std::cout << "DEBUG: ChunkSelector not available " << std::endl;
        }

        std::cout << "DEBUG: set ChunkSelector end " << std::endl;

        // Enable timestamping
        BoolNode_spin timeStampEnableNode = nodeMapCamera_.getNodeByName<BoolNode_spin>("ChunkEnable");
        if (timeStampEnableNode.isAvailable())
        {
            timeStampEnableNode.setValue(true);
        }

        std::cout << "DEBUG: " << __FUNCTION__ << " end" << std::endl;

    }


    void CameraDevice_spin::updateTimeStamp()
    {
        spinError err = spinImageChunkDataGetIntValue(hSpinImage_, "ChunkTimestamp", &timeStamp_ns_);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": unable to timestamp from image chunk data, error = " << err;
            throw RuntimeError(ERROR_SPIN_CHUNKDATA_TIMESTAMP, ssError.str());
        }

        int64_t seconds = timeStamp_ns_/INT64_C(1000000000);
        int64_t microSeconds = timeStamp_ns_/INT64_C(1000) - INT64_C(1000000)*seconds;

        timeStamp_.seconds = (unsigned long long)(seconds);
        timeStamp_.microSeconds = (unsigned int)(microSeconds);
    }


    void CameraDevice_spin::initCounter()
    {


        EnumNode_spin CounterSelector = nodeMapCamera_.getNodeByName<EnumNode_spin>("CounterSelector");
        if(CounterSelector.isAvailable() && CounterSelector.isWritable())
        {
            CounterSelector.setEntryBySymbolic("Counter0");

        }else{

            std::cout << "DEBUG: CounterEventSource not available " << std::endl;

        }

        //source event to increment the counter
        EnumNode_spin CounterEventSource = nodeMapCamera_.getNodeByName<EnumNode_spin>("CounterEventSource");
        if(CounterEventSource.isAvailable() && CounterEventSource.isWritable())
        {
            CounterEventSource.setEntryBySymbolic("Line0");

        }else{

            std::cout << "DEBUG: CounterEventSource not available " << std::endl;

        }


        EnumNode_spin CounterEventActivation = nodeMapCamera_.getNodeByName<EnumNode_spin>("CounterEventActivation");
        if(CounterEventActivation.isAvailable() && CounterEventActivation.isWritable())
        {
            CounterEventActivation.setEntryBySymbolic("RisingEdge");

        }else{

            std::cout << "DEBUG: CounterEventSource not available " << std::endl;

        }


        // source event to start the counter
        EnumNode_spin CounterTriggerSource= nodeMapCamera_.getNodeByName<EnumNode_spin>("CounterTriggerSource");
        if(CounterTriggerSource.isAvailable() && CounterTriggerSource.isWritable())
        {
            CounterTriggerSource.setEntryBySymbolic("Line0");

        }else{

            std::cout << "DEBUG: CounterTriggerSource not available " << std::endl;

        }


        EnumNode_spin CounterTriggerActivation = nodeMapCamera_.getNodeByName<EnumNode_spin>("CounterTriggerActivation");
        if(CounterTriggerActivation.isAvailable() && CounterTriggerActivation.isWritable())
        {
            CounterTriggerActivation.setEntryBySymbolic("RisingEdge");

        }else{

            std::cout << "DEBUG: CounterEventSource not available " << std::endl;

        }

        //set the counter duration
        IntegerNode_spin CounterDuration = nodeMapCamera_.getNodeByName<IntegerNode_spin>("CounterDuration");
        if(CounterDuration.isAvailable())
        {
            CounterDuration.setValue(1000);

        }else{

            std::cout << "DEBUG: CounterDuration not available " << std::endl;

        }


    }


    TimeStamp CameraDevice_spin::getDeviceTimeStamp()
    {

        //Bandwidth
        //IntegerNode_spin hDeviceThroughput = nodeMapCamera_.getNodeByName<IntegerNode_spin>("DeviceMaxThorughput");
        //std::cout << hDeviceThroughput.value() << std::endl;
        //IntegerNode_spin hLinkThroughput = nodeMapCamera_.getNodeByName<IntegerNode_spin>("DeviceLinkThroughputLimit");
        //std::cout << hLinkThroughput.value() << std::endl;


        TimeStamp ts;
        // Retrieve TimestampLatch
        CommandNode_spin hTimestampLatch = nodeMapCamera_.getNodeByName<CommandNode_spin>("TimestampLatch");

        //Execute Command
        if(hTimestampLatch.isAvailable())
        {
            spinCommandExecute(hTimestampLatch.handle());

        }else{

            std::cout << "DEBUG: TimestampLatch not available " << std::endl;
        }

        //Increment Timer
        /*IntegerNode_spin hTimestampIncrementValue = nodeMapCamera_.getNodeByName<IntegerNode_spin>("TimestampIncrement");
        if(hTimestampIncrementValue.isAvailable())
        {
            timeStamp_inc = (int64_t)hTimestampIncrementValue.value();

        }else{

            std::cout << "DEBUG: TimestamplatchValue not available " << std::endl;
        }*/


        //Get TimeStampLatch Value
        IntegerNode_spin hTimestampLatchValue = nodeMapCamera_.getNodeByName<IntegerNode_spin>("TimestampLatchValue");
        if(hTimestampLatchValue.isAvailable())
        {
            timeStamp_cam = (int64_t)hTimestampLatchValue.value();
            ts.seconds = timeStamp_cam/INT64_C(1000000000);
            ts.microSeconds = uint64_t(timeStamp_cam/INT64_C(1000) - INT64_C(1000000)*ts.seconds);


        }else{

            std::cout << "DEBUG: TimestamplatchValue not available " << std::endl;
        }


        return ts;

    }

    // Get PropertyInfo methods
    // --------------------------

    PropertyInfo CameraDevice_spin::getPropertyInfoBrightness()
    {
        FloatNode_spin blackLevelNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("BlackLevel");

        PropertyInfo propInfo;
        propInfo.type = PROPERTY_TYPE_BRIGHTNESS;
        propInfo.present = blackLevelNode.isAvailable();

        if (propInfo.present)
        {
            propInfo.autoCapable = false;
            propInfo.manualCapable = true;
            propInfo.absoluteCapable = true;
            propInfo.onePushCapable = false;
            propInfo.onOffCapable = false;
            propInfo.readOutCapable = false;
            propInfo.minValue = blackLevelNode.minIntValue();
            propInfo.maxValue = blackLevelNode.maxIntValue();
            propInfo.minAbsoluteValue = static_cast<float>(blackLevelNode.minValue());
            propInfo.maxAbsoluteValue = static_cast<float>(blackLevelNode.maxValue());
            propInfo.haveUnits = !blackLevelNode.unit().empty();
            propInfo.units = blackLevelNode.unit();
            propInfo.unitsAbbr = blackLevelNode.unit();
        }

        return propInfo;
    }

    PropertyInfo CameraDevice_spin::getPropertyInfoGamma()
    {
        FloatNode_spin gammaNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("Gamma");
        BoolNode_spin gammaEnableNode = nodeMapCamera_.getNodeByName<BoolNode_spin>("GammaEnable");

        PropertyInfo propInfo;
        propInfo.type = PROPERTY_TYPE_GAMMA;
        propInfo.present = gammaEnableNode.isAvailable();

        if (propInfo.present)
        {
            propInfo.autoCapable = false;
            propInfo.manualCapable = true;
            propInfo.absoluteCapable = true;
            propInfo.onePushCapable = false;
            propInfo.onOffCapable = gammaEnableNode.isAvailable() && gammaEnableNode.isWritable();
            propInfo.readOutCapable = false;
            if (gammaNode.isAvailable())
            {
                propInfo.minValue = gammaNode.minIntValue();
                propInfo.maxValue = gammaNode.maxIntValue();
                propInfo.minAbsoluteValue = static_cast<float>(gammaNode.minValue());
                propInfo.maxAbsoluteValue = static_cast<float>(gammaNode.maxValue());
                propInfo.haveUnits = !gammaNode.unit().empty();
                propInfo.units =  gammaNode.unit();
                propInfo.unitsAbbr = gammaNode.unit();
            }

        }

        return propInfo;
    }


    PropertyInfo CameraDevice_spin::getPropertyInfoShutter()
    {

        EnumNode_spin exposureAutoNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("ExposureAuto");
        FloatNode_spin exposureTimeNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("ExposureTime");

        PropertyInfo propInfo;
        propInfo.type = PROPERTY_TYPE_SHUTTER;
        propInfo.present = exposureAutoNode.isAvailable();

        if (propInfo.present)
        {
            if (exposureAutoNode.isReadable())
            {
                propInfo.autoCapable = exposureAutoNode.hasEntrySymbolic("Continuous");
                propInfo.manualCapable = exposureAutoNode.hasEntrySymbolic("Off");
                propInfo.onePushCapable = exposureAutoNode.hasEntrySymbolic("Once");
            }
            propInfo.absoluteCapable = true;
            propInfo.onOffCapable = false;
            propInfo.readOutCapable = false;
            if (exposureTimeNode.isAvailable() && exposureTimeNode.isReadable())
            {
                propInfo.minValue = exposureTimeNode.minIntValue();
                propInfo.maxValue = exposureTimeNode.maxIntValue();
                //propInfo.minAbsoluteValue = static_cast<float>(std::max(exposureTimeNode.minValue(), CameraDevice_spin::MinAllowedShutterUs));
                //propInfo.maxAbsoluteValue = static_cast<float>(std::min(exposureTimeNode.maxValue(), CameraDevice_spin::MaxAllowedShutterUs));
                propInfo.haveUnits = !exposureTimeNode.unit().empty();
                propInfo.units =  exposureTimeNode.unit();
                propInfo.unitsAbbr = exposureTimeNode.unit();
            }
        }

        return propInfo;
    }



    PropertyInfo CameraDevice_spin::getPropertyInfoGain()
    {
        EnumNode_spin gainAutoNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("GainAuto");
        FloatNode_spin gainNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("Gain");

        PropertyInfo propInfo;
        propInfo.type = PROPERTY_TYPE_GAIN;
        propInfo.present = gainAutoNode.isAvailable();

        if (propInfo.present)
        {
            if (gainAutoNode.isReadable())
            {
                propInfo.autoCapable = gainAutoNode.hasEntrySymbolic("Continuous");
                propInfo.manualCapable = gainAutoNode.hasEntrySymbolic("Off");
                propInfo.onePushCapable = gainAutoNode.hasEntrySymbolic("Once");
            }
            propInfo.absoluteCapable = true;
            propInfo.onOffCapable = false;
            propInfo.readOutCapable = false;
            if (gainNode.isAvailable() && gainNode.isReadable())
            {
                propInfo.minValue = gainNode.minIntValue();
                propInfo.maxValue = gainNode.maxIntValue();
                propInfo.minAbsoluteValue = static_cast<float>(gainNode.minValue());
                propInfo.maxAbsoluteValue = static_cast<float>(gainNode.maxValue());
                propInfo.haveUnits = !gainNode.unit().empty();
                propInfo.units =  gainNode.unit();
                propInfo.unitsAbbr = gainNode.unit();
            }
        }

        return propInfo;
    }


    PropertyInfo CameraDevice_spin::getPropertyInfoTriggerDelay()
    {
        FloatNode_spin triggerDelayNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("TriggerDelay");

        PropertyInfo propInfo;
        propInfo.type = PROPERTY_TYPE_TRIGGER_DELAY;
        propInfo.present = triggerDelayNode.isAvailable();

        if (propInfo.present)
        {
            propInfo.autoCapable = false;
            propInfo.manualCapable = true;
            propInfo.absoluteCapable = true;
            propInfo.onePushCapable = false;
            propInfo.onOffCapable = false;
            propInfo.readOutCapable = false;
            propInfo.minValue = triggerDelayNode.minIntValue();
            propInfo.maxValue = triggerDelayNode.maxIntValue();
            propInfo.minAbsoluteValue = static_cast<float>(triggerDelayNode.minValue());
            propInfo.maxAbsoluteValue = static_cast<float>(triggerDelayNode.maxValue());
            propInfo.haveUnits = !triggerDelayNode.unit().empty();
            propInfo.units =  triggerDelayNode.unit();
            propInfo.unitsAbbr = triggerDelayNode.unit();
        }

        return propInfo;
    }


    PropertyInfo CameraDevice_spin::getPropertyInfoFrameRate()
    {
        FloatNode_spin frameRateNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("AcquisitionFrameRate");

        PropertyInfo propInfo;
        propInfo.type = PROPERTY_TYPE_FRAME_RATE;
        propInfo.present = frameRateNode.isAvailable();

        if (propInfo.present)
        {
            propInfo.autoCapable = false;
            propInfo.manualCapable = true;
            propInfo.absoluteCapable = true;
            propInfo.onePushCapable = false;
            propInfo.onOffCapable = false;
            propInfo.readOutCapable = false;
            propInfo.minValue = frameRateNode.minIntValue();
            propInfo.maxValue = frameRateNode.maxIntValue();
            propInfo.minAbsoluteValue =  static_cast<float>(frameRateNode.minValue());
            propInfo.maxAbsoluteValue = static_cast<float>(frameRateNode.maxValue());
            propInfo.haveUnits = !frameRateNode.unit().empty();
            propInfo.units =  frameRateNode.unit();
            propInfo.unitsAbbr = frameRateNode.unit();
        }

        return propInfo;
    }



    PropertyInfo CameraDevice_spin::getPropertyInfoTemperature()
    {
        FloatNode_spin tempNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("DeviceTemperature");

        PropertyInfo propInfo;
        propInfo.type = PROPERTY_TYPE_TEMPERATURE;
        propInfo.present = tempNode.isAvailable();

        if (propInfo.present)
        {
            propInfo.autoCapable = false;
            propInfo.manualCapable = false;
            propInfo.absoluteCapable = true;
            propInfo.onePushCapable = false;
            propInfo.onOffCapable = false;
            propInfo.readOutCapable = false;
            propInfo.minValue = tempNode.minIntValue();
            propInfo.maxValue = tempNode.maxIntValue();
            propInfo.minAbsoluteValue = static_cast<float>(tempNode.minValue());
            propInfo.maxAbsoluteValue = static_cast<float>(tempNode.maxValue());
            propInfo.haveUnits = !tempNode.unit().empty();
            propInfo.units =  tempNode.unit();
            propInfo.unitsAbbr = tempNode.unit();
        }

        return propInfo;
    }


    PropertyInfo CameraDevice_spin::getPropertyInfoTriggerMode()
    {
        // Not implemented - dummy method
        PropertyInfo propInfo;
        propInfo.type = PROPERTY_TYPE_TRIGGER_MODE;
        propInfo.present = true;
        return propInfo;
    }


    // Get Property methods
    // ------------------------

    Property CameraDevice_spin::getPropertyBrightness()
    {
        FloatNode_spin blackLevelNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("BlackLevel");

        Property prop;
        prop.type = PROPERTY_TYPE_BRIGHTNESS;
        prop.present = blackLevelNode.isAvailable();

        if(prop.present)
        {
            prop.absoluteControl = true;
            prop.onePush = false;
            prop.on = true;
            prop.autoActive = false;
            prop.value = blackLevelNode.intValue();
            prop.valueA = 0;
            prop.valueB = 0;
            prop.absoluteValue = float(blackLevelNode.value());
        }

        return prop;
    }


    Property CameraDevice_spin::getPropertyGamma()
    {
        FloatNode_spin gammaNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("Gamma");
        BoolNode_spin gammaEnableNode = nodeMapCamera_.getNodeByName<BoolNode_spin>("GammaEnable");

        Property prop;
        prop.type = PROPERTY_TYPE_GAMMA;
        prop.present = gammaEnableNode.isAvailable();

        if (prop.present)
        {
            prop.absoluteControl = true;
            prop.onePush = false;
            prop.autoActive = false;
            prop.on = gammaEnableNode.value();
            if (gammaNode.isAvailable())
            {
                prop.value = gammaNode.intValue();
                prop.absoluteValue = static_cast<float>(gammaNode.value());
            }
        }

        return prop;
    }


    Property CameraDevice_spin::getPropertyShutter()
    {
        EnumNode_spin exposureAutoNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("ExposureAuto");
        FloatNode_spin exposureTimeNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("ExposureTime");

        Property prop;
        prop.type = PROPERTY_TYPE_SHUTTER;
        prop.present = exposureAutoNode.isAvailable();

        if (prop.present)
        {
            prop.absoluteControl = true;
            prop.onePush = false;
            prop.autoActive = false;
            if (exposureAutoNode.isReadable())
            {
                EntryNode_spin autoEntry = exposureAutoNode.currentEntry();
                prop.autoActive = autoEntry.isSymbolicValueEqualTo("Continuous");
            }
            if (exposureTimeNode.isAvailable() && exposureTimeNode.isReadable())
            {
                prop.value = exposureTimeNode.intValueWithLimits(MinAllowedShutterUs, MaxAllowedShutterUs);
                prop.absoluteValue = static_cast<float>(exposureTimeNode.value());
            }
        }

        return prop;
    }


    Property CameraDevice_spin::getPropertyGain()
    {
        EnumNode_spin gainAutoNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("GainAuto");
        FloatNode_spin gainNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("Gain");

        Property prop;
        prop.type = PROPERTY_TYPE_GAIN;
        prop.present = gainAutoNode.isAvailable();

        if (prop.present)
        {
            prop.absoluteControl = true;
            prop.onePush = false;
            prop.autoActive = false;
            if (gainAutoNode.isReadable())
            {
                EntryNode_spin autoEntry = gainAutoNode.currentEntry();
                prop.autoActive = autoEntry.isSymbolicValueEqualTo("Continuous");
            }
            if (gainNode.isAvailable() && gainNode.isReadable())
            {
                prop.value = gainNode.intValue();
                prop.absoluteValue = static_cast<float>(gainNode.value());
            }

        }

        return prop;
    }


    Property CameraDevice_spin::getPropertyTriggerDelay()
    {
        FloatNode_spin triggerDelayNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("TriggerDelay");

        Property prop;
        prop.type = PROPERTY_TYPE_TRIGGER_DELAY;
        prop.present = triggerDelayNode.isAvailable();

        if (prop.present)
        {
            prop.absoluteControl = true;
            prop.onePush = false;
            prop.autoActive = false;
            prop.value = triggerDelayNode.intValue();
            prop.valueA = 0;
            prop.valueB = 0;
            prop.absoluteValue = static_cast<float>(triggerDelayNode.value());
        }

        return prop;
    }



    Property CameraDevice_spin::getPropertyFrameRate()
    {
        FloatNode_spin frameRateNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("AcquisitionFrameRate");

        Property prop;
        prop.type = PROPERTY_TYPE_FRAME_RATE;
        prop.present = frameRateNode.isAvailable();

        if (prop.present)
        {
            prop.absoluteControl = true;
            prop.onePush = false;
            prop.autoActive = false;
            prop.value = frameRateNode.intValue();
            prop.valueA = 0;
            prop.valueB = 0;
            prop.absoluteValue = static_cast<float>(frameRateNode.value());
        }

        return prop;
    }




    Property CameraDevice_spin::getPropertyTemperature()
    {
        FloatNode_spin tempNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("DeviceTemperature");

        Property prop;
        prop.type = PROPERTY_TYPE_TEMPERATURE;
        prop.present = tempNode.isAvailable();

        if (prop.present)
        {
            prop.absoluteControl = true;
            prop.onePush = false;
            prop.autoActive = false;
            prop.value = tempNode.intValue();
            prop.valueA = 0;
            prop.valueB = 0;
            prop.absoluteValue = static_cast<float>(tempNode.value());
        }

        return prop;
    }


    Property CameraDevice_spin::getPropertyTriggerMode()
    {
        // Not implemented - dummy method
        Property prop;
        prop.type = PROPERTY_TYPE_TRIGGER_MODE;
        prop.present = true;
        return prop;
    }


    // Set Property methods
    // ---------------------

    void CameraDevice_spin::setPropertyBrightness(Property prop)
    {
        FloatNode_spin blackLevelNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("BlackLevel");
        if (blackLevelNode.isWritable())
        {
            if (prop.absoluteControl)
            {
                blackLevelNode.setValue(prop.absoluteValue);
            }
            else
            {
                blackLevelNode.setValueFromInt(prop.value);
            }
        }
    }


    void CameraDevice_spin::setPropertyGamma(Property prop)
    {
        FloatNode_spin gammaNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("Gamma");
        BoolNode_spin gammaEnableNode = nodeMapCamera_.getNodeByName<BoolNode_spin>("GammaEnable");


        if (gammaEnableNode.isAvailable() && gammaEnableNode.isWritable())
        {
                gammaEnableNode.setValue(prop.on);
        }
        if (gammaNode.isAvailable() && gammaNode.isReadable() && gammaNode.isWritable())
        {
            if (prop.absoluteControl)
            {
                gammaNode.setValue(prop.absoluteValue);
            }
            else
            {
                gammaNode.setValueFromInt(prop.value);
            }
        }
    }

    void CameraDevice_spin::setPropertyShutter(Property prop)
    {

        EnumNode_spin exposureAutoNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("ExposureAuto");
        FloatNode_spin exposureTimeNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("ExposureTime");

        if (exposureAutoNode.isAvailable() && exposureAutoNode.isWritable())
        {
            if (prop.onePush)
            {
                // Seems to need to be called more than once for the value to stabilize - not sure why this is.
                for (int i=0; i<AutoOnePushSetCount; i++)
                {
                    exposureAutoNode.setEntryBySymbolic("Once");
                }
                return;
            }

            if (prop.autoActive)
            {
                exposureAutoNode.setEntryBySymbolic("Continuous");
                return;
            }
            else
            {
                exposureAutoNode.setEntryBySymbolic("Off");
            }
        }

        if (exposureTimeNode.isAvailable() && exposureTimeNode.isReadable() && exposureTimeNode.isWritable())
        {
            if (prop.absoluteControl)
            {
                exposureTimeNode.setValue(prop.absoluteValue);
            }
            else
            {
                exposureTimeNode.setValueFromIntWithLimits(prop.value, MinAllowedShutterUs, MaxAllowedShutterUs);
            }
        }
    }


    void CameraDevice_spin::setPropertyGain(Property prop)
    {
        EnumNode_spin gainAutoNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("GainAuto");
        FloatNode_spin gainNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("Gain");

        if (gainAutoNode.isAvailable() && gainAutoNode.isWritable())
        {
            if (prop.onePush)
            {
                // Seems to need to be called more than once for the value to stabilize - not sure why this is.
                for (int i=0; i<AutoOnePushSetCount; i++)
                {
                    gainAutoNode.setEntryBySymbolic("Once");
                }
                return;
            }

            if (prop.autoActive)
            {
                gainAutoNode.setEntryBySymbolic("Continuous");
                return;
            }
            else
            {
                gainAutoNode.setEntryBySymbolic("Off");
            }
        }

        if (gainNode.isAvailable() && gainNode.isReadable() && gainNode.isWritable())
        {
            if (prop.absoluteControl)
            {
                gainNode.setValue(prop.absoluteValue);
            }
            else
            {
                gainNode.setValueFromInt(prop.value);
            }
        }
    }



    void CameraDevice_spin::setPropertyTriggerDelay(Property prop)
    {
        FloatNode_spin triggerDelayNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("TriggerDelay");

        if (triggerDelayNode.isWritable())
        {
            if (prop.absoluteControl)
            {
                triggerDelayNode.setValue(prop.absoluteValue);
            }
            else
            {
                triggerDelayNode.setValueFromInt(prop.value);
            }
        }
    }


    void CameraDevice_spin::setPropertyFrameRate(Property prop)
    {
        FloatNode_spin frameRateNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("AcquisitionFrameRate");

        if (frameRateNode.isWritable())
        {
            if (prop.absoluteControl)
            {
                frameRateNode.setValue(prop.absoluteValue);
            }
            else
            {
                frameRateNode.setValueFromInt(prop.value);
            }
        }
    }


    void CameraDevice_spin::setPropertyTemperature(Property prop)
    {
        // Do nothing
    }


    void CameraDevice_spin::setPropertyTriggerMode(Property prop)
    {
        // Do nothing
    }


    PixelFormat CameraDevice_spin::getPixelFormat()
    {
        spinPixelFormatEnums currPixelFormat_spin = getPixelFormat_spin();
        return convertPixelFormat_from_spin(currPixelFormat_spin);
    }


    bool CameraDevice_spin::isSupportedPixelFormat(PixelFormat pixelFormat, ImageMode imgMode)
    {
        PixelFormatList formatList = getListOfSupportedPixelFormats(imgMode);
        return std::find(std::begin(formatList), std::end(formatList), pixelFormat) != std::end(formatList);
    }


    // spin get methods
    // ---------------

    std::vector<spinPixelFormatEnums> CameraDevice_spin::getSupportedPixelFormats_spin()
    {
        EnumNode_spin pixelFormatNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("PixelFormat");
        std::vector<EntryNode_spin> pixelFormatEntryVec = pixelFormatNode.entries();

        std::vector<spinPixelFormatEnums> pixelFormatValueVec;
        for (auto entry: pixelFormatEntryVec)
        {
            //std::cout << entry.symbolic() << ", " << entry.value() << std::endl;
            pixelFormatValueVec.push_back(spinPixelFormatEnums(entry.value()));
        }
        return pixelFormatValueVec;
    }

    spinPixelFormatEnums CameraDevice_spin::getPixelFormat_spin()
    {
        EnumNode_spin pixelFormatNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("PixelFormat");
        EntryNode_spin currentEntry = pixelFormatNode.currentEntry();
        return spinPixelFormatEnums(currentEntry.value());
    }


    // PropertyInfo, and Property dispatch maps
    // ----------------------------------------

    std::map<PropertyType, std::function<PropertyInfo(CameraDevice_spin*)>> CameraDevice_spin::getPropertyInfoDispatchMap_ =
    {
        {PROPERTY_TYPE_BRIGHTNESS,     std::function<PropertyInfo(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyInfoBrightness)},
        {PROPERTY_TYPE_GAMMA,          std::function<PropertyInfo(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyInfoGamma)},
        {PROPERTY_TYPE_SHUTTER,        std::function<PropertyInfo(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyInfoShutter)},
        {PROPERTY_TYPE_GAIN,           std::function<PropertyInfo(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyInfoGain)},
        {PROPERTY_TYPE_TRIGGER_DELAY,  std::function<PropertyInfo(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyInfoTriggerDelay)},
        {PROPERTY_TYPE_FRAME_RATE,     std::function<PropertyInfo(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyInfoFrameRate)},
        {PROPERTY_TYPE_TEMPERATURE,    std::function<PropertyInfo(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyInfoTemperature)},
        {PROPERTY_TYPE_TRIGGER_MODE,   std::function<PropertyInfo(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyInfoTriggerMode)},
    };


    std::map<PropertyType, std::function<Property(CameraDevice_spin*)>> CameraDevice_spin::getPropertyDispatchMap_ =
    {
        {PROPERTY_TYPE_BRIGHTNESS,     std::function<Property(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyBrightness)},
        {PROPERTY_TYPE_GAMMA,          std::function<Property(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyGamma)},
        {PROPERTY_TYPE_SHUTTER,        std::function<Property(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyShutter)},
        {PROPERTY_TYPE_GAIN,           std::function<Property(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyGain)},
        {PROPERTY_TYPE_TRIGGER_DELAY,  std::function<Property(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyTriggerDelay)},
        {PROPERTY_TYPE_FRAME_RATE,     std::function<Property(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyFrameRate)},
        {PROPERTY_TYPE_TEMPERATURE,    std::function<Property(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyTemperature)},
        {PROPERTY_TYPE_TRIGGER_MODE,   std::function<Property(CameraDevice_spin*)>(&CameraDevice_spin::getPropertyTriggerMode)},
    };


    std::map<PropertyType, std::function<void(CameraDevice_spin*,Property)>> CameraDevice_spin::setPropertyDispatchMap_ =
    {
        {PROPERTY_TYPE_BRIGHTNESS,     std::function<void(CameraDevice_spin*, Property)>(&CameraDevice_spin::setPropertyBrightness)},
        {PROPERTY_TYPE_GAMMA,          std::function<void(CameraDevice_spin*, Property)>(&CameraDevice_spin::setPropertyGamma)},
        {PROPERTY_TYPE_SHUTTER,        std::function<void(CameraDevice_spin*, Property)>(&CameraDevice_spin::setPropertyShutter)},
        {PROPERTY_TYPE_GAIN,           std::function<void(CameraDevice_spin*, Property)>(&CameraDevice_spin::setPropertyGain)},
        {PROPERTY_TYPE_TRIGGER_DELAY,  std::function<void(CameraDevice_spin*, Property)>(&CameraDevice_spin::setPropertyTriggerDelay)},
        {PROPERTY_TYPE_FRAME_RATE,     std::function<void(CameraDevice_spin*, Property)>(&CameraDevice_spin::setPropertyFrameRate)},
        {PROPERTY_TYPE_TEMPERATURE,    std::function<void(CameraDevice_spin*, Property)>(&CameraDevice_spin::setPropertyTemperature)},
        {PROPERTY_TYPE_TRIGGER_MODE,   std::function<void(CameraDevice_spin*, Property)>(&CameraDevice_spin::setPropertyTriggerMode)},
    };

    // DEVEL
    // ----------------------------------------------------------------------------------------------------------------------------------
    void CameraDevice_spin::develExpProps()
    {
        if (!connected_)
        {
            std::cout << "camera is not connected - can't explore properties" << std::endl;
        }


        //// Exposure (Used instead of Shutter ... what about gain, intensity, etc. ???
        //// --------------------------------------------------------------------------------------------------

        //EnumNode_spin exposureModeNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("ExposureMode");
        //exposureModeNode.setEntryBySymbolic("Timed");
        //exposureModeNode.print();

        //EnumNode_spin exposureAutoNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("ExposureAuto");
        //exposureAutoNode.setEntryBySymbolic("Off");
        //exposureAutoNode.print();

        //FloatNode_spin exposureTimeNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("ExposureTime");
        //exposureTimeNode.print();

        //// Gain .
        //// --------------------------------------------------------------------------------------------------

        //EnumNode_spin gainSelectorNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("GainSelector");
        //gainSelectorNode.print();

        //EnumNode_spin gainAutoNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("GainAuto");
        //gainAutoNode.print();

        //FloatNode_spin gainNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("Gain");
        //gainNode.print();

        // Trigger
        // --------------------------------------------------------------------------------------------------

        //EnumNode_spin triggerSelectorNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("TriggerSelector");
        //triggerSelectorNode.print();

        //EnumNode_spin triggerModeNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("TriggerMode");
        //triggerModeNode.print();

        //EnumNode_spin triggerSourceNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("TriggerSource");
        //triggerSourceNode.print();

        //EnumNode_spin triggerOverlapNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("TriggerOverlap");
        //triggerOverlapNode.print();

        //FloatNode_spin triggerDelayNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("TriggerDelay");
        //triggerDelayNode.print();


        //// FrameRate
        //// --------------------------------------------------------------------------------------------------

        //BoolNode_spin frameRateEnableNode = nodeMapCamera_.getNodeByName<BoolNode_spin>("AcquisitionFrameRateEnable");
        //frameRateEnableNode.setValue(true);
        //frameRateEnableNode.print();

        //FloatNode_spin frameRateNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("AcquisitionFrameRate");
        //frameRateNode.print();

        //FloatNode_spin resFrameRateNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("AcquisitionResultingFrameRate");
        //resFrameRateNode.print();


        //// Temperature
        ////---------------------------------------------------------------------------------------------------
        //FloatNode_spin tempNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("DeviceTemperature");
        //tempNode.print();

        //// Blacklevel
        //// ---------------------------------------------------------------------------------------------------
        //EnumNode_spin blackLevelSelectorNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("BlackLevelSelector");
        //blackLevelSelectorNode.print();

        //FloatNode_spin blackLevelNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("BlackLevel");
        //blackLevelNode.print();

        //// Gamma
        //// ----------------------------------------------------------------------------------------------------
        //FloatNode_spin gammaNode = nodeMapCamera_.getNodeByName<FloatNode_spin>("Gamma");
        //gammaNode.print();

        //BoolNode_spin gammaEnableNode = nodeMapCamera_.getNodeByName<BoolNode_spin>("GammaEnable");
        //gammaEnableNode.print();


        // Offset
        // -------------------------------------------------------------------------------------------------------
        IntegerNode_spin offsetXNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("OffsetX");
        int64_t offsetXVal = offsetXNode.value();
        int64_t offsetXInc = offsetXNode.increment();
        int64_t offsetXMax = offsetXNode.maxValue();
        int64_t offsetXMin = offsetXNode.minValue();

        std::cout << "offsetX Val: " << offsetXVal << std::endl;
        std::cout << "offsetX Inc: " << offsetXInc << std::endl;
        std::cout << "offsetX min: " << offsetXMin << std::endl;
        std::cout << "offsetX max: " << offsetXMax << std::endl;
        std::cout << std::endl;

        IntegerNode_spin offsetYNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("OffsetY");
        int64_t offsetYVal = offsetYNode.value();
        int64_t offsetYInc = offsetYNode.increment();
        int64_t offsetYMax = offsetYNode.maxValue();
        int64_t offsetYMin = offsetYNode.minValue();

        std::cout << "offsetY Val: " << offsetYVal << std::endl;
        std::cout << "offsetY Inc: " << offsetYInc << std::endl;
        std::cout << "offsetY min: " << offsetYMin << std::endl;
        std::cout << "offsetY max: " << offsetYMax << std::endl;
        std::cout << std::endl;

        IntegerNode_spin widthNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("Width");
        int64_t widthVal = widthNode.value();
        int64_t widthInc = widthNode.increment();
        int64_t widthMax = widthNode.maxValue();
        int64_t widthMin = widthNode.minValue();

        std::cout << "width Val:   " << widthVal << std::endl;
        std::cout << "width Inc:   " << widthInc << std::endl;
        std::cout << "width max:   " << widthMax << std::endl;
        std::cout << "width min:   " << widthMin << std::endl;
        std::cout << std::endl;

        IntegerNode_spin heightNode = nodeMapCamera_.getNodeByName<IntegerNode_spin>("Height");
        int64_t heightVal = heightNode.value();
        int64_t heightInc = heightNode.increment();
        int64_t heightMax = heightNode.maxValue();
        int64_t heightMin = heightNode.minValue();

        std::cout << "height Val:   " << heightVal << std::endl;
        std::cout << "height Inc:   " << heightInc << std::endl;
        std::cout << "height max:   " << heightMax << std::endl;
        std::cout << "height min:   " << heightMin << std::endl;
        std::cout << std::endl;

        EnumNode_spin pixelFormatNode = nodeMapCamera_.getNodeByName<EnumNode_spin>("PixelFormat");
        //std::vector<EntryNode_spin> pixelFormatEntryVec  = pixelFormatNode.entries();
        pixelFormatNode.print();


        //// -------------------------------------------------------------------------------------------------------

        ////std::vector<EnumNode_spin> nodeVec = nodeMapTLDevice_.nodes<EnumNode_spin>();
        ////std::vector<EnumNode_spin> nodeVec = nodeMapCamera_.nodes<EnumNode_spin>();
        ////
        //std::vector<BaseNode_spin> nodeVec = nodeMapCamera_.nodes<BaseNode_spin>();

        //std::ofstream fout;
        //fout.open("camera_map_nodes.txt");

        //for (auto node : nodeVec)
        //{
        //    fout <<  node.name() << ",  " << node.typeAsString() << std::endl;
        //    std::cout << node.name() << ",  " << node.typeAsString() << std::endl;

        ////    std::cout << "name: " << node.name() << ", numberOfEntries: " << node.numberOfEntries() << std::endl;
        ////    std::vector<EntryNode_spin> entryNodeVec = node.entries();
        ////    EntryNode_spin currEntryNode = node.currentEntry();
        ////    std::cout << "  current: " << currEntryNode.name() << std::endl;
        ////    for (auto entry : entryVec)
        ////    {
        ////        std::cout << "  name:    " << entry.name() << ", " << entry.displayName() << std::endl;
        ////    }
        //}
        //
        //fout.close();
        //// --------------------------------------------------------------------------------------------------------

    }


    TimeStamp CameraDevice_spin::cameraOffsetTime()
    {

        TimeStamp pc_ts, cam_ts;
        double pc_s, cam_s, offset_s;
        std::vector<double> timeofs;

        for (int ind = 0; ind < 10; ind++)
        {

            //get computer local time since midnight
            GetTime* gettime = new GetTime(0, 0);
            pc_ts = gettime->getPCtime();
            pc_s = (double)((pc_ts.seconds*1e6) + (pc_ts.microSeconds))*1e-6;

            //calculate camera time
            cam_ts = getDeviceTimeStamp();
            cam_s = (double)((cam_ts.seconds*1e6) + (cam_ts.microSeconds))*1e-6;

            timeofs.push_back(pc_s - cam_s);
            //printf("%0.06f \n" ,pc_s-cam_s);
            //printf("%0.06f  %0.06f pc_s-cam_us\n ", pc_s ,cam_s);
            //printf("%0.06f \n", pc_s);
        }

        //write_time("offset.csv",20,timeofs);

        //calculate mean
        offset_s = accumulate(timeofs.begin(), timeofs.end(), 0.0) / timeofs.size();
        cam_ofs.seconds = int(offset_s);
        cam_ofs.microSeconds = (offset_s - cam_ofs.seconds)*1e6;
        //ofs_isSet = false;

        //calculate std dev
        double std_sum = 0;
        for (int k = 0; k < timeofs.size(); k++)
        {
            std_sum += (timeofs[k] - offset_s) * (timeofs[k] - offset_s);
        }

        std_sum = std_sum / timeofs.size();
        std_sum = sqrt(std_sum);

        //printf("%0.06f average offset \n" ,offset_s);
        //printf("%0.06f std deviation \n ",std_sum);
        printf("%d seconds %d microseconds", cam_ofs.seconds, cam_ofs.microSeconds);
        return cam_ofs;

    }

    TimeStamp CameraDevice_spin::getCPUtime()
    {
        return cpu_time;
    }

    void CameraDevice_spin::setupNIDAQ(NIDAQUtils* nidaq_task, unsigned int cameraNumber)
    {
        nidaq_task_ = nidaq_task;
        cameraNumber_ = cameraNumber;
        std::cout << "setup " << cameraNumber_ <<  std::endl;
    }

}
#endif
