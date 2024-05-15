BIAS
*****

BIAS is a software application for recording video from IEEE 1394 and USB3
Cameras.  BIAS was intially designed as image acquisition software for
experiments in animial behavior. For example, recording the behavior of fruit
flies in a walking arena. 

|

.. figure:: http://public.iorodeo.com/notes/bias/_images/bias_charlie.png
    :scale: 100 %
    :alt: alternate text
    :align: center



Features
---------

BIAS provides the following features: 

* Control of camera properties (brightness, shutter, gain, etc.)
* Timed video recordings
* Support for a variety of video file formats (avi,fmf, ufmf, mjpg, raw image
* files) etc. 
* JSON based configuration files 
* External control via http commands - start/stop recording, set camera
* configuration etc.
* A plugin system for machine vision applications and for controlling external
* instrumentation
* Multiple cameras
* Image alignment tools
* Cross platform - windows, linux


Documentation
-------------

http://public.iorodeo.com/notes/bias/

Installation
------------

TODO

Developer Build Instructions
----------------------------

Requirements
^^^^^^^^^^^^

- Visual Studio 2022 Community Edition
- CMake 3.29.2
- Qt 5
- OpenCV 4
- Spinnaker 2.6.0 

Notes
^^^^^
  
Here is how I built BIAS on Windows, May 2024. 

- Installed Visual Studio 2022 Community Edition.
- Selected the following workloads during install
    - Desktop development with C++
    - Universal Windows Platform development (not sure if this is necessary)
    - Python development (probably not necessary)
    - Github copilot workloads (definitely not necessary)

- Installed CMake 3.29.2
https://cmake.org/download/

- Anaconda version 2023.07.1 was installed on my machine already, used its build of Qt
  - Added Qt to my PATH environment variable:
  - <anaconda3>\Library\Lib\cmake\Qt5
  - <anaconda3>\Library\plugins\platforms
  - <anaconda3>\Library\bin

- Cloned OpenCV 4 from https://github.com/opencv/opencv

- Built OpenCV:

  - In CMake, set <opencv> as source directory and <opencv>/build as build directory
  - Clicked Configure
  - Made sure WITH_FFMPEG was checked (default)
  - Made sure CMAKE_INSTALL_PREFIX was <opencv4>/build/install (default)
  - Clicked Configure
  - Clicked Generate
  - Opened the project in VisualStudio
  - Chose Release x64 and Build

- Added <opencv>\build\bin\Release to PATH environment variable
    
- Downloaded and installed Spinnaker 2.6.0. There are newer versions, code might not be compatible.

- Opened BIAS in CMake, configured, made sure Spinnaker and Video backends were selected, generated, opened in VisualStudio, and built. 

- This builds test_gui.exe, which can be run by double-clicking or from the command line.

Creating a distributable
------------------------

- Make a directory, e.g. v0.71.0, and copy test_gui.exe into it
- Copy vc_redist.x64.exe into this directory from
Visual Studio's redistributable directory, e.g.
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC\v143
- Within this directory, run
windeployqt.exe test_gui.exe
to copy Qt requirements into this directory
- Manually copy OpenCV required dlls from <opencv>\build\bin\Release:

  - opencv_core490.dll
  - opencv_imgcodecs490.dll
  - opencv_imgproc490.dll
  - opencv_videoio490.dll
- Manually copy Spinnaker requirements to this directory:
  - C:\Program Files\FLIR Systems\Spinnaker\bin\vs2015\SpinnakerC_v140.dll
  - C:\Program Files\FLIR Systems\Spinnaker\lib64\vs2015\SpinVideoC_v140.lib

- Make a .bat file setup.bat with the following::
  @echo off
  vc_redist.x64.exe
  robocopy . "C:\Program Files\BIAS" /E

- Follow these instructions to use iexpress.exe to create a self-extracting executable installer:
https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170
I copied all files in the deploy directory, which required going into each directory created by windeployqt and selecting all the files in there. 
