#include <stdio.h>
#include "string.h"
#include "spin_utils.hpp"


#define MAX_BUFF_LEN 256
using namespace bias;
// Use the following enum and global constant to select whether chunk data is
// displayed from the image or the nodemap.


// This function helps to check if a node is available and readable
SpinUtils::SpinUtils() {

    gettime = new bias::GetTime(0,0);
    //timeStamps.assign(500000, std::vector<float>(2, 0.0));
    
}

// This function handles the error prints when a node or entry is unavailable or
// not readable/writable on the connected camera
void SpinUtils::PrintRetrieveNodeFailure(char node[], char name[])
{
    printf("Unable to get %s (%s %s retrieval failed).\n\n", node, name, node);
}

bool8_t SpinUtils::IsAvailableAndReadable(spinNodeHandle hNode, char nodeName[])
{
    bool8_t pbAvailable = False;
    spinError err = SPINNAKER_ERR_SUCCESS;
    err = spinNodeIsAvailable(hNode, &pbAvailable);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve node availability (%s node), with error %d...\n\n", nodeName, err);
    }

    bool8_t pbReadable = False;
    err = spinNodeIsReadable(hNode, &pbReadable);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve node readability (%s node), with error %d...\n\n", nodeName, err);
    }
    return pbReadable && pbAvailable;
}

// This function helps to check if a node is available and writable
bool8_t SpinUtils::IsAvailableAndWritable(spinNodeHandle hNode, char nodeName[])
{
    bool8_t pbAvailable = False;
    spinError err = SPINNAKER_ERR_SUCCESS;
    err = spinNodeIsAvailable(hNode, &pbAvailable);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve node availability (%s node), with error %d...\n\n", nodeName, err);
    }

    bool8_t pbWritable = False;
    err = spinNodeIsWritable(hNode, &pbWritable);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve node writability (%s node), with error %d...\n\n", nodeName, err);
    }
    return pbWritable && pbAvailable;
}

// This function prints the device information of the camera from the transport
// layer; please see NodeMapInfo_C example for more in-depth comments on
// printing device information from the nodemap.
spinError SpinUtils::PrintDeviceInfo(spinNodeMapHandle hNodeMap)
{
    spinError err = SPINNAKER_ERR_SUCCESS;
    unsigned int i = 0;

    printf("\n*** DEVICE INFORMATION ***\n\n");

    // Retrieve device information category node
    spinNodeHandle hDeviceInformation = NULL;

    err = spinNodeMapGetNode(hNodeMap, "DeviceInformation", &hDeviceInformation);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve node. Non-fatal error %d...\n\n", err);
        return err;
    }

    // Retrieve number of nodes within device information node
    size_t numFeatures = 0;

    if (IsAvailableAndReadable(hDeviceInformation, "DeviceInformation"))
    {
        err = spinCategoryGetNumFeatures(hDeviceInformation, &numFeatures);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to retrieve number of nodes. Non-fatal error %d...\n\n", err);
            return err;
        }
    }
    else
    {
        err = SPINNAKER_ERR_ACCESS_DENIED;
        printf("Unable to read device information. Non-fatal error %d...\n\n", err);
        return err;
    }

    // Iterate through nodes and print information
    for (i = 0; i < numFeatures; i++)
    {
        spinNodeHandle hFeatureNode = NULL;

        err = spinCategoryGetFeatureByIndex(hDeviceInformation, i, &hFeatureNode);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to retrieve node. Non-fatal error %d...\n\n", err);
            continue;
        }

        spinNodeType featureType = UnknownNode;

        // Get feature node name
        char featureName[MAX_BUFF_LEN];
        size_t lenFeatureName = MAX_BUFF_LEN;

        err = spinNodeGetName(hFeatureNode, featureName, &lenFeatureName);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            strcpy(featureName, "Unknown name");
        }

        if (IsAvailableAndReadable(hFeatureNode, featureName))
        {
            err = spinNodeGetType(hFeatureNode, &featureType);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("Unable to retrieve node type. Non-fatal error %d...\n\n", err);
                continue;
            }
        }
        else
        {
            printf("%s: Node not readable\n", featureName);
            continue;
        }

        char featureValue[MAX_BUFF_LEN];
        size_t lenFeatureValue = MAX_BUFF_LEN;

        err = spinNodeToString(hFeatureNode, featureValue, &lenFeatureValue);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            strcpy(featureValue, "Unknown value");
        }

        printf("%s: %s\n", featureName, featureValue);
    }
    printf("\n");

    return err;
}

spinError SpinUtils::ConfigureTrigger(spinNodeMapHandle hNodeMap)
{
    spinError err = SPINNAKER_ERR_SUCCESS;

    printf("\n\n*** TRIGGER CONFIGURATION ***\n\n");

    if (chosenTrigger == SOFTWARE)
    {
        printf("Software trigger chosen...\n\n");
    }
    else if (chosenTrigger == HARDWARE)
    {
        printf("Hardware trigger chosen...\n\n");
    }

    //
    // Ensure trigger mode off
    //
    // *** NOTES ***
    // The trigger must be disabled in order to configure whether the source
    // is software or hardware.
    //
    spinNodeHandle hTriggerMode = NULL;
    spinNodeHandle hTriggerModeOff = NULL;
    int64_t triggerModeOff = 0;

    err = spinNodeMapGetNode(hNodeMap, "TriggerMode", &hTriggerMode);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to disable trigger mode. Aborting with error %d...\n\n", err);
        return err;
    }

    // check if available and writable
    if (!IsAvailableAndReadable(hTriggerMode, "TriggerMode"))
    {
        PrintRetrieveNodeFailure("node", "TriggerMode");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    // fetch entry
    err = spinEnumerationGetEntryByName(hTriggerMode, "Off", &hTriggerModeOff);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to disable trigger mode. Aborting with error %d...\n\n", err);
        return err;
    }

    // check if available and readable
    if (!IsAvailableAndReadable(hTriggerModeOff, "TriggerModeOff"))
    {
        PrintRetrieveNodeFailure("entry", "TriggerMode 'Off'");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    err = spinEnumerationEntryGetIntValue(hTriggerModeOff, &triggerModeOff);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to disable trigger mode. Aborting with error %d...\n\n", err);
        return err;
    }

    // turn trigger mode off
    if (!IsAvailableAndWritable(hTriggerMode, "TriggerMode"))
    {
        PrintRetrieveNodeFailure("node", "TriggerMode");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    err = spinEnumerationSetIntValue(hTriggerMode, triggerModeOff);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to disable trigger mode. Aborting with error %d...\n\n", err);
        return err;
    }

    printf("Trigger mode disabled...\n");

    //
    // Set TriggerSelector to FrameStart
    //
    // *** NOTES ***
    // For this example, the trigger selector should be set to frame start.
    // This is the default for most cameras.
    //
    spinNodeHandle hTriggerSelector = NULL;
    spinNodeHandle hTriggerSelectorChoice = NULL;
    int64_t triggerSelectorChoice = 0;

    // Retrieve enumeration node from nodemap
    err = spinNodeMapGetNode(hNodeMap, "TriggerSelector", &hTriggerSelector);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to choose trigger selector. Aborting with error %d...\n\n", err);
        return err;
    }

    // check if readable
    if (!IsAvailableAndReadable(hTriggerSelector, "TriggerSelector"))
    {
        PrintRetrieveNodeFailure("node", "TriggerSelector");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    // Retrieve entry node from enumeration node to set selector
    err = spinEnumerationGetEntryByName(hTriggerSelector, "FrameStart", &hTriggerSelectorChoice);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to choose trigger selector. Aborting with error %d...\n\n", err);
        return err;
    }

    if (!IsAvailableAndReadable(hTriggerSelectorChoice, "TriggerSelectorChoice"))
    {
        PrintRetrieveNodeFailure("entry", "TriggerSelector 'Choice'");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    // Retrieve integer value from entry node
    err = spinEnumerationEntryGetIntValue(hTriggerSelectorChoice, &triggerSelectorChoice);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to choose trigger selector. Aborting with error %d...\n\n", err);
        return err;
    }

    // set trigger source choice
    if (!IsAvailableAndWritable(hTriggerSelector, "TriggerSelector"))
    {
        PrintRetrieveNodeFailure("node", "TriggerSelector");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    err = spinEnumerationSetIntValue(hTriggerSelector, triggerSelectorChoice);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to choose trigger selector. Aborting with error %d...\n\n", err);
        return err;
    }

    printf("Trigger selector set to frame start...\n");

    // Set Triggeractivation to Rising Edge
//
// *** NOTES ***
// For this example, the trigger selector should be set to frame start.
// This is the default for most cameras.
//

    //
    // Choose trigger source
    //
    // *** NOTES ***
    // The trigger source must be set to hardware or software while trigger
    // mode is off.
    //
    spinNodeHandle hTriggerSource = NULL;
    spinNodeHandle hTriggerSourceChoice = NULL;
    int64_t triggerSourceChoice = 0;

    // Retrieve enumeration node from nodemap
    err = spinNodeMapGetNode(hNodeMap, "TriggerSource", &hTriggerSource);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to choose trigger source. Aborting with error %d...\n\n", err);
        return err;
    }

    // check if readable
    if (!IsAvailableAndReadable(hTriggerSource, "TriggerSource"))
    {
        PrintRetrieveNodeFailure("node", "TriggerSource");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    if (chosenTrigger == SOFTWARE)
    {
        // Retrieve entry node from enumeration node to set software
        err = spinEnumerationGetEntryByName(hTriggerSource, "Software", &hTriggerSourceChoice);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to choose trigger source. Aborting with error %d...\n\n", err);
            return err;
        }
    }
    else if (chosenTrigger == HARDWARE)
    {
        // Retrieve entry node from enumeration node to set hardware ('Line0')
        err = spinEnumerationGetEntryByName(hTriggerSource, "Line0", &hTriggerSourceChoice);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to choose trigger source. Aborting with error %d...\n\n", err);
            return err;
        }
    }

    if (!IsAvailableAndReadable(hTriggerSourceChoice, "TriggerSourceChoice"))
    {
        PrintRetrieveNodeFailure("entry", "TriggerSource 'Choice'");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    // Retrieve integer value from entry node
    err = spinEnumerationEntryGetIntValue(hTriggerSourceChoice, &triggerSourceChoice);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to choose trigger source. Aborting with error %d...\n\n", err);
        return err;
    }

    // set trigger source choice
    if (!IsAvailableAndWritable(hTriggerSource, "TriggerSource"))
    {
        PrintRetrieveNodeFailure("node", "TriggerSource");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    err = spinEnumerationSetIntValue(hTriggerSource, triggerSourceChoice);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to choose trigger source. Aborting with error %d...\n\n", err);
        return err;
    }

    if (chosenTrigger == SOFTWARE)
    {
        printf("Trigger source set to software...\n");
    }
    else if (chosenTrigger == HARDWARE)
    {
        printf("Trigger source set to line 0...\n");
    }

    spinNodeHandle hTriggerActivation = NULL;
    spinNodeHandle hTriggerActivationChoice = NULL;
    int64_t triggerActivationChoice = 0;

    // Retrieve enumeration node from nodemap
    err = spinNodeMapGetNode(hNodeMap, "TriggerActivation", &hTriggerActivation);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to choose trigger selector. Aborting with error %d...\n\n", err);
        return err;
    }

    // check if readable
    if (!IsAvailableAndReadable(hTriggerActivation, "TriggerActivation"))
    {
        PrintRetrieveNodeFailure("node", "TriggerActivation");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    // Retrieve entry node from enumeration node to set selector
    err = spinEnumerationGetEntryByName(hTriggerActivation, "RisingEdge", &hTriggerActivationChoice);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to choose trigger selector. Aborting with error %d...\n\n", err);
        return err;
    }

    if (!IsAvailableAndReadable(hTriggerActivationChoice, "TriggerActivationChoice"))
    {
        PrintRetrieveNodeFailure("entry", "TriggerActivation 'Choice'");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    // Retrieve integer value from entry node
    err = spinEnumerationEntryGetIntValue(hTriggerActivationChoice, &triggerActivationChoice);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to choose trigger selector. Aborting with error %d...\n\n", err);
        return err;
    }

    // set trigger source choice
    if (!IsAvailableAndWritable(hTriggerActivation, "TriggerActivation"))
    {
        PrintRetrieveNodeFailure("node", "TriggerSelector");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    err = spinEnumerationSetIntValue(hTriggerActivation, triggerActivationChoice);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to choose trigger selector. Aborting with error %d...\n\n", err);
        return err;
    }

    printf("Trigger selector set to frame start...\n");

    //
    // Turn trigger mode on
    //
    // *** LATER ***
    // Once the appropriate trigger source has been set, turn trigger mode
    // in order to retrieve images using the trigger.
    //

    spinNodeHandle hTriggerModeOn = NULL;
    int64_t triggerModeOn = 0;

    // TODO: Blackfly and Flea3 GEV cameras need 1 second delay after trigger mode is
    // turned on
    err = spinEnumerationGetEntryByName(hTriggerMode, "On", &hTriggerModeOn);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to enable trigger mode. Aborting with error %d...\n\n", err);
        return err;
    }

    if (!IsAvailableAndReadable(hTriggerModeOn, "TriggerModeOn"))
    {
        PrintRetrieveNodeFailure("entry", "Trigger Mode 'On'");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    err = spinEnumerationEntryGetIntValue(hTriggerModeOn, &triggerModeOn);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to enable trigger mode. Aborting with error %d...\n\n", err);
        return err;
    }

    if (!IsAvailableAndWritable(hTriggerSource, "TriggerSource"))
    {
        PrintRetrieveNodeFailure("node", "TriggerSource");
        return SPINNAKER_ERR_ACCESS_DENIED;
    }

    err = spinEnumerationSetIntValue(hTriggerMode, triggerModeOn);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to enable trigger mode. Aborting with error %d...\n\n", err);
        return err;
    }

    printf("Trigger mode enabled...\n\n");

    return err;
}

spinError SpinUtils::ConfigureChunkData(spinNodeMapHandle hNodeMap)
{
    spinError err = SPINNAKER_ERR_SUCCESS;

    unsigned int i = 0;

    //printf("\n\n*** CONFIGURING CHUNK DATA ***\n\n");

    //
    // Activate chunk mode
    //
    // *** NOTES ***
    // Once enabled, chunk data will be available at the end of hte payload of
    // every image captured until it is disabled. Chunk data can also be
    // retrieved from the nodemap.
    //
    spinNodeHandle hChunkModeActive = NULL;

    err = spinNodeMapGetNode(hNodeMap, "ChunkModeActive", &hChunkModeActive);

    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to activate chunk mode. Aborting with error %d...\n\n", err);
        return err;
    }

    // Check if available and writable
    if (IsAvailableAndWritable(hChunkModeActive, "ChunkModeActive"))
    {
        err = spinBooleanSetValue(hChunkModeActive, True);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to activate chunk mode. Aborting with error %d...\n\n", err);
            return err;
        }
    }
    else
    {
        err = SPINNAKER_ERR_ACCESS_DENIED;
        printf("Unable to write to chunk mode. Aborting with error %d...\n\n", err);
        return err;
    }

    //printf("Chunk mode activated...\n");

    //
    // Enable all types of chunk data
    //
    // *** NOTES ***
    // Enabling chunk data requires working with nodes: "ChunkSelector" is an
    // enumeration selector node and "ChunkEnable" is a boolean. It requires
    // retrieving the selector node (which is of enumeration node type),
    // selecting the entry of the chunk data to be enabled, retrieving the
    // corresponding boolean, and setting it to true.
    //
    // In this example, all chunk data is enabled, so these steps are performed
    // in a loop. Once this is complete, chunk mode still needs to be activated.
    //
    spinNodeHandle hChunkSelector = NULL;
    size_t numEntries = 0;

    // Retrieve selector node, check if available and readable and writable
    err = spinNodeMapGetNode(hNodeMap, "ChunkSelector", &hChunkSelector);

    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve chunk selector entries. Aborting with error %d...\n\n", err);
        return err;
    }

    // Retrieve number of entries
    if (IsAvailableAndReadable(hChunkSelector, "ChunkSelector"))
    {
        err = spinEnumerationGetNumEntries(hChunkSelector, &numEntries);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to retrieve number of entries. Aborting with error %d...\n\n", err);
            return err;
        }
    }
    else
    {
        err = SPINNAKER_ERR_ACCESS_DENIED;
        printf("Unable to read number of entries. Aborting with error %d...\n\n", err);
        return err;
    }

    //printf("Enabling entries...\n");

    for (i = 0; i < numEntries; i++)
    {
        // Retrieve entry node
        spinNodeHandle hEntry = NULL;

        err = spinEnumerationGetEntryByIndex(hChunkSelector, i, &hEntry);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("\tUnable to enable chunk entry (error %d)...\n\n", err);
            continue;
        }

        // Check if available and readable, retrieve entry name
        char entryName[MAX_BUFF_LEN];
        size_t lenEntryName = MAX_BUFF_LEN;

        if (IsAvailableAndReadable(hEntry, "ChunkEntry"))
        {
            err = spinNodeGetDisplayName(hEntry, entryName, &lenEntryName);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("\t%s: unable to retrieve chunk entry display name (error %d)...\n", entryName, err);
            }
        }
        else
        {
            continue;
        }
        // Retrieve enum entry integer value
        int64_t value = 0;

        err = spinEnumerationEntryGetIntValue(hEntry, &value);

        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("\t%s: unable to get chunk entry value (error %d)...\n", entryName, err);
            continue;
        }

        // Set integer value
        if (IsAvailableAndWritable(hChunkSelector, "ChunkSelector"))
        {
            err = spinEnumerationSetIntValue(hChunkSelector, value);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("\t%s: unable to set chunk entry value (error %d)...\n", entryName, err);
                continue;
            }
        }
        else
        {
            err = SPINNAKER_ERR_ACCESS_DENIED;
            printf("\t%s: unable to write to chunk entry value (error %d)...\n", entryName, err);
            return err;
        }

        // Retrieve corresponding chunk enable node
        spinNodeHandle hChunkEnable = NULL;

        err = spinNodeMapGetNode(hNodeMap, "ChunkEnable", &hChunkEnable);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("\t%s: unable to get entry from nodemap (error %d)...\n", entryName, err);
            continue;
        }

        // Retrieve chunk enable value and set to true if necessary
        bool8_t isEnabled = False;

        if (IsAvailableAndWritable(hChunkEnable, "ChunkEnable"))
        {
            err = spinBooleanGetValue(hChunkEnable, &isEnabled);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("\t%s: unable to get chunk entry boolean value (error %d)...\n", entryName, err);
                continue;
            }
        }
        else
        {
            printf("\t%s: not writable\n", entryName);
            continue;
        }
        // Consider the case in which chunk data is enabled but not writable
        if (!isEnabled)
        {
            // Set chunk enable value to true

            err = spinBooleanSetValue(hChunkEnable, True);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("\t%s: unable to set chunk entry boolean value (error %d)...\n", entryName, err);
                continue;
            }
        }

        //printf("\t%s: enabled\n", entryName);
    }

    return err;
}

// This function disables each type of chunk data before disabling chunk data mode.
spinError SpinUtils::DisableChunkData(spinNodeMapHandle hNodeMap)
{
    spinError err = SPINNAKER_ERR_SUCCESS;

    spinNodeHandle hChunkSelector = NULL;
    size_t numEntries = 0;
    unsigned int i = 0;

    // Retrieve selector node
    err = spinNodeMapGetNode(hNodeMap, "ChunkSelector", &hChunkSelector);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve chunk selector entries. Aborting with error %d...\n\n", err);
        return err;
    }

    // Retrieve number of entries, check if readable
    if (IsAvailableAndReadable(hChunkSelector, "ChunkSelector"))
    {
        err = spinEnumerationGetNumEntries(hChunkSelector, &numEntries);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to retrieve number of entries. Aborting with error %d...\n\n", err);
            return err;
        }
    }
    else
    {
        err = SPINNAKER_ERR_ACCESS_DENIED;
        printf("Unable to read number of entries. Aborting with error %d...\n\n", err);
        return err;
    }

    printf("Disabling entries...\n");

    for (i = 0; i < numEntries; i++)
    {
        // Retrieve entry node
        spinNodeHandle hEntry = NULL;

        err = spinEnumerationGetEntryByIndex(hChunkSelector, i, &hEntry);

        // Go to next node if problem occurs
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            continue;
        }

        // Retrieve entry name
        char entryName[MAX_BUFF_LEN];
        size_t lenEntryName = MAX_BUFF_LEN;

        if (IsAvailableAndReadable(hEntry, "ChunkEntry"))
        {
            err = spinNodeGetDisplayName(hEntry, entryName, &lenEntryName);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("\t%s: unable to retrieve chunk entry by display name (error %d)...\n", entryName, err);
            }
        }
        else
        {
            continue;
        }
        // Retrieve enum entry integer value
        int64_t value = 0;

        err = spinEnumerationEntryGetIntValue(hEntry, &value);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("\t%s: unable to get chunk entry value (error %d)...\n", entryName, err);
            continue;
        }

        // Set integer value
        if (IsAvailableAndWritable(hChunkSelector, "ChunkSelector"))
        {
            err = spinEnumerationSetIntValue(hChunkSelector, value);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("\t%s: unable to set chunk entry value (error %d)...\n", entryName, err);
                continue;
            }
        }
        else
        {
            err = SPINNAKER_ERR_ACCESS_DENIED;
            printf("\t%s: unable to set chunk entry value (error %d)...\n", entryName, err);
            return err;
        }

        // Retrieve corresponding chunk enable node
        spinNodeHandle hChunkEnable = NULL;
        err = spinNodeMapGetNode(hNodeMap, "ChunkEnable", &hChunkEnable);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("\t%s: unable to get entry from nodemap (error %d)...\n", entryName, err);
            continue;
        }

        // Retrieve chunk enable value and set to false if necessary
        bool8_t isEnabled = False;

        if (IsAvailableAndWritable(hChunkEnable, "ChunkEnable"))
        {
            err = spinBooleanGetValue(hChunkEnable, &isEnabled);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("\t%s: unable to get chunk entry boolean value (error %d)...\n", entryName, err);
                continue;
            }
        }
        else
        {
            printf("\t%s: not writable\n", entryName);
            continue;
        }

        // Consider the case in which chunk data is enabled but not writable
        if (isEnabled)
        {
            // Set chunk enable value to false
            err = spinBooleanSetValue(hChunkEnable, False);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("\t%s: unable to set chunk entry boolean value (error %d)...\n", entryName, err);
                continue;
            }
        }

        printf("\t%s: disabled\n", entryName);
    }

    printf("\n");

    // Disabling ChunkModeActive
    spinNodeHandle hChunkModeActive = NULL;

    err = spinNodeMapGetNode(hNodeMap, "ChunkModeActive", &hChunkModeActive);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to get ChunkModeActive node. Aborting with error %d...\n\n", err);
        return err;
    }

    if (IsAvailableAndWritable(hChunkModeActive, "ChunkModeActive"))
    {
        err = spinBooleanSetValue(hChunkModeActive, False);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to deactivate chunk mode. Aborting with error %d...\n\n", err);
            return err;
        }
    }
    else
    {
        err = SPINNAKER_ERR_ACCESS_DENIED;
        printf("Unable to write to chunk mode. Aborting with error %d...\n\n", err);
        return err;
    }

    printf("Chunk mode deactivated...\n");
    return err;
}


spinError SpinUtils::deInitialize_camera(spinCamera& hCam, spinNodeMapHandle& hNodeMap) {

    spinError err = SPINNAKER_ERR_SUCCESS;

    // Disable chunck data
    err = DisableChunkData(hNodeMap);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        return err;
    }

    // Deinitialize camera
    err = spinCameraDeInit(hCam);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to deinitialize camera. Non-fatal error %d...\n\n", err);
    }

    return err;
}


spinError SpinUtils::ReleaseSystem(spinSystem& hSystem, spinCameraList& hCameraList) {

    spinError err = SPINNAKER_ERR_SUCCESS;
    // Clear and destroy camera list before releasing system
    err = spinCameraListClear(hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to clear camera list. Aborting with error %d...\n\n", err);
        return err;
    }

    err = spinCameraListDestroy(hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to destroy camera list. Aborting with error %d...\n\n", err);
        return err;
    }

    // Release system
    err = spinSystemReleaseInstance(hSystem);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to release system instance. Aborting with error %d...\n\n", err);
        return err;
    }

    return err;
}

spinError SpinUtils::setupSystem(spinSystem& hSystem, spinCameraList& hCameraList) {

    spinError err = SPINNAKER_ERR_SUCCESS;

    err = spinSystemGetInstance(&hSystem);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve system instance. Aborting with error %d...\n\n", err);
        return err;
    }

    err = spinCameraListCreateEmpty(&hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to create camera list. Aborting with error %d...\n\n", err);
        return err;
    }

    err = spinSystemGetCameras(hSystem, hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve camera list. Aborting with error %d...\n\n", err);
        return err;
    }

    return err;
}


spinError SpinUtils::initialize_camera(spinCamera& hCam, spinNodeMapHandle& hNodeMap,
                            spinNodeMapHandle& hNodeMapTLDevice) {

    spinError err = SPINNAKER_ERR_SUCCESS;

    // Retrieve TL device nodemap and print device information
    err = spinCameraGetTLDeviceNodeMap(hCam, &hNodeMapTLDevice);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve TL device nodemap. Non-fatal error %d...\n\n", err);
    }
    else
    {
        err = PrintDeviceInfo(hNodeMapTLDevice);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            return err;
        }
    }

    // Initialize camera
    err = spinCameraInit(hCam);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to initialize camera. Aborting with error %d...\n\n", err);
        return err;
    }

    // Retrieve GenICam nodemap
    err = spinCameraGetNodeMap(hCam, &hNodeMap);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve GenICam nodemap. Aborting with error %d...\n\n", err);
        return err;
    }

    // Configure chunk data
    err = ConfigureChunkData(hNodeMap);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        return err;
    }

    // Configure trigger
    err = ConfigureTrigger(hNodeMap);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        return err;
    }

    // Set acquisition mode to continuous
    spinNodeHandle hAcquisitionMode = NULL;
    spinNodeHandle hAcquisitionModeContinuous = NULL;
    int64_t acquisitionModeContinuous = 0;

    err = spinNodeMapGetNode(hNodeMap, "AcquisitionMode", &hAcquisitionMode);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to set acquisition mode to continuous (node retrieval). Aborting with error %d...\n\n", err);
        return err;
    }

    if (IsAvailableAndReadable(hAcquisitionMode, "AcquisitionMode"))
    {
        err = spinEnumerationGetEntryByName(hAcquisitionMode, "Continuous", &hAcquisitionModeContinuous);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf(
                "Unable to set acquisition mode to continuous (entry 'continuous' retrieval). Aborting with error "
                "%d...\n\n",
                err);
            return err;
        }
    }
    else
    {
        err = SPINNAKER_ERR_ACCESS_DENIED;
        printf("Unable to read acquisition mode. Aborting with error %d...\n\n", err);
        return err;
    }

    if (IsAvailableAndReadable(hAcquisitionModeContinuous, "AcquisitionModeContinuous"))
    {
        err = spinEnumerationEntryGetIntValue(hAcquisitionModeContinuous, &acquisitionModeContinuous);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf(
                "Unable to set acquisition mode to continuous (entry int value retrieval). Aborting with error "
                "%d...\n\n",
                err);
            return err;
        }
    }
    else
    {
        err = SPINNAKER_ERR_ACCESS_DENIED;
        printf("Unable to read acquisition mode continuous. Aborting with error %d...\n\n", err);
        return err;
    }

    // set acquisition mode to continuous
    if (IsAvailableAndWritable(hAcquisitionMode, "AcquisitionMode"))
    {
        err = spinEnumerationSetIntValue(hAcquisitionMode, acquisitionModeContinuous);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf(
                "Unable to set acquisition mode to continuous (entry int value setting). Aborting with error %d...\n\n",
                err);
            return err;
        }
        printf("Acquisition mode set to continuous...\n");
    }
    else
    {
        err = SPINNAKER_ERR_ACCESS_DENIED;
        printf("Unable to write to acquisition mode. Aborting with error %d...\n\n", err);
        return err;
    }

    // Begin acquiring images
    err = spinCameraBeginAcquisition(hCam);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to begin image acquisition. Aborting with error %d...\n\n", err);
        return err;
    }

}


spinError SpinUtils::getFrame_camera(spinCamera& hCam, spinImage& hImage, 
                                     bias::NIDAQUtils* nidaq_task,
                                     std::vector<std::vector<float>>& timeStamps,
                                     int framenum) {

    spinError err = SPINNAKER_ERR_SUCCESS;
    uInt32 read_buffer, read_ondemand;

    //pc_ts1 = getPCtime();
    //printf("Image Complete %d...\n", framenum);
    err = spinCameraGetNextImageEx(hCam, 10, &hImage);

    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to get next image. Non-fatal error %d...\n\n", err);

    }

    // Ensure image completion
    bool8_t isIncomplete = False;
    bool8_t hasFailed = False;

    err = spinImageIsIncomplete(hImage, &isIncomplete);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to determine image completion. Non-fatal error %d...\n\n", err);
        hasFailed = True;

    }

    if (isIncomplete)
    {
        spinImageStatus imageStatus = IMAGE_NO_ERROR;

        err = spinImageGetStatus(hImage, &imageStatus);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to retrieve image status. Non-fatal error %d...\n\n", err);
        }
        else
        {
            printf("Image incomplete with image status %d...\n", imageStatus);
        }

    }

    // Release incomplete or failed image
    if (hasFailed)
    {
        err = spinImageRelease(hImage);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to release image. Non-fatal error %d...\n\n", err);
        }

        return err;
    }
    
    /*if (nidaq_task != nullptr) {

        DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task->taskHandle_trigger_in, 10.0, &read_buffer, NULL));
        DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task->taskHandle_grab_in, 10.0, &read_ondemand, NULL));
        timeStamps[framenum][1] = static_cast<float>((read_ondemand - read_buffer)*0.02);
        
    }*/

  
    return err;
}

SpinUtils::~SpinUtils() {

    delete gettime;
}
