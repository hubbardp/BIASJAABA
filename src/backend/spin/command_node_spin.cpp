#include "command_node_spin.hpp"
#include "basic_types.hpp"
#include "exception.hpp"
#include <sstream>


namespace bias
{
    spinNodeType CommandNode_spin::ExpectedType()
    {
        return CommandNode;
    }


    CommandNode_spin::CommandNode_spin() : BaseNode_spin()
    {}


    CommandNode_spin::CommandNode_spin(spinNodeHandle hNode) : BaseNode_spin(hNode)
    {
        if (hNode_ == nullptr)
        {
            return;
        }

        if (!isOfType(ExpectedType()))
        {
            std::stringstream ssError;
            ssError << __FUNCTION__;
            ssError << ": incorrect node type";
            throw RuntimeError(ERROR_SPIN_INCORRECT_NODE_TYPE, ssError.str());
        }
    }


    CommandNode_spin::CommandNode_spin(BaseNode_spin node)
    {
        hNode_ = node.handle();
    }


    /*void CommandNode_spin::setValue(bool value)
    {
        checkNodeHandle();
        checkAvailable();
        checkWritable();

        //bool8_t value_bool8 = (value) ? True : False;

        spinError err = spinCommandSetValue(hNode_,value_bool8);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            std::stringstream ssError;
            ssError << __PRETTY_FUNCTION__;
            ssError << ": unable to set boolean node  value, error = " << err;
            throw RuntimeError(ERROR_SPIN_SET_BOOLEAN_VALUE, ssError.str());
        }
    }*/


}
