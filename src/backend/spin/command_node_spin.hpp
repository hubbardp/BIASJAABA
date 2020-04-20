#ifndef COMMAND_NODE_SPIN_HPP
#define COMMAND_NODE_SPIN_HPP

#include "base_node_spin.hpp"

namespace bias
{

  class CommandNode_spin : public BaseNode_spin
  {

      public:

          static spinNodeType ExpectedType(); 

          CommandNode_spin();
          CommandNode_spin(spinNodeHandle hNode);
          CommandNode_spin(BaseNode_spin node);  

  };

}

#endif
