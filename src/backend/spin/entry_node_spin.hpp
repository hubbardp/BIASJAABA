#ifndef ENTRY_NODE_SPIN_HPP
#define ENTRY_NODE_SPIN_HPP
#include "base_node_spin.hpp"

namespace bias
{

    class EntryNode_spin : public BaseNode_spin
    {

        public:

            static spinNodeType ExpectedType();

            EntryNode_spin();
            EntryNode_spin(spinNodeHandle hNode);
            EntryNode_spin(BaseNode_spin node);

            size_t value();
            int64_t intValue();
            std::string symbolic();

            bool isSymbolicValueEqualTo(std::string value);
          
            void setEntryByInt(int64_t val);	    
            virtual void print();
            std::string toString();

    };




} // namespace bias

#endif
