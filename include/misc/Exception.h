#pragma once

#include <stdexcept>
#include <sstream>
#include <psdr/macros.h>

#define PSDR_ASSERT( cond )                                                    \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            std::stringstream ss;                                              \
            std::operator<<(ss, __FILE__);                                     \
            std::operator<<(ss, " (");                                         \
            ss << __LINE__;                                                    \
            std::operator<<(ss, "): ");                                        \
            std::operator<<(ss, #cond);                                        \
            throw psdr_jit::Exception( ss.str().c_str() );                     \
        }                                                                      \
    } while( 0 )


#define PSDR_ASSERT_MSG( cond, msg )                                           \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            std::stringstream ss;                                              \
            std::operator<<(ss, "\n  File \"");                                \
            std::operator<<(ss, __FILE__);                                     \
            std::operator<<(ss, "\", line ");                                  \
            ss << __LINE__;                                                    \
            throw psdr_jit::Exception( ( std::string(msg) + ss.str() ).c_str() );  \
        }                                                                      \
    } while( 0 )



NAMESPACE_BEGIN(psdr_jit)

class Exception : public std::runtime_error
{
 public:
     Exception( const char* msg )
         : std::runtime_error( msg )
     { }
};

NAMESPACE_END(psdr_jit)
