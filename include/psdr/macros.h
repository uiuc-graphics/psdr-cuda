#pragma once

#include <drjit/map.h>
#include <drjit/call.h>

#ifdef _WIN32
#	define likely(x)       (x)
#	define unlikely(x)     (x)
#else
#	define likely(x)       __builtin_expect((x),1)
#	define unlikely(x)     __builtin_expect((x),0)
#endif

// #define PSDR_OPTIX_DEBUG
// #define PSDR_MESH_ENABLE_1D_VERTEX_OFFSET
// #define PSDR_PRIMARY_EDGE_VIS_CHECK


#define PSDR_CLASS_DECL_BEGIN(_class_, _mode_, _parent_)    \
    class _class_ _mode_ : public _parent_ {


#define PSDR_CLASS_DECL_END(_class_)                        \
    public:                                                 \
        virtual std::string type_name() const override {    \
            return #_class_;                                \
        }                                                   \
    };

// #define NAMESPACE_BEGIN(name) namespace name {


#define __PSDR_USING_MEMBERS_MACRO__(x) using Base::x;
#define PSDR_USING_MEMBERS(...) DRJIT_MAP(__PSDR_USING_MEMBERS_MACRO__, __VA_ARGS__)


#define PSDR_IMPORT_BASE(Name, ...)                         \
    using Base = Name;                                      \
    PSDR_USING_MEMBERS(__VA_ARGS__)

// #define NAMESPACE_BEGIN(psdr_jit) namespace psdr_jit {

// #define PSDR_NAMESPACE_END }


#if !defined(NAMESPACE_BEGIN)
#  define NAMESPACE_BEGIN(name) namespace name {
#endif
#if !defined(NAMESPACE_END)
#  define NAMESPACE_END(name) }
#endif
