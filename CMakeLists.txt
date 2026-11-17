cmake_minimum_required(VERSION 3.19...3.31)

project(psdr_jit VERSION 0.2.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec('drjit').origin).parent / 'cmake' / 'drjit')"
    OUTPUT_VARIABLE drjit_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
# Dr.Jit 1.5.0 wheels use nanobind ABI 20 and the 'drjit' domain.
find_package(nanobind 2.13.0 EXACT CONFIG REQUIRED)
find_package(drjit 1.5.0 EXACT CONFIG REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(PSDR_INCLUDE_DIR include/psdr)
set(PSDR_SOURCE_DIR src)
set(PSDR_SOURCE_FILES
    ${PSDR_INCLUDE_DIR}/psdr.h
    ${PSDR_INCLUDE_DIR}/constants.h
    ${PSDR_INCLUDE_DIR}/fwd.h
    ${PSDR_INCLUDE_DIR}/macros.h
    ${PSDR_INCLUDE_DIR}/object.h
    ${PSDR_INCLUDE_DIR}/types.h
    ${PSDR_INCLUDE_DIR}/utils.h
    ${PSDR_INCLUDE_DIR}/core/ray.h
    ${PSDR_INCLUDE_DIR}/core/frame.h
    ${PSDR_INCLUDE_DIR}/core/intersection.h
    ${PSDR_INCLUDE_DIR}/core/bitmap.h
    ${PSDR_SOURCE_DIR}/core/bitmap.cpp
    ${PSDR_INCLUDE_DIR}/core/bitmap_loader.h
    ${PSDR_SOURCE_DIR}/core/bitmap_loader.cpp
    ${PSDR_INCLUDE_DIR}/core/sampler.h
    ${PSDR_SOURCE_DIR}/core/sampler.cpp
    ${PSDR_INCLUDE_DIR}/core/records.h
    ${PSDR_INCLUDE_DIR}/core/warp.h
    ${PSDR_INCLUDE_DIR}/core/transform.h

    ${PSDR_INCLUDE_DIR}/core/cube_distrb.h
    ${PSDR_SOURCE_DIR}/core/cube_distrb.cpp

    ${PSDR_INCLUDE_DIR}/core/pmf.h
    ${PSDR_SOURCE_DIR}/core/pmf.cpp

    ${PSDR_INCLUDE_DIR}/bsdf/bsdf.h
    ${PSDR_INCLUDE_DIR}/bsdf/diffuse.h
    ${PSDR_SOURCE_DIR}/bsdf/diffuse.cpp

    ${PSDR_INCLUDE_DIR}/bsdf/microfacet.h
    ${PSDR_SOURCE_DIR}/bsdf/microfacet.cpp

    ${PSDR_INCLUDE_DIR}/bsdf/microfacet_pv.h
    ${PSDR_SOURCE_DIR}/bsdf/microfacet_pv.cpp

    ${PSDR_INCLUDE_DIR}/bsdf/roughconductor.h
    ${PSDR_SOURCE_DIR}/bsdf/roughconductor.cpp
    ${PSDR_INCLUDE_DIR}/bsdf/roughdielectric.h
    ${PSDR_SOURCE_DIR}/bsdf/roughdielectric.cpp

    ${PSDR_INCLUDE_DIR}/bsdf/normalmap.h
    ${PSDR_SOURCE_DIR}/bsdf/normalmap.cpp

    ${PSDR_INCLUDE_DIR}/bsdf/ggx.h
    ${PSDR_SOURCE_DIR}/bsdf/ggx.cpp

    ${PSDR_INCLUDE_DIR}/edge/edge.h

    ${PSDR_INCLUDE_DIR}/shape/mesh.h
    ${PSDR_SOURCE_DIR}/shape/mesh.cpp

    ${PSDR_INCLUDE_DIR}/emitter/emitter.h
    ${PSDR_INCLUDE_DIR}/emitter/area.h
    ${PSDR_SOURCE_DIR}/emitter/area.cpp
    ${PSDR_INCLUDE_DIR}/emitter/envmap.h
    ${PSDR_SOURCE_DIR}/emitter/envmap.cpp

    ${PSDR_INCLUDE_DIR}/sensor/sensor.h
    ${PSDR_SOURCE_DIR}/sensor/sensor.cpp

    ${PSDR_INCLUDE_DIR}/sensor/perspective.h
    ${PSDR_SOURCE_DIR}/sensor/perspective.cpp
    ${PSDR_INCLUDE_DIR}/sensor/orthographic.h
    ${PSDR_SOURCE_DIR}/sensor/orthographic.cpp

    ${PSDR_INCLUDE_DIR}/core/tinyexr.h
    ${PSDR_INCLUDE_DIR}/core/miniz.h
    ${PSDR_SOURCE_DIR}/core/miniz.cpp

    ${PSDR_INCLUDE_DIR}/scene/scene_optix.h
    ${PSDR_SOURCE_DIR}/scene/scene_optix.cpp

    ${PSDR_INCLUDE_DIR}/scene/scene.h
    ${PSDR_SOURCE_DIR}/scene/scene.cpp

    ${PSDR_INCLUDE_DIR}/scene/scene_loader.h
    ${PSDR_SOURCE_DIR}/scene/scene_loader.cpp

    ${PSDR_INCLUDE_DIR}/integrator/integrator.h
    ${PSDR_SOURCE_DIR}/integrator/integrator.cpp
    ${PSDR_INCLUDE_DIR}/integrator/field.h
    ${PSDR_SOURCE_DIR}/integrator/field.cpp

    ${PSDR_INCLUDE_DIR}/integrator/collocated.h
    ${PSDR_SOURCE_DIR}/integrator/collocated.cpp

    ${PSDR_INCLUDE_DIR}/integrator/direct.h
    ${PSDR_SOURCE_DIR}/integrator/direct.cpp

    ${PSDR_INCLUDE_DIR}/integrator/path.h
    ${PSDR_SOURCE_DIR}/integrator/path.cpp

    ${PSDR_SOURCE_DIR}/jit_optix_test.cpp

    ${PSDR_SOURCE_DIR}/optix_stubs.cpp

    ${PSDR_SOURCE_DIR}/psdr.cpp
)

nanobind_add_module(psdr_jit NB_STATIC NB_DOMAIN drjit ${PSDR_SOURCE_FILES})
target_include_directories(psdr_jit PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(psdr_jit PRIVATE drjit drjit-core drjit-extra nanothread)
if(MSVC)
    target_compile_definitions(psdr_jit PRIVATE _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
    target_compile_options(psdr_jit PRIVATE /wd4251 /bigobj)
else()
    target_compile_options(psdr_jit PRIVATE -Wall -Wno-reorder -Wno-sign-compare)
endif()

install(TARGETS psdr_jit LIBRARY DESTINATION psdr_jit RUNTIME DESTINATION psdr_jit)

set_target_properties(psdr_jit PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH "$ORIGIN/../drjit"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
set_target_properties(psdr_jit PROPERTIES PREFIX "")
