# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

file(GLOB_RECURSE SYMFORCE_INVESTIGATE_OPT_HEADERS CONFIGURE_DEPENDS *.h)

add_executable(
    investigate_opt
    main.cc
    ${SYMFORCE_INVESTIGATE_OPT_HEADERS}
)

target_link_libraries(
    investigate_opt
    symforce_gen
    symforce_opt
)

set_target_properties(investigate_opt
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples
)