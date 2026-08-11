# Imported targets for the RDKit components SmallField links.
#
# Included from two places, which is why it is a module rather than inline: the top-level build, and the installed
# `SmallFieldConfig.cmake`, where a consumer's project has no `RDKit::*` targets and the export set would otherwise
# reference names that do not exist.
#
# The conda `rdkit-config.cmake` requires a Boost "headers" component that confuses CMake < 3.30's FindBoost, so the
# targets are set up by hand instead of fighting it.

# Prefix resolution, most explicit first: a cached or command-line `RDKIT_PREFIX`, then an activated conda/micromamba
# environment, then the in-tree pixi environment that `pixi install` creates.
#
# Testing the value rather than `DEFINED` matters: `-DRDKIT_PREFIX=$CONDA_PREFIX` with no environment active passes the
# empty string, which `DEFINED` accepts and which then searches `/include/rdkit`.
if (NOT RDKIT_PREFIX)
    if (DEFINED ENV{CONDA_PREFIX})
        set(RDKIT_PREFIX "$ENV{CONDA_PREFIX}")
    elseif (SMALLFIELD_IN_TREE_PIXI_PREFIX AND EXISTS
                                               "${SMALLFIELD_IN_TREE_PIXI_PREFIX}/include/rdkit/GraphMol/GraphMol.h"
    )
        set(RDKIT_PREFIX "${SMALLFIELD_IN_TREE_PIXI_PREFIX}")
    endif ()
endif ()
set(RDKIT_PREFIX
    "${RDKIT_PREFIX}"
    CACHE PATH "Environment prefix providing RDKit headers and libraries"
)

if (NOT EXISTS "${RDKIT_PREFIX}/include/rdkit/GraphMol/GraphMol.h")
    message(FATAL_ERROR "RDKit headers not found under '${RDKIT_PREFIX}'.  Run `pixi install`, or pass "
                        "-DRDKIT_PREFIX=/path/to/environment."
    )
endif ()

set(SMALLFIELD_RDKIT_COMPONENTS
    Alignment
    ChemTransforms
    Depictor
    Descriptors
    DistGeomHelpers
    DistGeometry
    EigenSolvers
    ForceField
    ForceFieldHelpers
    GraphMol
    MolStandardize
    MolTransforms
    RDGeneral
    SmilesParse
    SubstructMatch
)

# One imported shared library per component, bundled behind a single interface target so consumers name one dependency
# rather than fifteen, and so the list is a real edge in the target graph instead of a directory-scope variable that
# silently evaluates to nothing when read from elsewhere.
if (NOT TARGET smallfield_rdkit)
    add_library(smallfield_rdkit INTERFACE IMPORTED GLOBAL)
    foreach (_component IN LISTS SMALLFIELD_RDKIT_COMPONENTS)
        if (NOT TARGET RDKit::${_component})
            # Searched fresh on every configure rather than through a cache entry: a cached path survives a change to
            # RDKIT_PREFIX, and the build then links one environment's libraries against another's headers.
            unset(_component_path)
            file(GLOB _component_path "${RDKIT_PREFIX}/lib/libRDKit${_component}.so"
                 "${RDKIT_PREFIX}/lib/libRDKit${_component}.dylib" "${RDKIT_PREFIX}/lib/RDKit${_component}.lib"
            )
            if (NOT _component_path)
                message(FATAL_ERROR "RDKit component RDKit${_component} not found in ${RDKIT_PREFIX}/lib")
            endif ()
            list(GET _component_path 0 _component_path)
            add_library(RDKit::${_component} SHARED IMPORTED GLOBAL)
            set_target_properties(
                RDKit::${_component}
                PROPERTIES IMPORTED_LOCATION "${_component_path}"
                           INTERFACE_INCLUDE_DIRECTORIES "${RDKIT_PREFIX}/include/rdkit;${RDKIT_PREFIX}/include"
            )
        endif ()
        target_link_libraries(smallfield_rdkit INTERFACE RDKit::${_component})
    endforeach ()
endif ()
