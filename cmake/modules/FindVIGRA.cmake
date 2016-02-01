# This module finds an installed Vigra package. Since we only need the headers
# we don't search for libraries.
#
# It sets the following variables:
#  VIGRA_FOUND              - Set to false, or undefined, if vigra isn't found.
#  VIGRA_INCLUDE_DIR        - Vigra include directory.

# config_version.hxx only present, after build of Vigra
FIND_PATH(VIGRA_INCLUDE_DIR vigra/config_version.hxx PATHS $ENV{VIGRA_ROOT}/include ENV CPLUS_INCLUDE_PATH)

# handle the QUIETLY and REQUIRED arguments and set VIGRA_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(VIGRA DEFAULT_MSG VIGRA_INCLUDE_DIR)

MARK_AS_ADVANCED( VIGRA_INCLUDE_DIR VIGRA_IMPEX_LIBRARY VIGRA_IMPEX_LIBRARY_DIR VIGRA_NUMPY_CORE_LIBRARY)
