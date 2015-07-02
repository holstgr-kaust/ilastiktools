

#--------------------------------------------------------------
# Add Library
#--------------------------------------------------------------
file(GLOB_RECURSE headers ${CMAKE_SOURCE_DIR}/include/*.hxx ${CMAKE_CURRENT_SOURCE_DIR}/*.hxx)

add_library( _core SHARED
    core.cxx ${headers}
)
#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
find_library(RT rt)
target_link_libraries( _core
    ${PYTHON_LIBRARIES}
    ${Boost_PYTHON_LIBRARIES}
)

IF(PYTHON_PLATFORM MATCHES "windows$" OR WIN32)
    SET_TARGET_PROPERTIES( _core PROPERTIES OUTPUT_NAME "_core" PREFIX "" SUFFIX ".pyd")
ELSEIF(APPLE)
    SET_TARGET_PROPERTIES( _core PROPERTIES OUTPUT_NAME "_core" PREFIX "" SUFFIX ".so")
ELSE()
    SET_TARGET_PROPERTIES( _core PROPERTIES OUTPUT_NAME "_core" PREFIX "")
ENDIF()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py  DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

install(FILES __init__.py
        DESTINATION ${PYTHON_SITE_PACKAGES}/ilastiktools)

IF(PYTHON_PLATFORM MATCHES "windows$" OR WIN32)
    install(TARGETS _core
            RUNTIME DESTINATION ${PYTHON_SITE_PACKAGES}/ilastiktools)
ELSE()
    install(TARGETS _core
            LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/ilastiktools)
ENDIF()

