cmake_minimum_required(VERSION 2.8.3)
project(dr_spaam_ros)

find_package(catkin REQUIRED
  COMPONENTS
)

catkin_package(
  CATKIN_DEPENDS
)

install(DIRECTORY ckpts
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_python_setup()

set(MODELS_FILE best_ckpt.pth)
set(MODELS_DEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ckpts/)
set(MODELS_DOWNLOAD_URL https://www.dropbox.com/s/v1ec37xctih9p7c/best_ckpt.pth?dl=0)



add_custom_command(OUTPUT ${MODELS_FILE}
  COMMAND curl --progress-bar -o "${MODELS_FILE}" "${MODELS_DOWNLOAD_URL}"
  COMMENT "downloading additional models ${CATKIN_PACKAGE_SHARE_DESTINATION}"
  VERBATIM
)

add_custom_target(zoo ALL
    COMMAND echo "This is ALL target 'zoo', and it depends on ${TEST_FILE}"
    # If the file exists, then commands related to that file won't be executed
    # DONOT let other target depends on the same OUTPUT as current target,
    #   or it may be bad when doing parallel make
    DEPENDS ${MODELS_FILE}

    # to make quotes printable,for example
    VERBATIM
)

add_custom_target(get_models ALL
  DEPENDS 
  ${MODELS_FILE}
)

add_dependencies(get_models zoo)
