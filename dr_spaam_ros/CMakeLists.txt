cmake_minimum_required(VERSION 3.0)

project(dr_spaam_ros)

find_package(catkin REQUIRED
  COMPONENTS
)

catkin_package(
  CATKIN_DEPENDS
)

catkin_python_setup()

set(MODELS_FILE best_ckpt.pth)
set(MODELS_DEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dr_spaam/model_ckpt/")
set(MODELS_DOWNLOAD_URL https://www.dropbox.com/s/v1ec37xctih9p7c/best_ckpt.pth?dl=0)

add_custom_command(OUTPUT "${MODELS_DEST_DIR}/${MODELS_FILE}"
  COMMAND curl -L --progress-bar -o "${MODELS_DEST_DIR}/${MODELS_FILE}" "${MODELS_DOWNLOAD_URL}"
  COMMENT "downloading additional models ${CATKIN_PACKAGE_SHARE_DESTINATION}"
  VERBATIM
)

add_custom_target(get_models ALL
  DEPENDS 
  "${MODELS_DEST_DIR}/${MODELS_FILE}"
)

add_custom_target(dr_spaam_model ALL
    COMMAND echo "dependency for installation"
    DEPENDS "${MODELS_DEST_DIR}/${MODELS_FILE}"
    VERBATIM
)

add_dependencies(get_models dr_spaam_model)

# INSTALL

install(PROGRAMS scripts/node.py
  scripts/drow_data_converter.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS src/dr_spaam_ros/dr_spaam_ros.py
  DESTINATION "${CATKIN_PACKAGE_BIN_DESTINATION}/dr_spaam_ros"
)

install(DIRECTORY dr_spaam
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
