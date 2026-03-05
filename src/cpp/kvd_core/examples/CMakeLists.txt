add_executable(kvd_scan
  scan_path.cpp
)

target_link_libraries(kvd_scan
  PRIVATE
    axon_engine
)

add_custom_command(TARGET kvd_scan POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:axon_engine> $<TARGET_FILE_DIR:kvd_scan>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/LIEF.dll" $<TARGET_FILE_DIR:kvd_scan>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/lib_lightgbm.dll" $<TARGET_FILE_DIR:kvd_scan>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/fmt.dll" $<TARGET_FILE_DIR:kvd_scan>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/spdlog.dll" $<TARGET_FILE_DIR:kvd_scan>
)

add_executable(kvd_scan_loader
  kvd_scan_loader.cpp
)

target_include_directories(kvd_scan_loader
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

add_custom_command(TARGET kvd_scan_loader POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:axon_engine> $<TARGET_FILE_DIR:kvd_scan_loader>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/LIEF.dll" $<TARGET_FILE_DIR:kvd_scan_loader>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/lib_lightgbm.dll" $<TARGET_FILE_DIR:kvd_scan_loader>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/fmt.dll" $<TARGET_FILE_DIR:kvd_scan_loader>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/spdlog.dll" $<TARGET_FILE_DIR:kvd_scan_loader>
)

add_executable(kvd_dll_selftest
  kvd_dll_selftest.cpp
)

target_include_directories(kvd_dll_selftest
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

add_custom_command(TARGET kvd_dll_selftest POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:signature_engine> $<TARGET_FILE_DIR:kvd_dll_selftest>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:axon_engine> $<TARGET_FILE_DIR:kvd_dll_selftest>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/LIEF.dll" $<TARGET_FILE_DIR:kvd_dll_selftest>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/lib_lightgbm.dll" $<TARGET_FILE_DIR:kvd_dll_selftest>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/fmt.dll" $<TARGET_FILE_DIR:kvd_dll_selftest>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/spdlog.dll" $<TARGET_FILE_DIR:kvd_dll_selftest>
)
