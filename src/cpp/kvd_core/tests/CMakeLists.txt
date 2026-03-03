find_package(GTest CONFIG REQUIRED)

add_executable(kvd_tests
  test_smoke.cpp
)

target_link_libraries(kvd_tests
  PRIVATE
    axon_engine
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(kvd_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

add_test(NAME kvd_tests COMMAND kvd_tests)

add_custom_command(TARGET kvd_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:axon_engine> $<TARGET_FILE_DIR:kvd_tests>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/LIEF.dll" $<TARGET_FILE_DIR:kvd_tests>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/lib_lightgbm.dll" $<TARGET_FILE_DIR:kvd_tests>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/fmt.dll" $<TARGET_FILE_DIR:kvd_tests>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/spdlog.dll" $<TARGET_FILE_DIR:kvd_tests>
)
