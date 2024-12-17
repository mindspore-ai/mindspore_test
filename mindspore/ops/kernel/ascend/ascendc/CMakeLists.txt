project(ascendc_opp)

macro(run_shell_script script)
    execute_process(
            COMMAND bash ${script}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE result
            OUTPUT_VARIABLE output
            ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Custom ascendc opp compile failed, please see\
        '${CMAKE_SOURCE_DIR}/custom_workspace/custom_compiler/build.log' for more detailed info.")
    endif()
    message(STATUS "Custom ascendc opp compile succeed.")
endmacro()

run_shell_script("${CMAKE_SOURCE_DIR}/scripts/ascendc/ascendc_compile.sh")