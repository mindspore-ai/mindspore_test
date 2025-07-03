# rename foo.cc to foo_1.cc, or to foo_2.cc if foo_1.cc exists, if it has more than 10000 lines
function(check_line_number FILE_PATH)
    set(MAX_LINES 10000)
    file(READ ${FILE_PATH} FILE_CONTENT)
    string(REGEX MATCHALL "\n" LINES ${FILE_CONTENT})
    list(LENGTH LINES LINES_COUNT)
    if(${LINES_COUNT} GREATER ${MAX_LINES})
        set(suffix 1)
        get_filename_component(FILE_PATH_NO_EXT ${FILE_PATH} NAME_WE)
        get_filename_component(FILE_EXT ${FILE_PATH} EXT)
        get_filename_component(FILE_DIR ${FILE_PATH} DIRECTORY)
        while(EXISTS "${FILE_DIR}/${FILE_PATH_NO_EXT}${suffix}${FILE_EXT}")
            math(EXPR suffix "${suffix} + 1")
        endwhile()
        set(NEW_FILE_PATH "${FILE_DIR}/${FILE_PATH_NO_EXT}${suffix}${FILE_EXT}")
        file(RENAME ${FILE_PATH} ${NEW_FILE_PATH})
    endif()
endfunction()

function(merge_ops_files SRC_DIR OUT_FILE_FOLDER OUT_FILE_PREFIX EXCLUDE_FILES_PATTERN)
    message(STATUS "[merge_ops_files] From ${SRC_DIR} to ${OUT_FILE_FOLDER}, exclude files: ${EXCLUDE_FILES_PATTERN}")
    set(MAX_TIMESTAMP "000000000000.00")

    set(RECURSIVE TRUE)
    if(ARGN) # optional argument indicating whether glob file NOT recursively
        if(${ARGN})
            set(RECURSIVE FALSE)
        endif()
    endif()

    # Use either GLOB_RECURSE or GLOB based on RECURSIVE parameter
    if(${RECURSIVE})
        file(GLOB_RECURSE SRC_LIST ${SRC_DIR}/*.cc)
    else()
        file(GLOB SRC_LIST ${SRC_DIR}/*.cc)
    endif()

    list(SORT SRC_LIST)

    # Create temporary directory for new merged files
    set(TEMP_DIR ${OUT_FILE_FOLDER}/temp)
    file(MAKE_DIRECTORY ${TEMP_DIR})

    foreach(file_path ${SRC_LIST})
        set(orig_file_path ${file_path})
        if(NOT ${EXCLUDE_FILES_PATTERN} STREQUAL "")
            string(REGEX REPLACE ${EXCLUDE_FILES_PATTERN} "" file_path ${file_path})
        endif()

        if(EXISTS ${file_path} AND (NOT IS_DIRECTORY ${file_path}))
            # Get filename without path
            get_filename_component(filename ${file_path} NAME)
            # Get first letter and convert to lowercase
            string(SUBSTRING ${filename} 0 1 first_letter)
            string(TOLOWER ${first_letter} first_letter)

            # Update timestamp if needed
            file(TIMESTAMP ${file_path} CUR_TIMESTAMP "%Y%m%d%H%M.%S")
            string(COMPARE GREATER ${CUR_TIMESTAMP} ${MAX_TIMESTAMP} IS_GREATER)
            if(IS_GREATER)
                set(MAX_TIMESTAMP ${CUR_TIMESTAMP})
            endif()

            # Append to the temporary output file
            set(TEMP_FILE ${TEMP_DIR}/${OUT_FILE_PREFIX}_${first_letter}.cc)
            file(STRINGS ${file_path} READ_CC_CONTEXT NEWLINE_CONSUME NO_HEX_CONVERSION)
            file(APPEND ${TEMP_FILE} "#line 1 \"${file_path}\"\n")
            file(APPEND ${TEMP_FILE} ${READ_CC_CONTEXT})
            check_line_number(${TEMP_FILE})
        else()
            message(STATUS "[merge_ops_files] exclude file: ${orig_file_path}")
            continue()
        endif()
    endforeach()

    # Compare and replace files only if changed
    file(GLOB TEMP_FILES ${TEMP_DIR}/${OUT_FILE_PREFIX}_*.cc)
    foreach(temp_file ${TEMP_FILES})
        get_filename_component(filename ${temp_file} NAME)
        set(target_file ${OUT_FILE_FOLDER}/${filename})

        set(should_copy TRUE)
        if(EXISTS ${target_file})
            file(MD5 ${temp_file} TEMP_MD5)
            file(MD5 ${target_file} TARGET_MD5)
            if(${TEMP_MD5} STREQUAL ${TARGET_MD5})
                set(should_copy FALSE)
            endif()
        endif()

        if(${should_copy})
            file(COPY ${temp_file} DESTINATION ${OUT_FILE_FOLDER})
            execute_process(COMMAND touch -c -t ${MAX_TIMESTAMP} ${target_file})
            message(STATUS "[merge_ops_files] Updated ${filename} (MD5 changed)")
        endif()
    endforeach()

    # Check for and remove obsolete files
    # NOTE: this method won't be able to remove those files if the merge function is not called with the same
    # OUT_FILE_PREFIX anymore. We decided not to handle this situation because it would complicate the merge process
    # and it won't affect the CI. In such cases, manual cleanup should be done.
    file(GLOB EXISTING_FILES ${OUT_FILE_FOLDER}/${OUT_FILE_PREFIX}_*.cc)
    foreach(existing_file ${EXISTING_FILES})
        get_filename_component(existing_filename ${existing_file} NAME)
        set(temp_file ${TEMP_DIR}/${existing_filename})
        if(NOT EXISTS ${temp_file})
            file(REMOVE ${existing_file})
            message(STATUS "[merge_ops_files] Removed obsolete file: ${existing_filename}")
        endif()
    endforeach()

    # Cleanup temporary directory
    file(REMOVE_RECURSE ${TEMP_DIR})
endfunction()
