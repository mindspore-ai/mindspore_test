file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        include(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/cmake/dataset.cmake)
        list(APPEND IGNORE_FILE_LIST "basic_tokenizer_op.cc")
        list(APPEND IGNORE_FILE_LIST "bert_tokenizer_op.cc")
        list(APPEND IGNORE_FILE_LIST "case_fold_op.cc")
        list(APPEND IGNORE_FILE_LIST "filter_wikipedia_xml_op.cc")
        list(APPEND IGNORE_FILE_LIST "normalize_utf8_op.cc")
        list(APPEND IGNORE_FILE_LIST "regex_replace_op.cc")
        list(APPEND IGNORE_FILE_LIST "regex_tokenizer_op.cc")
        list(APPEND IGNORE_FILE_LIST "unicode_script_tokenizer_op.cc")
        list(APPEND IGNORE_FILE_LIST "whitespace_tokenizer_op.cc")
        set(EXCLUDE_FILES "")
        foreach(IGNORE_FILE ${IGNORE_FILE_LIST})
                if(EXCLUDE_FILES)
                        set(EXCLUDE_FILES "${EXCLUDE_FILES}|${IGNORE_FILE}")
                else()
                        set(EXCLUDE_FILES "${IGNORE_FILE}")
                endif()
        endforeach()
        set(TEXT_KERNELS ${CMAKE_BINARY_DIR}/merge/mindspore/minddata/text/kernels)
        merge_minddata_files(${CMAKE_CURRENT_SOURCE_DIR}/ ${TEXT_KERNELS} dataset_text_kernel
            "${EXCLUDE_FILES}" "" TRUE)
        file(GLOB TEXT_KERNEL_MERGE_LIST "${TEXT_KERNELS}/*.cc")
        add_library(text-kernels OBJECT ${TEXT_KERNEL_MERGE_LIST})
else()
        set(ICU_DEPEND_FILES
                basic_tokenizer_op.cc
                bert_tokenizer_op.cc
                case_fold_op.cc
                filter_wikipedia_xml_op.cc
                normalize_utf8_op.cc
                regex_replace_op.cc
                regex_tokenizer_op.cc
                unicode_script_tokenizer_op.cc
                whitespace_tokenizer_op.cc)
        add_library(text-kernels OBJECT
                add_token_op.cc
                data_utils.cc
                lookup_op.cc
                jieba_tokenizer_op.cc
                tokenizer_op.cc
                unicode_char_tokenizer_op.cc
                ngram_op.cc
                sliding_window_op.cc
                wordpiece_tokenizer_op.cc
                truncate_op.cc
                truncate_sequence_pair_op.cc
                to_number_op.cc
                to_vectors_op.cc
                sentence_piece_tokenizer_op.cc
                ${ICU_DEPEND_FILES}
                )
endif()
