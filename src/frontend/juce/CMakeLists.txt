cmake_minimum_required(VERSION 3.15)
project(nn_bending VERSION 1.6.0)

# Helper function to configure a JUCE target with shared settings
function(configure_nn_bending_target target_name)
    juce_generate_juce_header(${target_name})

    # Source files
    target_sources(${target_name} PRIVATE
        PluginProcessor.cpp
        PluginProcessor.h
        PluginEditor.cpp
        PluginEditor.h
    )

    # Include directories
    target_include_directories(${target_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/backend"
        "${CMAKE_SOURCE_DIR}/shared"
        "${CMAKE_BINARY_DIR}/_deps"
    )

    # Compile flags and options
    target_compile_features(${target_name} PRIVATE cxx_std_20)
    target_compile_definitions(${target_name} PRIVATE
        JUCE_IGNORE_VST3_MISMATCHED_PARAMETER_ID_WARNING=1
        NOMINMAX=1
    )
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        COPY_PLUGIN_AFTER_BUILD OFF
    )

    # Link dependencies
    target_link_libraries(${target_name} PRIVATE
        backend
        "${TORCH_LIBRARIES}"
        nlohmann_json::nlohmann_json
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_audio_formats
        juce::juce_audio_devices
        juce::juce_audio_basics
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
        juce::juce_data_structures
        juce::juce_events
        juce::juce_core
    )

    # Configure macOS link flags for dynamic library linking
    if (APPLE)
        set_target_properties(${target_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "/Library/Application Support/nn_bending/lib;${TORCH_INSTALL_PREFIX}/lib;${torch_dir}/libtorch/lib;${CONDA_ENV_PATH}/lib;@loader_path/;@loader_path/../Frameworks"
        )

        # Post-build library handling and local plugin installation
        get_target_property(active_targets ${target_name} JUCE_ACTIVE_PLUGIN_TARGETS)
        if (active_targets)
            foreach(sub_target IN LISTS active_targets)
                set_target_properties(${sub_target} PROPERTIES
                    BUILD_WITH_INSTALL_RPATH TRUE
                    INSTALL_RPATH "/Library/Application Support/nn_bending/lib;${TORCH_INSTALL_PREFIX}/lib;${torch_dir}/libtorch/lib;${CONDA_ENV_PATH}/lib;@loader_path/;@loader_path/../Frameworks"
                )

                # Determine destination path for local installation
                set(COPY_DEST "")
                if (sub_target MATCHES ".*_AU$")
                    set(COPY_DEST "$ENV{HOME}/Library/Audio/Plug-Ins/Components/nn~ Bending.component")
                elseif (sub_target MATCHES ".*_VST3$")
                    set(COPY_DEST "$ENV{HOME}/Library/Audio/Plug-Ins/VST3/nn~ Bending.vst3")
                endif()

                if (NOT "${COPY_DEST}" STREQUAL "")
                    if (BUNDLE_DEPENDENCIES)
                        add_custom_command(
                            TARGET ${sub_target} POST_BUILD
                            COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_SOURCE_DIR}/../install/dylib_fix.py"
                                -p "$<TARGET_FILE:${sub_target}>"
                                -l "${torch_dir}/libtorch" "${CMAKE_BINARY_DIR}/_deps" "${CMAKE_SOURCE_DIR}/../env" "${HOMEBREW_PREFIX}"
                                --sign_id "${SIGN_ID}"
                                --entitlements "${CMAKE_CURRENT_SOURCE_DIR}/entitlements.plist"
                            COMMAND ${CMAKE_COMMAND} -E rm -rf "${COPY_DEST}"
                            COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_FILE_DIR:${sub_target}>/../.." "${COPY_DEST}"
                            COMMENT "Bundling libraries, codesigning, and installing ${sub_target} to local plugin folder"
                            VERBATIM
                        )
                    else()
                        add_custom_command(
                            TARGET ${sub_target} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E rm -rf "${COPY_DEST}"
                            COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_FILE_DIR:${sub_target}>/../.." "${COPY_DEST}"
                            COMMAND codesign --force --deep --sign "${SIGN_ID}" "${COPY_DEST}"
                            COMMENT "Installing and codesigning ${sub_target} in local plugin folder"
                            VERBATIM
                        )
                    endif()
                else()
                    if (BUNDLE_DEPENDENCIES)
                        add_custom_command(
                            TARGET ${sub_target} POST_BUILD
                            COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_SOURCE_DIR}/../install/dylib_fix.py"
                                -p "$<TARGET_FILE:${sub_target}>"
                                -l "${torch_dir}/libtorch" "${CMAKE_BINARY_DIR}/_deps" "${CMAKE_SOURCE_DIR}/../env" "${HOMEBREW_PREFIX}"
                                --sign_id "${SIGN_ID}"
                                --entitlements "${CMAKE_CURRENT_SOURCE_DIR}/entitlements.plist"
                            COMMENT "Bundling libraries and codesigning ${sub_target}"
                            VERBATIM
                        )
                    else()
                        add_custom_command(
                            TARGET ${sub_target} POST_BUILD
                            COMMAND codesign --force --deep --sign "${SIGN_ID}" "$<TARGET_FILE_DIR:${sub_target}>/../.."
                            COMMENT "Codesigning ${sub_target}"
                            VERBATIM
                        )
                    endif()
                endif()
            endforeach()
        endif()
    elseif (WIN32)
        get_target_property(active_targets ${target_name} JUCE_ACTIVE_PLUGIN_TARGETS)
        if (active_targets)
            file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
            foreach(sub_target IN LISTS active_targets)
                # Copy LibTorch DLLs next to the target executable or plugin binary
                if (TORCH_DLLS)
                    add_custom_command(
                        TARGET ${sub_target} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            ${TORCH_DLLS}
                            "$<TARGET_FILE_DIR:${sub_target}>"
                        COMMENT "Copying LibTorch DLLs to $<TARGET_FILE_DIR:${sub_target}>"
                    )
                endif()

                # For VST3 plugin, install bundle to system and/or user VST3 directory
                if (sub_target MATCHES ".*_VST3$")
                    set(SRC_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_artefacts/$<CONFIG>/VST3/nn~ Bending.vst3")
                    file(TO_CMAKE_PATH "$ENV{CommonProgramFiles}" CPF_DIR)
                    set(SYS_DEST "${CPF_DIR}/VST3/nn~ Bending.vst3")

                    file(TO_CMAKE_PATH "$ENV{LOCALAPPDATA}" LAPPDATA_DIR)
                    set(USER_DEST "${LAPPDATA_DIR}/Programs/Common/VST3/nn~ Bending.vst3")

                    set(CUSTOM_DEST "")
                    if (DEFINED ENV{VST3_INSTALL_PATH})
                        file(TO_CMAKE_PATH "$ENV{VST3_INSTALL_PATH}" VST3_DIR)
                        set(CUSTOM_DEST "${VST3_DIR}/nn~ Bending.vst3")
                    endif()

                    add_custom_command(
                        TARGET ${sub_target} POST_BUILD
                        COMMAND ${CMAKE_COMMAND}
                            -D "SRC_BUNDLE=${SRC_BUNDLE}"
                            -D "SYS_DEST=${SYS_DEST}"
                            -D "USER_DEST=${USER_DEST}"
                            -D "CUSTOM_DEST=${CUSTOM_DEST}"
                            -P "${CMAKE_SOURCE_DIR}/../install/install_vst3_windows.cmake"
                        COMMENT "Installing VST3 plugin bundle..."
                        VERBATIM
                    )
                endif()
            endforeach()
        endif()
    endif()
endfunction()

# -------------------------------------------------------------------
# Plugin target (VST3 / AU)
# -------------------------------------------------------------------
juce_add_plugin(nn_bending_plugin
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    PLUGIN_MANUFACTURER_CODE FLTT
    PLUGIN_CODE Nnbd
    FORMATS VST3 AU
    PRODUCT_NAME "nn~ Bending"
    COMPANY_NAME "leofltt"
)
configure_nn_bending_target(nn_bending_plugin)

# -------------------------------------------------------------------
# Standalone target
# -------------------------------------------------------------------
juce_add_plugin(nn_bending_standalone
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    PLUGIN_MANUFACTURER_CODE FLTT
    PLUGIN_CODE Nnbs
    FORMATS Standalone
    PRODUCT_NAME "nn~ Bending"
    COMPANY_NAME "leofltt"
)
configure_nn_bending_target(nn_bending_standalone)
