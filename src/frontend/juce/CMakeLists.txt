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
    )
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
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

    # Configure macOS link flags for dynamic library bundling
    if (APPLE)
        set_target_properties(${target_name} PROPERTIES
            BUILD_WITH_INSTALL_RPATH FALSE
            LINK_FLAGS "-Wl,-rpath,@loader_path/"
        )

        # Post-build codesigning for JUCE targets using SIGN_ID
        if (NOT "${SIGN_ID}" STREQUAL "-")
            get_target_property(active_targets ${target_name} JUCE_ACTIVE_PLUGIN_TARGETS)
            if (active_targets)
                foreach(sub_target IN LISTS active_targets)
                    add_custom_command(
                        TARGET ${sub_target} POST_BUILD
                        COMMAND codesign --force --deep --sign "${SIGN_ID}" "$<TARGET_PROPERTY:${sub_target},JUCE_PLUGIN_ARTEFACT_FILE>"
                        COMMENT "Codesigning ${sub_target} with ${SIGN_ID}"
                        VERBATIM
                    )
                endforeach()
            endif()
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
