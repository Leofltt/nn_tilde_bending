# install_vst3_windows.cmake
# Post-build helper to install VST3 plugin bundle to Windows VST3 directories.
#
# Variables expected via -D:
#   SRC_BUNDLE   : Full path to built "nn~ Bending.vst3" folder
#   CUSTOM_DEST  : Optional override from VST3_INSTALL_PATH
#   SYS_DEST     : System VST3 destination (e.g. C:/Program Files/Common Files/VST3/nn~ Bending.vst3)
#   USER_DEST    : User VST3 destination (e.g. %LOCALAPPDATA%/Programs/Common/VST3/nn~ Bending.vst3)

if (DEFINED CUSTOM_DEST AND NOT "${CUSTOM_DEST}" STREQUAL "" AND NOT "${CUSTOM_DEST}" STREQUAL "OFF" AND NOT "${CUSTOM_DEST}" STREQUAL "NONE")
    message(STATUS "Installing VST3 plugin to custom path: ${CUSTOM_DEST}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E rm -rf "${CUSTOM_DEST}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${SRC_BUNDLE}" "${CUSTOM_DEST}"
        RESULT_VARIABLE res
    )
    if (NOT res EQUAL 0)
        message(WARNING "Failed to copy VST3 plugin to ${CUSTOM_DEST}")
    endif()
    return()
endif()

set(INSTALLED_SYS FALSE)

# 1. Try installing to the standard System VST3 directory (where Ableton Live and other DAWs look by default)
if (DEFINED SYS_DEST AND NOT "${SYS_DEST}" STREQUAL "")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${SRC_BUNDLE}" "${SYS_DEST}"
        RESULT_VARIABLE res
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if (res EQUAL 0)
        set(INSTALLED_SYS TRUE)
        message(STATUS "Successfully installed VST3 plugin to System directory: ${SYS_DEST}")
    endif()
endif()

# 2. Also install to the per-user VST3 directory (works without admin rights, scanned by Reaper, etc.)
if (DEFINED USER_DEST AND NOT "${USER_DEST}" STREQUAL "")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${SRC_BUNDLE}" "${USER_DEST}"
        RESULT_VARIABLE user_res
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if (user_res EQUAL 0)
        message(STATUS "Successfully installed VST3 plugin to User directory: ${USER_DEST}")
    endif()
endif()

if (NOT INSTALLED_SYS)
    message(STATUS "Note: Plugin installed to User VST3 directory (${USER_DEST}).")
    message(STATUS "Ableton Live only scans '${SYS_DEST}' by default.")
    message(STATUS "To use in Live without installing as Admin, in Live go to: Preferences > Plug-ins > enable 'Use VST3 Plug-In Custom Folder' and select '${USER_DEST}/..'")
    message(STATUS "Alternatively, run an elevated PowerShell once to grant write permission: icacls \"C:\\Program Files\\Common Files\\VST3\" /grant Users:(OI)(CI)M")
endif()
