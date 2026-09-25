#!/usr/bin/env bash
# ==============================================================================
# package_macos.sh
# Packages macOS audio plugins (VST3, AU) and Standalone App into a signed/notarized
# Apple Component Installer (.pkg).
#
# Supports reading configuration from .env in repository root.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ------------------------------------------------------------------------------
# 1. Load .env if present
# ------------------------------------------------------------------------------
if [ -f "${REPO_ROOT}/.env" ]; then
    echo "==> Loading environment variables from ${REPO_ROOT}/.env"
    # Export non-comment lines
    set -a
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/.env"
    set +a
fi

# ------------------------------------------------------------------------------
# 2. Configuration & Defaults
# ------------------------------------------------------------------------------
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
JUCE_BUILD_DIR="${BUILD_DIR}/frontend/juce"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/dist}"
STAGE_DIR="${BUILD_DIR}/installer_stage"
ENTITLEMENTS="${REPO_ROOT}/src/frontend/juce/entitlements.plist"

# Signing Identifiers
# SIGN_ID can be "Developer ID Application: ...", a hash, or "-" for ad-hoc
SIGN_ID="${SIGN_ID:--}"
# INSTALLER_SIGN_ID can be "Developer ID Installer: ..." (optional)
INSTALLER_SIGN_ID="${INSTALLER_SIGN_ID:-}"

# Notarization credentials (supports Melatonin / Pamplejuce convention)
KEYCHAIN_PROFILE="${KEYCHAIN_PROFILE:-nnbend-notary}"
APPLE_ID="${NOTARIZATION_USERNAME:-${APPLE_ID:-}}"
APPLE_ID_PASSWORD="${NOTARIZATION_PASSWORD:-${APPLE_ID_PASSWORD:-}}"
APPLE_TEAM_ID="${TEAM_ID:-${APPLE_TEAM_ID:-}}"

# Bundle and Package metadata
BUNDLE_PREFIX="${BUNDLE_PREFIX:-com.leofltt.nnbending}"
VERSION="${VERSION:-}"
if [ -z "${VERSION}" ]; then
    if git -C "${REPO_ROOT}" describe --tags --always >/dev/null 2>&1; then
        VERSION="$(git -C "${REPO_ROOT}" describe --tags --always | sed 's/^v//')"
    else
        VERSION="1.6.0"
    fi
fi
PKG_NAME="nn_bending_macOS_${VERSION}.pkg"

# Dependencies bundling flag (1 to run dylib_fix.py, 0 to skip)
BUNDLE_DEPS="${BUNDLE_DEPS:-1}"

echo "=================================================================="
echo " Packaging nn~ Bending macOS Installer"
echo " Version:           ${VERSION}"
echo " App Sign Identity: ${SIGN_ID}"
echo " Pkg Sign Identity: ${INSTALLER_SIGN_ID:-<none (unsigned installer)>}"
echo " Keychain Profile:  ${KEYCHAIN_PROFILE:-<none>}"
echo " Bundle Deps:       ${BUNDLE_DEPS}"
echo " Output Package:    ${OUTPUT_DIR}/${PKG_NAME}"
echo "=================================================================="

# Check build artifacts (handles both multi-config like Xcode with Release/ and single-config like Ninja/Makefiles)
CONFIG="${CMAKE_BUILD_TYPE:-Release}"
if [ -d "${JUCE_BUILD_DIR}/nn_bending_plugin_artefacts/${CONFIG}/VST3/nn~ Bending.vst3" ]; then
    VST3_SRC="${JUCE_BUILD_DIR}/nn_bending_plugin_artefacts/${CONFIG}/VST3/nn~ Bending.vst3"
    AU_SRC="${JUCE_BUILD_DIR}/nn_bending_plugin_artefacts/${CONFIG}/AU/nn~ Bending.component"
    APP_SRC="${JUCE_BUILD_DIR}/nn_bending_standalone_artefacts/${CONFIG}/Standalone/nn~ Bending.app"
elif [ -d "${JUCE_BUILD_DIR}/nn_bending_plugin_artefacts/Release/VST3/nn~ Bending.vst3" ]; then
    VST3_SRC="${JUCE_BUILD_DIR}/nn_bending_plugin_artefacts/Release/VST3/nn~ Bending.vst3"
    AU_SRC="${JUCE_BUILD_DIR}/nn_bending_plugin_artefacts/Release/AU/nn~ Bending.component"
    APP_SRC="${JUCE_BUILD_DIR}/nn_bending_standalone_artefacts/Release/Standalone/nn~ Bending.app"
elif [ -d "${JUCE_BUILD_DIR}/nn_bending_plugin_artefacts/VST3/nn~ Bending.vst3" ]; then
    VST3_SRC="${JUCE_BUILD_DIR}/nn_bending_plugin_artefacts/VST3/nn~ Bending.vst3"
    AU_SRC="${JUCE_BUILD_DIR}/nn_bending_plugin_artefacts/AU/nn~ Bending.component"
    APP_SRC="${JUCE_BUILD_DIR}/nn_bending_standalone_artefacts/Standalone/nn~ Bending.app"
else
    VST3_SRC="${JUCE_BUILD_DIR}/nn_bending_plugin_artefacts/Release/VST3/nn~ Bending.vst3"
    AU_SRC="${JUCE_BUILD_DIR}/nn_bending_plugin_artefacts/Release/AU/nn~ Bending.component"
    APP_SRC="${JUCE_BUILD_DIR}/nn_bending_standalone_artefacts/Release/Standalone/nn~ Bending.app"
fi

for src in "${VST3_SRC}" "${AU_SRC}" "${APP_SRC}"; do
    if [ ! -d "${src}" ]; then
        echo "Error: Missing build artifact: ${src}"
        echo "Please build the project first (e.g. cmake --build build --target all)"
        exit 1
    fi
done

# ------------------------------------------------------------------------------
# 3. Prepare Staging Directory
# ------------------------------------------------------------------------------
rm -rf "${STAGE_DIR}"
SHARED_LIB_DEST="/Library/Application Support/nn_bending/lib"
mkdir -p "${STAGE_DIR}/support_root${SHARED_LIB_DEST}"
mkdir -p "${STAGE_DIR}/vst3_root/Library/Audio/Plug-Ins/VST3"
mkdir -p "${STAGE_DIR}/au_root/Library/Audio/Plug-Ins/Components"
mkdir -p "${STAGE_DIR}/app_root/Applications"
mkdir -p "${STAGE_DIR}/packages"
mkdir -p "${OUTPUT_DIR}"

echo "==> Copying binaries to staging..."
cp -R "${VST3_SRC}" "${STAGE_DIR}/vst3_root/Library/Audio/Plug-Ins/VST3/"
cp -R "${AU_SRC}"   "${STAGE_DIR}/au_root/Library/Audio/Plug-Ins/Components/"
cp -R "${APP_SRC}"  "${STAGE_DIR}/app_root/Applications/"

STAGE_VST3="${STAGE_DIR}/vst3_root/Library/Audio/Plug-Ins/VST3/nn~ Bending.vst3"
STAGE_AU="${STAGE_DIR}/au_root/Library/Audio/Plug-Ins/Components/nn~ Bending.component"
STAGE_APP="${STAGE_DIR}/app_root/Applications/nn~ Bending.app"
STAGE_SUPPORT_LIB="${STAGE_DIR}/support_root${SHARED_LIB_DEST}"

# ------------------------------------------------------------------------------
# 4. Bundle Dynamic Dependencies (LibTorch, C10, etc.)
# ------------------------------------------------------------------------------
# We place shared LibTorch runtime dylibs in /Library/Application Support/nn_bending/lib
# so that when a DAW loads both AU and VST3 in the same process, dyld resolves to
# the exact same dylib instances and does not duplicate operator registrations.
if [ "${BUNDLE_DEPS}" = "1" ]; then
    echo "==> Bundling shared runtime dependencies into ${SHARED_LIB_DEST}..."
    LIB_SEARCH_PATHS=(
        "${REPO_ROOT}/libtorch"
        "${BUILD_DIR}/../torch/libtorch"
        "${BUILD_DIR}/_deps"
        "${REPO_ROOT}/env"
        "$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
    )

    DYLIB_FIX_SCRIPT="${REPO_ROOT}/install/dylib_fix.py"
    if [ -f "${DYLIB_FIX_SCRIPT}" ]; then
        # Check if Python is available
        PYTHON_CMD="python3"
        if ! command -v python3 >/dev/null 2>&1; then
            if [ -f "${REPO_ROOT}/env/bin/python" ]; then
                PYTHON_CMD="${REPO_ROOT}/env/bin/python"
            fi
        fi

        echo "-> Fixing dependencies for VST3 into shared support dir..."
        "${PYTHON_CMD}" "${DYLIB_FIX_SCRIPT}" \
            -p "${STAGE_VST3}/Contents/MacOS/nn~ Bending" \
            -o "${STAGE_SUPPORT_LIB}" \
            -l "${LIB_SEARCH_PATHS[@]}" \
            --use_rpath \
            --keep_rpaths "${SHARED_LIB_DEST}" \
            --sign_id "${SIGN_ID}" \
            --entitlements "${ENTITLEMENTS}"

        echo "-> Fixing dependencies for AU into shared support dir..."
        "${PYTHON_CMD}" "${DYLIB_FIX_SCRIPT}" \
            -p "${STAGE_AU}/Contents/MacOS/nn~ Bending" \
            -o "${STAGE_SUPPORT_LIB}" \
            -l "${LIB_SEARCH_PATHS[@]}" \
            --use_rpath \
            --keep_rpaths "${SHARED_LIB_DEST}" \
            --sign_id "${SIGN_ID}" \
            --entitlements "${ENTITLEMENTS}"

        echo "-> Fixing dependencies for Standalone App into shared support dir..."
        "${PYTHON_CMD}" "${DYLIB_FIX_SCRIPT}" \
            -p "${STAGE_APP}/Contents/MacOS/nn~ Bending" \
            -o "${STAGE_SUPPORT_LIB}" \
            -l "${LIB_SEARCH_PATHS[@]}" \
            --use_rpath \
            --keep_rpaths "${SHARED_LIB_DEST}" \
            --sign_id "${SIGN_ID}" \
            --entitlements "${ENTITLEMENTS}"

        # Ensure binaries link to @rpath for shared dylibs and have SHARED_LIB_DEST in rpaths
        for bin in "${STAGE_VST3}/Contents/MacOS/nn~ Bending" \
                   "${STAGE_AU}/Contents/MacOS/nn~ Bending" \
                   "${STAGE_APP}/Contents/MacOS/nn~ Bending"; do
            # Add shared rpath if not present
            install_name_tool -add_rpath "${SHARED_LIB_DEST}" "${bin}" 2>/dev/null || true
            # Clean any references that pointed to support_root or local files
            for lib in "${STAGE_SUPPORT_LIB}"/*.dylib; do
                if [ -f "${lib}" ]; then
                    lib_name="$(basename "${lib}")"
                    # Fix ID of shared library
                    install_name_tool -id "@rpath/${lib_name}" "${lib}" 2>/dev/null || true
                    # Find any existing load command in bin ending with /lib_name and rewrite to @rpath/lib_name
                    old_refs="$(otool -L "${bin}" | awk '{print $1}' | grep "/${lib_name}$" || true)"
                    for old_ref in ${old_refs}; do
                        install_name_tool -change "${old_ref}" "@rpath/${lib_name}" "${bin}" 2>/dev/null || true
                    done
                    # Also change inter-library dependencies between shared dylibs
                    for other_lib in "${STAGE_SUPPORT_LIB}"/*.dylib; do
                        old_other_refs="$(otool -L "${other_lib}" | awk '{print $1}' | grep "/${lib_name}$" || true)"
                        for old_ref in ${old_other_refs}; do
                            install_name_tool -change "${old_ref}" "@rpath/${lib_name}" "${other_lib}" 2>/dev/null || true
                        done
                    done
                fi
            done
        done
    else
        echo "Warning: ${DYLIB_FIX_SCRIPT} not found. Skipping dependency bundling."
    fi
fi

# ------------------------------------------------------------------------------
# 5. Sign Bundles and Support Libraries
# ------------------------------------------------------------------------------
sign_bundle() {
    local target="$1"
    echo "==> Codesigning: ${target}"
    if [ -n "${SIGN_ID}" ] && [ "${SIGN_ID}" != "-" ]; then
        codesign --force --deep --options runtime --timestamp \
            --entitlements "${ENTITLEMENTS}" \
            --sign "${SIGN_ID}" "${target}"
    else
        codesign --force --deep \
            --entitlements "${ENTITLEMENTS}" \
            --sign - "${target}"
    fi
}

sign_bundle "${STAGE_VST3}"
sign_bundle "${STAGE_AU}"
sign_bundle "${STAGE_APP}"

# Sign shared libraries in support_root
if [ -d "${STAGE_SUPPORT_LIB}" ]; then
    echo "==> Codesigning shared support libraries..."
    for lib in "${STAGE_SUPPORT_LIB}"/*.dylib; do
        if [ -f "${lib}" ]; then
            sign_bundle "${lib}"
        fi
    done
fi

# ------------------------------------------------------------------------------
# 6. Build Component Packages (.pkg)
# ------------------------------------------------------------------------------
echo "==> Building component packages with pkgbuild..."

pkgbuild --root "${STAGE_DIR}/support_root" \
         --identifier "${BUNDLE_PREFIX}.support" \
         --version "${VERSION}" \
         --install-location "/" \
         "${STAGE_DIR}/packages/support.pkg"

pkgbuild --root "${STAGE_DIR}/vst3_root" \
         --identifier "${BUNDLE_PREFIX}.vst3" \
         --version "${VERSION}" \
         --install-location "/" \
         "${STAGE_DIR}/packages/vst3.pkg"

pkgbuild --root "${STAGE_DIR}/au_root" \
         --identifier "${BUNDLE_PREFIX}.au" \
         --version "${VERSION}" \
         --install-location "/" \
         "${STAGE_DIR}/packages/au.pkg"

pkgbuild --root "${STAGE_DIR}/app_root" \
         --identifier "${BUNDLE_PREFIX}.app" \
         --version "${VERSION}" \
         --install-location "/" \
         "${STAGE_DIR}/packages/app.pkg"

# ------------------------------------------------------------------------------
# 7. Synthesize Distribution XML & Build Final Product
# ------------------------------------------------------------------------------
echo "==> Synthesizing product distribution..."

cat <<EOF > "${STAGE_DIR}/distribution.xml"
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>nn~ Bending ${VERSION}</title>
    <options customize="always" require-scripts="false" hostArchitectures="arm64,x86_64"/>
    <domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>
    
    <choices-outline>
        <line choice="choice_support"/>
        <line choice="choice_vst3"/>
        <line choice="choice_au"/>
        <line choice="choice_app"/>
    </choices-outline>
    
    <choice id="choice_support" title="Shared AI Runtime" description="Installs shared LibTorch runtime libraries into /Library/Application Support/nn_bending/lib" visible="false" enabled="true" selected="true">
        <pkg-ref id="${BUNDLE_PREFIX}.support"/>
    </choice>

    <choice id="choice_vst3" title="VST3 Plugin" description="Installs nn~ Bending VST3 plugin into /Library/Audio/Plug-Ins/VST3">
        <pkg-ref id="${BUNDLE_PREFIX}.vst3"/>
    </choice>
    
    <choice id="choice_au" title="Audio Unit (AU) Plugin" description="Installs nn~ Bending Audio Unit into /Library/Audio/Plug-Ins/Components">
        <pkg-ref id="${BUNDLE_PREFIX}.au"/>
    </choice>
    
    <choice id="choice_app" title="Standalone Application" description="Installs nn~ Bending Standalone Application into /Applications">
        <pkg-ref id="${BUNDLE_PREFIX}.app"/>
    </choice>
    
    <pkg-ref id="${BUNDLE_PREFIX}.support" version="${VERSION}">support.pkg</pkg-ref>
    <pkg-ref id="${BUNDLE_PREFIX}.vst3" version="${VERSION}">vst3.pkg</pkg-ref>
    <pkg-ref id="${BUNDLE_PREFIX}.au" version="${VERSION}">au.pkg</pkg-ref>
    <pkg-ref id="${BUNDLE_PREFIX}.app" version="${VERSION}">app.pkg</pkg-ref>
</installer-gui-script>
EOF

PRODUCT_SIGN_FLAGS=()
if [ -n "${INSTALLER_SIGN_ID:-}" ]; then
    echo "==> Will sign final product package with installer certificate: ${INSTALLER_SIGN_ID}"
    PRODUCT_SIGN_FLAGS=(--sign "${INSTALLER_SIGN_ID}")
else
    echo "==> INSTALLER_SIGN_ID not set. Final product package will be unsigned (local test mode)."
fi

productbuild --distribution "${STAGE_DIR}/distribution.xml" \
             --package-path "${STAGE_DIR}/packages" \
             "${PRODUCT_SIGN_FLAGS[@]}" \
             "${OUTPUT_DIR}/${PKG_NAME}"

echo "==> Successfully created installer at: ${OUTPUT_DIR}/${PKG_NAME}"

# ------------------------------------------------------------------------------
# 8. Notarization & Stapling (Optional)
# ------------------------------------------------------------------------------
if [ -n "${APPLE_API_KEY_ID:-}" ] && [ -n "${APPLE_API_ISSUER_ID:-}" ]; then
    echo "==> Submitting package to Apple Notary Service via API Key..."
    NOTARY_CMD=(xcrun notarytool submit "${OUTPUT_DIR}/${PKG_NAME}"
        --key-id "${APPLE_API_KEY_ID}"
        --issuer "${APPLE_API_ISSUER_ID}"
        --wait)

    if [ -n "${APPLE_API_KEY_PATH:-}" ]; then
        NOTARY_CMD+=(--key "${APPLE_API_KEY_PATH}")
    elif [ -n "${APPLE_API_KEY_BASE64:-}" ]; then
        TEMP_KEY_PATH="${STAGE_DIR}/AuthKey_${APPLE_API_KEY_ID}.p8"
        echo "${APPLE_API_KEY_BASE64}" | base64 -d > "${TEMP_KEY_PATH}"
        NOTARY_CMD+=(--key "${TEMP_KEY_PATH}")
    fi

    "${NOTARY_CMD[@]}"
    echo "==> Stapling notarization ticket to package..."
    xcrun stapler staple "${OUTPUT_DIR}/${PKG_NAME}"
elif [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_ID_PASSWORD:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ]; then
    echo "==> Submitting package to Apple Notary Service via Apple ID credentials..."
    xcrun notarytool submit "${OUTPUT_DIR}/${PKG_NAME}" \
        --apple-id "${APPLE_ID}" \
        --password "${APPLE_ID_PASSWORD}" \
        --team-id "${APPLE_TEAM_ID}" \
        --wait
    echo "==> Stapling notarization ticket to package..."
    xcrun stapler staple "${OUTPUT_DIR}/${PKG_NAME}"
elif [ -n "${KEYCHAIN_PROFILE:-}" ]; then
    echo "==> Submitting package to Apple Notary Service via Keychain Profile: ${KEYCHAIN_PROFILE}..."
    xcrun notarytool submit "${OUTPUT_DIR}/${PKG_NAME}" \
        --keychain-profile "${KEYCHAIN_PROFILE}" \
        --wait
    echo "==> Stapling notarization ticket to package..."
    xcrun stapler staple "${OUTPUT_DIR}/${PKG_NAME}"
else
    echo "==> No Apple Notary credentials found. Skipping notarization."
fi

# Cleanup staging
rm -rf "${STAGE_DIR}"

echo "==> Packaging complete! Output: ${OUTPUT_DIR}/${PKG_NAME}"
