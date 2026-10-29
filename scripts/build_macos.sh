#!/usr/bin/env bash
# Build the Hey Spark macOS app and disk image (Apple Silicon, macOS 14+).
#
#   bash scripts/build_macos.sh
#
# 1. syncs the env with the desktop + openWakeWord extras and the build group
# 2. freezes the app with PyInstaller   -> build/dist/HeySpark.app
# 3. signs it                           -> Developer ID if MACOS_SIGN_IDENTITY is set, else ad hoc
# 4. smoke-tests the frozen binary      -> HeySpark --self-test loads every native library
# 5. packs a drag-to-Applications DMG   -> build/installer/HeySpark-<version>-macos-arm64.dmg
# 6. notarizes and staples the DMG      -> only when signing and APPLE_ID, APPLE_TEAM_ID and
#                                          APPLE_APP_PASSWORD (an app-specific password) are set
#
# An ad hoc signature is enough to run on the Mac that built it. A downloaded ad hoc build is
# blocked by Gatekeeper until the user allows it in System Settings > Privacy & Security; see
# README "Install on macOS". Set SPARK_CONSOLE=1 to get a console build for debugging.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: build_macos.sh runs on macOS only (use scripts/build_windows.ps1 on Windows)" >&2
  exit 1
fi

echo "==> uv sync (desktop, oww, kokoro, build)"
uv sync --extra desktop --extra oww --extra kokoro --group build --locked

version=$(uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
arch=$(uname -m)
[[ "$arch" == "x86_64" ]] && arch_label="intel" || arch_label="$arch"
echo "==> version $version ($arch)"

echo "==> PyInstaller"
rm -rf build/dist/HeySpark build/dist/HeySpark.app
uv run --group build pyinstaller packaging/hey_spark.spec --noconfirm --distpath build/dist --workpath build/work
app="build/dist/HeySpark.app"
[[ -d "$app" ]] || { echo "error: $app was not produced" >&2; exit 1; }

identity="${MACOS_SIGN_IDENTITY:-}"
if [[ -n "$identity" ]]; then
  echo "==> codesign (Developer ID, hardened runtime)"
  codesign --force --deep --timestamp --options runtime \
    --entitlements packaging/macos/entitlements.plist --sign "$identity" "$app"
else
  echo "==> codesign (ad hoc)"
  codesign --force --deep --sign - "$app"
fi
codesign --verify --deep --strict --verbose=2 "$app"

echo "==> smoke test"
"$app/Contents/MacOS/HeySpark" --self-test

echo "==> DMG"
dmg="build/installer/HeySpark-${version}-macos-${arch_label}.dmg"
staging="build/dmg"
rm -rf "$staging" "$dmg"
mkdir -p "$staging" build/installer
cp -R "$app" "$staging/"
ln -s /Applications "$staging/Applications"
hdiutil create -volname "Hey Spark" -srcfolder "$staging" -fs HFS+ -format UDZO -ov "$dmg"
rm -rf "$staging"

if [[ -n "$identity" ]]; then
  codesign --force --timestamp --sign "$identity" "$dmg"
  if [[ -n "${APPLE_ID:-}" && -n "${APPLE_TEAM_ID:-}" && -n "${APPLE_APP_PASSWORD:-}" ]]; then
    echo "==> notarize"
    xcrun notarytool submit "$dmg" --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" \
      --password "$APPLE_APP_PASSWORD" --wait
    xcrun stapler staple "$dmg"
  else
    echo "warning: signed but not notarized (APPLE_ID / APPLE_TEAM_ID / APPLE_APP_PASSWORD unset)" >&2
  fi
fi

ls -lh "$dmg"
