#!/bin/sh
set -e

REPO="Malav-P/mlobs"
BIN="mlc"
INSTALL_DIR="${MLC_INSTALL_DIR:-/usr/local/bin}"

# Detect OS
case "$(uname -s)" in
  Linux)  OS="linux" ;;
  Darwin) OS="macos" ;;
  *)
    echo "Unsupported OS: $(uname -s)" >&2
    exit 1
    ;;
esac

# Detect architecture
case "$(uname -m)" in
  x86_64 | amd64) ARCH="x86_64" ;;
  arm64 | aarch64) ARCH="arm64" ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

ARTIFACT="${BIN}-${OS}-${ARCH}"
URL="https://github.com/${REPO}/releases/latest/download/${ARTIFACT}"

echo "Downloading mlc (${OS}/${ARCH})..."

# Download to a temp file first, then install (may need sudo for the destination)
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$URL" -o "$TMP"
elif command -v wget >/dev/null 2>&1; then
  wget -qO "$TMP" "$URL"
else
  echo "curl or wget is required" >&2
  exit 1
fi

chmod +x "$TMP"

# Install — use sudo if the directory isn't writable by the current user
mkdir -p "$INSTALL_DIR" 2>/dev/null || sudo mkdir -p "$INSTALL_DIR"
if [ -w "$INSTALL_DIR" ]; then
  mv "$TMP" "$INSTALL_DIR/$BIN"
else
  sudo mv "$TMP" "$INSTALL_DIR/$BIN"
fi

echo "Installed to $INSTALL_DIR/$BIN"

# Warn if the install dir isn't on PATH
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    echo ""
    echo "Note: $INSTALL_DIR is not on your PATH."
    echo "Add this to your shell profile:"
    echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
    ;;
esac
