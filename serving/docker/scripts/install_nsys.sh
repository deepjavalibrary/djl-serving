#!/usr/bin/env bash

# Define the base URL for Nsight Systems
BASE_URL="https://developer.download.nvidia.com/devtools/nsight-systems/"

# Check for LMI_DEBUG_NSIGHT_VERSION
if [ -n "${LMI_DEBUG_NSIGHT_VERSION}" ]; then
    # Check if the variable contains only numbers, dots, and hyphens
    echo "LMI_DEBUG_NSIGHT_VERSION is set: ${LMI_DEBUG_NSIGHT_VERSION}"
else
    # Find the latest version dynamically
    echo "Fetching the latest Nsight Systems version..."
    LMI_DEBUG_NSIGHT_VERSION=$(wget -qO- "$BASE_URL" | grep -oP 'NsightSystems-linux-public-\K[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+-[0-9]+' | sort -V | tail -1)

    if [ -z "$LMI_DEBUG_NSIGHT_VERSION" ]; then
        echo "Failed to fetch the latest version. Exiting."
        exit 1
    fi

    echo "Latest Nsight Systems version found: $LMI_DEBUG_NSIGHT_VERSION"
fi

# Security Validation
if [[ "${LMI_DEBUG_NSIGHT_VERSION}" =~ ^[0-9.-]+$ ]]; then
  echo "LMI_DEBUG_NSIGHT_VERSION is valid: ${LMI_DEBUG_NSIGHT_VERSION}"
else
  echo "LMI_DEBUG_NSIGHT_VERSION is invalid: ${LMI_DEBUG_NSIGHT_VERSION}"
  exit 1
fi

# Construct the download URL
DOWNLOAD_URL="${BASE_URL}NsightSystems-linux-public-${LMI_DEBUG_NSIGHT_VERSION}.run"

# Define the installation directory (default is /opt/nvidia/nsight-systems)
INSTALL_DIR="/opt/nvidia/nsight-systems"

# Update and install prerequisites
echo "Updating system and installing prerequisites..."
apt-get update
apt-get install -y wget build-essential aria2 expect

# Download Nsight Systems installer
echo "Downloading Nsight Systems ${LMI_DEBUG_NSIGHT_VERSION}..."
aria2c -x 16 "$DOWNLOAD_URL" -o nsight-systems-installer.run

# Verify the download
if [ ! -f "nsight-systems-installer.run" ]; then
    echo "Download failed. Exiting."
    exit 1
fi

# Make the installer executable
echo "Making the installer executable..."
chmod +x nsight-systems-installer.run

# Run the installer
echo "Running the Nsight Systems installer..."
# The installer is not respecting the CLI arguments
expect <<EOF
spawn ./nsight-systems-installer.run --quiet --accept --target ${INSTALL_DIR}
# Send ENTER and ACCEPT without waiting for specific prompts
send "\r"
sleep 1
send "ACCEPT\r"
sleep 1
send "${INSTALL_DIR}\r"
expect eof
EOF

# Add Nsight Systems to PATH
echo "Adding Nsight Systems to PATH..."
export PATH="${INSTALL_DIR}/pkg/bin:${PATH}"
echo "export PATH=${INSTALL_DIR}/pkg/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc

# Verify installation
echo "Verifying Nsight Systems installation..."
if command -v nsys &>/dev/null; then
    echo "Nsight Systems installed successfully!"
    nsys --version
else
    echo "Nsight Systems installation failed."
    exit 1
fi

# Clean up
echo "Cleaning up installer..."
rm -f nsight-systems-installer.run

echo "Installation complete. You can now use Nsight Systems with the 'nsys' command."
