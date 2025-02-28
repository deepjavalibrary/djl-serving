#!/bin/bash
set -e

apt-get update
apt-get install -y \
    build-essential \
    gfortran \
    autoconf \
    automake \
    libtool \
    flex \
    hwloc \
    libhwloc-dev \
    libevent-dev \
    libfabric-dev \
    libevent-2.1-7 \
    devscripts \
    debhelper
apt remove -y openmpi-bin libopenmpi-dev libpmix-dev libpmix2
apt autoremove -y

# Build and install libpmix2 as deb
cd /tmp
git clone --recursive -b v4.2.6 https://github.com/openpmix/openpmix.git
cd openpmix
./autogen.pl
./configure --prefix=/usr --with-devel-headers

mkdir -p debian/libpmix2/usr
mkdir -p debian/libpmix2/DEBIAN

make -j$(nproc)
make DESTDIR=$(pwd)/debian/libpmix2 install

cat > debian/libpmix2/DEBIAN/control << EOL
Package: libpmix2
Version: 4.2.6
Architecture: amd64
Section: libs
Depends: libc6, libevent-2.1-7, libhwloc15
Maintainer: Ubuntu Developers <ubuntu-devel-discuss@lists.ubuntu.com>
Description: Process Management Interface (Exascale) library
 This is the OpenMPI implementation of the Process Management Interface (PMI)
 Exascale API. PMIx aims to retain transparent compatibility with the existing
 PMI-1 and PMI-2 definitions, and any future PMI releases; Support
 the Instant On initiative for rapid startup of applications at exascale
 and beyond.
EOL

dpkg-deb --build debian/libpmix2
dpkg -i debian/libpmix2.deb

make install
ldconfig

# Reinstall OpenMPI
cd /tmp
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.2.tar.gz
tar -xvf openmpi-4.1.2.tar.gz
cd openmpi-4.1.2
./configure --prefix=/usr --with-pmix=/usr
make -j$(nproc)
make install
ldconfig

# Cleanup
apt remove -y \
    build-essential \
    gfortran \
    autoconf \
    automake \
    libtool \
    flex \
    libhwloc-dev \
    libevent-dev \
    libfabric-dev \
    devscripts \
    debhelper

cd /tmp
rm -rf openpmix openmpi-4.1.2 openmpi-4.1.2.tar.gz
