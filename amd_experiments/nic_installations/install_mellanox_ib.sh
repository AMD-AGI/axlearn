#!/bin/bash

MOFED_VERSION=5.9-0.5.6.0.127
OS_VERSION=ubuntu22.04
PLATFORM=x86_64
WORKDIR=/tmp/install_mellanox_ib

export DEBIAN_FRONTEND=noninteractive
export TZ=US
ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
apt -y update
apt install -y linux-headers-"$(uname -r)" libelf-dev
apt install -y wget unzip gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool libibverbs-dev rdma-core strace libibmad5 libibnetdisc5 ibverbs-providers libibumad-dev libibumad3 libibverbs1 libnl-3-dev libnl-route-3-dev
add-apt-repository universe
apt install -y --no-install-recommends pciutils numactl libnuma-dev libgtk2.0-dev libatk1.0-dev libcairo2 gfortran tcsh lsof libnl-3-dev libmnl-dev ethtool tcl tk perl make libusb-1.0-0 libusb-1.0-0-dev libfuse2 bison graphviz swig debhelper dpatch chrpath flex kmod

mkdir -p $WORKDIR && cd $WORKDIR
wget -q http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz
tar -xvf MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz
cd MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}
perl mlnxofedinstall --user-space-only --without-fw-update -q --distro ubuntu22.04

# echo InfiniBand Info
echo "ibstat"
ibstat
echo "ibv_devices"
ibv_devices
echo "ls -l /sys/class/infiniband/"
ls -l /sys/class/infiniband/        
echo "ip addr"
ip addr

# clean up
rm -rf $WORKDIR
apt clean