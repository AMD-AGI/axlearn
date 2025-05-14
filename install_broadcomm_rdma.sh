#!/bin/bash

# We want to install the broadcom drivers

export DEBIAN_FRONTEND=noninteractive

echo -e "\n\n============Installing required pkgs============\n\n"
apt update
apt install -y curl unzip
apt install -y linux-headers-"$(uname -r)" libelf-dev
apt install -y gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool libibverbs-dev rdma-core strace libibmad5 libibnetdisc5 ibverbs-providers libibumad-dev libibumad3 libibverbs1 libnl-3-dev libnl-route-3-dev


echo -e "\n\n============Compiling RoCE Lib now============\n\n"
mkdir -p /tmp/bc_install
cd /tmp/bc_install

# Highlighted item will change depending on the release
rm -rf bcm5760x_230.2.52.0a
curl -o bcm5760x_230.2.52.0a.zip https://docs.broadcom.com/docs-and-downloads/ethernet-network-adapters/NXE/Thor2/GCA1/bcm5760x_230.2.52.0a.zip
unzip bcm5760x_230.2.52.0a.zip
cd bcm5760x_230.2.52.0a/drivers_linux/bnxt_rocelib/
tar -xf libbnxt_re-230.2.52.0.tar.gz
cd libbnxt_re-230.2.52.0
sh autogen.sh
./configure
make
find /usr/lib64/ /usr/lib -name "libbnxt_re-rdmav*.so" -exec mv {} {}.inbox \;
make install all
sh -c "echo /usr/local/lib >> /etc/ld.so.conf"
ldconfig
cp -f bnxt_re.driver /etc/libibverbs.d/
find . -name "*.so" -exec md5sum {} \;
BUILT_MD5SUM=$(find . -name "libbnxt_re-rdmav*.so" -exec md5sum {} \; | cut -d " " -f 1)
echo -e "\n\nmd5sum of the built libbnxt_re is $BUILT_MD5SUM"
echo -e "\n\n===================RoCE userlib compile complete===================\n\n"
#!/bin/bash
