# skip interactive part of TZData
# Set timezone before installing packages
export DEBIAN_FRONTEND=noninteractive
echo "America/Los_Angeles" > /etc/timezone
dpkg-reconfigure -f noninteractive tzdata

cd /home/amd/1.110.0-a-102/ainic_bundle_1.110.0-a-104/host_sw_pkg/ionic_driver/src/ #please adjust the path to your ainic bundle folder

#You may need to install some packages as prerequisites through :
./setup_libs.sh --quiet  --assume=yes
#following line are extract of build-and-install.sh of this folder where the driver part have been removed : 
cd rdma-core
rm -rf build
mkdir -p build
cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX:PATH=/usr -DNO_PYVERBS=1 -DNO_MAN_PAGES=1 $EXTRA_CMAKE_FLAGS ..
ninja || exit 1
sudo ninja install || exit 1
cd ../..
# Clean out refs to drivers which have been removed
rm -f /etc/libibverbs.d/nes.driver
rm -f /etc/libibverbs.d/cxgb3.driver


