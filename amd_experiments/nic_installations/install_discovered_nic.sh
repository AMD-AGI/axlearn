#!/bin/bash
cd "$(dirname "$0")"

if lspci | grep -qi 'Mellanox'; then
  echo "Mellanox InfiniBand devices found, setting DISCOVERED_NIC to mlx"
  export DISCOVERED_NIC='mlx'
  echo "export DISCOVERED_NIC=mlx" >> /etc/bash.bashrc

elif lspci | grep -qi 'ionic' || lspci | grep -qi 'pollara' ; then
  echo "Ionic InfiniBand devices found, setting DISCOVERED_NIC to ionic"
  export DISCOVERED_NIC='ionic'
  echo "export DISCOVERED_NIC=ionic" >> /etc/bash.bashrc

elif lspci | grep -qi 'Broadcom'; then
  echo "Broadcom InfiniBand devices found, setting DISCOVERED_NIC to bcm"
  export DISCOVERED_NIC='bcm'
  echo "export DISCOVERED_NIC=bcm" >> /etc/bash.bashrc

else
  echo "No Mellanox, Ionic or Broadcom InfiniBand devices found, skipping NIC setup."
fi


if [ "${DISCOVERED_NIC:-}" = 'mlx' ]; then
  echo "Installing Mellanox drivers"
  bash install_mellanox_ib.sh
elif [ "${DISCOVERED_NIC:-}" = 'ionic' ]; then
  echo "Installing Ionic drivers"
  bash install_pollara_ib_noninteractive.sh
elif [ "${DISCOVERED_NIC:-}" = 'bcm' ]; then
  echo "Installing Broadcom drivers"
  bash install_broadcom_ib.sh
else
  echo "No specific NIC drivers to install."
fi

