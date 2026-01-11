cd ~/automate/IsaacGymEnvs_Assembly/ittapi/build
rm -rf *
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
sudo ldconfig