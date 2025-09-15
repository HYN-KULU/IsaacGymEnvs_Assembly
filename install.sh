cd ~/automate/IsaacGymEnvs_Assembly/ittapi
mkdir build && cd build
cmake .. -DITT_API_DYNAMIC=ON
make -j$(nproc)
sudo make install
