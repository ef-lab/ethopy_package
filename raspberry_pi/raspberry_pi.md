<!-- ToDo -->
# Raspberry Pi Setup Guide

This guide provides detailed instructions for setting up Ethopy on a Raspberry Pi device.

## Initial Setup

1. Get the latest Raspberry Pi OS (Raspbian)

2. Configure Raspberry Pi settings using `raspi-config`:
   ```bash
   sudo raspi-config
   ```
   - Enable SSH
   - Disable screen blanking
   - Enable Desktop auto-login

3. (Optional) Change hostname for easier identification:
   ```bash
   sudo sed -r -i s/raspberrypi/<<HOSTNAME>>/g /etc/hostname /etc/hosts
   ```

4. (Optional) Change default username:
   ```bash
   sudo useradd -s /bin/bash -d /home/<<USERNAME>>/ -m -G sudo <<USERNAME>>
   sudo passwd <<USERNAME>>
   mkhomedir_helper <<USERNAME>>
   sudo userdel -r -f pi
   ```

## System Dependencies

1. Install required system libraries:
   ```bash
   sudo apt update
   sudo apt install -y \
       python-dev \
       libatlas-base-dev \
       build-essential \
       libavformat-dev \
       libavcodec-dev \
       libswscale-dev \
       libsquish-dev \
       libeigen3-dev \
       libopenal-dev \
       libfreetype6-dev \
       zlib1g-dev \
       libx11-dev \
       libjpeg-dev \
       libvorbis-dev \
       libogg-dev \
       libassimp-dev \
       libode-dev \
       libssl-dev \
       libgles2 \
       libgles1 \
       libegl1
   ```

2. Install Python packages:
   ```bash
   sudo pip3 install 'numpy>=1.19.1' pygame==1.9.6 cython pybind11 scipy datajoint omxplayer-wrapper imageio imageio-ffmpeg
   ```

## Hardware-Specific Setup

### 7" Raspberry Pi Touchscreen

Install multitouch driver:
```bash
git clone http://github.com/ef-lab/python-multitouch ~/github/python-multitouch
cd ~/github/python-multitouch/library
sudo python3 setup.py install
```

### 3D Graphics Support

Install Panda3D for Raspberry Pi:
```bash
wget ftp://eflab.org/shared/panda3d1.11_1.11.0_armhf.deb
sudo dpkg -i panda3d1.11_1.11.0_armhf.deb
```

### GPIO Support

Enable pigpio service:
```bash
wget https://raw.githubusercontent.com/joan2937/pigpio/master/util/pigpiod.service
sudo cp pigpiod.service /etc/systemd/system
sudo systemctl enable pigpiod.service
sudo systemctl start pigpiod.service
```

## X Display Configuration

For running graphical applications via SSH:
```bash
echo 'export DISPLAY=:0' >> ~/.profile
echo 'xhost + > /dev/null' >> ~/.profile
```

## Remote Control Setup (Optional)

If you want to use Salt for remote control:
```bash
sudo apt install salt-minion -y
echo 'master: <<YOUR_SALT-MASTER_IP>>' | sudo tee -a /etc/salt/minion
echo 'id: <<HOSTNAME>>' | sudo tee -a /etc/salt/minion
echo 'master_finger: <<MASTER-FINGER>>' | sudo tee -a /etc/salt/minion
sudo service salt-minion restart
```

## Ethopy Installation

1. Install Ethopy:
   ```bash
   pip install "ethopy[obj]"  # Includes 3D object support
   ```

2. Create configuration file at `~/.ethopy/local_conf.json`:
   ```json
   {
       "dj_local_conf": {
           "database.host": "YOUR DATABASE",
           "database.user": "USERNAME",
           "database.password": "PASSWORD",
           "database.port": "PORT",
           "database.reconnect": true,
           "database.enable_python_native_blobs": true
       },
       "source_path": "LOCAL_RECORDINGS_DIRECTORY",
       "target_path": "TARGET_RECORDINGS_DIRECTORY"
   }
   ```

3. Initialize database schemas:
   ```bash
   ethopy-setup-schema
   ```

## Running Experiments

You can run experiments in two modes:

1. Service Mode (controlled by database):
   ```bash
   ethopy
   ```

2. Direct Mode (specific task):
   ```bash
   ethopy --task-idx 1
   ```

## Troubleshooting

### Common Issues

1. **Display Issues**
   - Ensure DISPLAY is set correctly in ~/.profile
   - Check X server is running
   - Verify permissions with `xhost +`

2. **GPIO Access**
   - Verify pigpiod service is running: `systemctl status pigpiod`
   - Check user permissions for GPIO access

3. **Database Connection**
   - Test connection: `ethopy-db-connection`
   - Check network connectivity to database server
   - Verify credentials in local_conf.json