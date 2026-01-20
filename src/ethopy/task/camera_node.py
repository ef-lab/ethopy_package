#!/usr/bin/env python3
"""
Camera node for Raspberry Pi
Connects to master and waits for recording commands
"""

from ethopy.utils.interface_proxy import RemoteInterfaceNode

# Configuration
MASTER_IP = "139.91.75.214"  # Your master computer IP
NODE_ID = "rpi_camera_1"
COMMAND_PORT = 5557
RESPONSE_PORT = 5558

print(f"Starting camera node: {NODE_ID}")
print(f"Master: {MASTER_IP}")
print(f"Ports: {COMMAND_PORT}/{RESPONSE_PORT}")
print("Waiting for commands from master...")

# Create and start remote node
node = RemoteInterfaceNode(
    master_host=MASTER_IP,
    node_id=NODE_ID,
    command_port=COMMAND_PORT,
    response_port=RESPONSE_PORT
)

# Run forever (listens for commands)
node.run()
