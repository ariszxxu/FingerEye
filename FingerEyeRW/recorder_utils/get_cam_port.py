import pyudev
import subprocess
import sys

def list_usb_cameras():
    """List all USB cameras and their properties"""
    context = pyudev.Context()
    
    print("Connected USB Cameras:")
    print("=" * 60)
    
    cameras = []
    for device in context.list_devices(subsystem='video4linux'):
        if device.parent and 'usb' in device.parent.subsystem:
            cam_info = {
                'dev_path': device.device_path,
                'dev_node': device.device_node,
                'serial': device.parent.get('ID_SERIAL_SHORT', 'N/A'),
                'model': device.parent.get('ID_MODEL', 'N/A'),
                'vendor': device.parent.get('ID_VENDOR', 'N/A'),
                'usb_port': device.parent.get('DEVPATH', 'N/A').split('/')[-1]
            }
            cameras.append(cam_info)
    
    for i, cam in enumerate(cameras):
        print(f"  Device: {cam['dev_node']}")
        print(f"  USB Port: {cam['usb_port']}")
        print("-" * 60)
    
    return cameras

if __name__ == "__main__":
    # Check if pyudev is installed
    try:
        import pyudev
    except ImportError:
        print("Please install pyudev: pip install pyudev")
        sys.exit(1)
    
    list_usb_cameras()