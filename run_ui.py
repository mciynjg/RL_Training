"""
Quick launcher for RL Training Framework Web UI.

Usage: python run_ui.py

Or simply use: streamlit run app.py
"""

import subprocess
import sys
import socket

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def check_port_available(port):
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except OSError:
            return False

def main():
    port = 8501
    allow_external = True  # Set to False to only allow local access

    # Check if streamlit is installed
    try:
        import streamlit
        print(f"Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("Error: streamlit is not installed.")
        print("Install with: pip install streamlit")
        sys.exit(1)

    # Get local IP
    local_ip = get_local_ip()

    # Check if port is available by trying to connect
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()

    if result == 0:
        print(f"Warning: Port {port} is already in use.")
        print()
        print("Streamlit may already be running.")
        print(f"  - Local: http://localhost:{port}")
        print(f"  - Network: http://{local_ip}:{port}")
        print()
        sys.exit(0)

    print()
    print("=" * 60)
    print("  RL Training Framework - Web UI")
    print("=" * 60)
    print()
    print(f"Starting Streamlit on port {port}...")
    print()
    print("Access URLs:")
    print(f"  - Local:   http://localhost:{port}")
    print(f"  - Network: http://{local_ip}:{port}")
    print()
    if allow_external:
        print("Note: External devices on the same network can access this server.")
        print("      Make sure your firewall allows incoming connections on port 8501.")
    else:
        print("Note: Only local access is allowed.")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Use streamlit run command with external access enabled
    args = [
        sys.executable, "-m", "streamlit", "run",
        "app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",  # Allow external connections
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            print(f"\nError: Could not bind to port {port}.")
            print("Another process may be using this port.")
        else:
            raise
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")

if __name__ == "__main__":
    main()
