import os
import sys
import socket
import webbrowser
import uvicorn

def is_port_in_use(port: int, host: str = '127.0.0.1') -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    return start_port

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    port = find_available_port(8000)
    url = f"http://127.0.0.1:{port}"

    print("=" * 65)
    print("   TCIS - Telecom Customer Intelligence System")
    print("   Enterprise AI Analytics & Incident Intelligence Platform")
    print("=" * 65)
    print(f"🚀 Server is running successfully at:")
    print(f"👉 {url}")
    print(f"👉 Presentation Deck: {url}/presentation")
    print(f"📚 Interactive API Documentation (Swagger):")
    print(f"👉 {url}/docs")
    print("=" * 65)
    print("Press Ctrl+C to terminate the server.\n")

    # Open browser automatically after a short moment
    try:
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("main:app", host="127.0.0.1", port=port, log_level="info")

if __name__ == "__main__":
    main()
