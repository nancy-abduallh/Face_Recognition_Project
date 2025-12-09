import subprocess
import sys

def install_requirements():
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def main():
    try:
        import streamlit
        print("All packages are already installed!")
    except ImportError:
        print("Some packages missing. Installing...")
        install_requirements()
    
    subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()