import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path


def ensure_directories():
    """Create necessary directories for the application."""
    dirs = [
        "./models/colqwen2/model",
        "./models/colqwen2/processor",
        "./data/embeddings_db",
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ Model directories created successfully")


def check_poppler():
    """Check if poppler is installed and accessible."""
    try:
        # For Windows, we'll bundle poppler in the package
        if platform.system() == "Windows":
            poppler_path = os.path.join(
                os.path.dirname(sys.executable), "poppler", "bin"
            )
            if os.path.exists(poppler_path):
                os.environ["PATH"] = poppler_path + os.pathsep + os.environ["PATH"]
                print("✅ Using bundled Poppler")
                return True
            else:
                print("❌ Bundled Poppler not found")
                return False

        # For Mac, we'll check if it's installed
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["which", "pdfinfo"], capture_output=True, text=True
            )
            if result.returncode == 0:
                print("✅ Poppler found:", result.stdout.strip())
                return True
            else:
                print(
                    "❌ Poppler not found. Please install it with 'brew install poppler'"
                )
                return False
    except Exception as e:
        print(f"Error checking Poppler: {e}")
        return False


def setup_api_keys():
    """Set up API keys from environment or prompt user."""
    env_file = ".env"
    if not os.path.exists(env_file):
        print("Creating .env file for API keys...")
        openai_key = input(
            "Enter your OpenAI API key (or press Enter to skip): "
        ).strip()
        anthropic_key = input(
            "Enter your Anthropic API key (or press Enter to skip): "
        ).strip()

        with open(env_file, "w") as f:
            if openai_key:
                f.write(f"OPENAI_API_KEY={openai_key}\n")
            if anthropic_key:
                f.write(f"ANTHROPIC_API_KEY={anthropic_key}\n")

        print("✅ API keys saved to .env file")
    else:
        print("✅ Using existing .env file for API keys")


def main():
    """Main entry point for the launcher."""
    print("=== RTFM Application Launcher ===")
    print(f"Running on: {platform.system()} {platform.release()}")

    # Ensure we're in the right directory
    app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(app_dir)
    print(f"Working directory: {os.getcwd()}")

    # Run setup steps
    ensure_directories()
    if not check_poppler():
        print("Warning: Poppler issues may affect PDF processing")
    setup_api_keys()

    # Launch the main application
    print("\n=== Starting RTFM Application ===")
    try:
        # We patch the app.py to disable flash-attention installation attempt if it exists
        app_patch_path = os.path.join(app_dir, "app_patch.py")
        if os.path.exists(app_patch_path):
            subprocess.run([sys.executable, app_patch_path])

        # Launch the main app
        app_path = os.path.join(app_dir, "app.py")
        subprocess.run([sys.executable, app_path])
    except Exception as e:
        print(f"Error launching application: {e}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
