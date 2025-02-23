import sys
from src.backend.app.main import app

if __name__ == "__main__":
    sys.path.insert(0, "src")  # Ensures Python can find the backend folder
    app.run(debug=True)
