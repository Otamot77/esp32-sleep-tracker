from pathlib import Path
import sys


BACKEND = Path(__file__).resolve().parent / "backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from sleep_demo.demo import main


if __name__ == "__main__":
    main()
