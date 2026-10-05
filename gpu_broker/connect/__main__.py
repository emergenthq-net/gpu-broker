"""`python -m gpu_broker.connect ...`, and the zipapp entry point that connect.sh runs."""
import sys

from gpu_broker.connect.cli import main

sys.exit(main(sys.argv[1:]))
