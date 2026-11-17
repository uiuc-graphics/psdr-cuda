"""Execute tutorials with this Python environment and save their fresh outputs."""

import argparse
import base64
from pathlib import Path
import sys
import time

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    tutorial_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Notebook filenames (default: all)")
    parser.add_argument("--in-place", action="store_true", help="Update successful source notebooks")
    parser.add_argument("--output", type=Path, default=tutorial_dir.parent / "build" / "executed-tutorials")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per code cell in seconds")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    notebooks = [tutorial_dir / name for name in args.notebooks] if args.notebooks else sorted(tutorial_dir.glob("*.ipynb"))

    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        # Avoid accidentally using a global Python that lacks the local build.
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv[0] = sys.executable
        client = NotebookClient(
            notebook, km=manager, timeout=args.timeout, allow_errors=False,
            resources={"metadata": {"path": str(tutorial_dir)}},
        )
        start = time.monotonic()
        print(f"Running {path.name} with {sys.executable}", flush=True)
        try:
            client.execute()
            nbformat.validate(notebook)
        except Exception:
            nbformat.write(notebook, args.output / path.name)
            print(f"FAILED {path.name}; partial outputs saved in {args.output}", flush=True)
            raise
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            if client.kc is not None:
                client.kc.stop_channels()
        nbformat.write(notebook, args.output / path.name)
        if args.in_place:
            nbformat.write(notebook, path)
        for index, cell in enumerate(notebook.cells):
            for output_index, output in enumerate(cell.get("outputs", [])):
                png = output.get("data", {}).get("image/png")
                if png:
                    target = args.output / f"{path.stem}-{index}-{output_index}.png"
                    target.write_bytes(base64.b64decode(png))
        print(f"PASSED {path.name} ({time.monotonic() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
