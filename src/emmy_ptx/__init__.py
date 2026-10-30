"""emmy-ptx: compile Emmy Kernel IR to NVIDIA PTX, without nvcc."""

from .compiler import Compiler, compile_file

__version__ = "0.1.0"
__all__ = ["Compiler", "compile_file", "__version__"]
