import os
import sys

# Ensure our local C++ extensions are loadable
cpp_ext_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "cpp_extensions")
)
if cpp_ext_path not in sys.path:
    sys.path.insert(0, cpp_ext_path)

mingw_bin = os.path.expanduser(r"~\scoop\apps\mingw\current\bin")
if os.path.exists(mingw_bin):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(mingw_bin)
    else:
        if mingw_bin not in os.environ["PATH"]:
            os.environ["PATH"] = mingw_bin + os.pathsep + os.environ["PATH"]
