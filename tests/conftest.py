# On macOS, pip's LightGBM and PyTorch each ship their own OpenMP runtime.
# Loading torch first and then training LightGBM in the same process can
# segfault. Importing LightGBM first (and keeping OpenMP to one copy) avoids it.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import lightgbm  # noqa: F401,E402  (must load before torch)
