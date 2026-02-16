"""fpm_py

Python 版影像樣板比對（NCC / 金字塔 / 角度搜尋 / NMS）。
所有註解與說明文件使用繁體中文，以利移植紀錄與維護。
"""

__all__ = [
    "ncc_match",
    "ncc_match_fft",
    "pyramid_match",
    "learn_pattern",
]

__version__ = "0.1.0"

from .ncc import ncc_match, ncc_match_fft
from .pyramid import learn_pattern, pyramid_match
