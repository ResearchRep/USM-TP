#   Copyright [2026] [Indian Institute of Science]
#   SPDX-License-Identifier: Apache-2.0

import ctypes
import atexit
import os

this_dir = __file__.rsplit ("/", maxsplit = 1)[0]
libze_util_file = f"{this_dir}/_force_usm.so"
if not os.path.exists (libze_util_file):
    e = Exception ("Please build _force_usm.so")
    raise e

print ("force_usm imported")
libze_util = ctypes.CDLL (libze_util_file)
libze_util.setup ()
atexit.register (libze_util.cleanup)
