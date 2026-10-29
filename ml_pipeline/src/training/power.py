"""
Power state, for long unattended runs on a laptop.

train.py keeps Windows awake while it trains. run_queue.py also waits for the
charger before each run: on battery this laptop's GPU runs at a third of its
speed, and a training run drained a full battery to 29% in 41 minutes.
"""

import sys


def keep_awake():
    """Ask Windows not to sleep while this process runs, as a video player does
    during playback. No setting changes, and the request ends with the process.
    It cannot stop sleep when the lid closes, or an update restart.

    Returns None off Windows, else whether Windows accepted the request.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_state.argtypes, set_state.restype = [ctypes.c_uint], ctypes.c_uint
    return set_state(es_continuous | es_system_required) != 0


def on_mains_power():
    """True on the charger, False on battery, None when it cannot be told (off
    Windows, or Windows does not know). A machine without a battery reports
    mains power."""
    if sys.platform != "win32":
        return None
    import ctypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    status = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    return {0: False, 1: True}.get(status.ACLineStatus)    # 255 means unknown
