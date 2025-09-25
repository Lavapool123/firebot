import importlib.util
import subprocess
import sys
if importlib.util.find_spec("pyautogui") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyautogui"])
import pyautogui
while True:
    pyautogui.hotkey("ctrl","v")
    pyautogui.hotkey("enter")
    import importlib.util