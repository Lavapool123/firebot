import importlib.util
import subprocess
import sys
import os
if importlib.util.find_spec('pygetwindow') is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pygetwindow"])
import pygetwindow
activeWindow=None
openedWindows={}
while True:
    banned=['Edge']
    if activeWindow!=pygetwindow.getActiveWindow():
        activeWindow=pygetwindow.getActiveWindow()
        if activeWindow!=None:
            openedWindows[pygetwindow.getActiveWindowTitle()]=activeWindow
            os.system('cls')
            print(f"All Opened Windows: {openedWindows} \nCurrently active window: {pygetwindow.getActiveWindowTitle()} ")
            for ban in banned:
                if ban in pygetwindow.getActiveWindowTitle() or pygetwindow.getActiveWindowTitle() in banned:
                    try:
                        activeWindow.close()
                        print(f"U naughty bad bad! Were you trying to open {pygetwindow.getActiveWindowTitle()}?")
                    except:
                        print(f"U naughty bad bad! Close {pygetwindow.getActiveWindowTitle()} right now!")