import importlib.util
import subprocess
import sys
for package in ['pyautogui','pygetwindow']:
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
import pyautogui
import pygetwindow
import time
import random
def chapterRun(chapter):
    if chapter==0:
        name=input("You turn back, into a cozy bedroom. A hearth crackles in the corner. A voice in your head speaks. You are not shocked by this, because you are used to it, though you've never heard it before. \nPray tell, adventurer, what is your name? ").strip().capitalize()
        dialogueChoice=input(f"Ah, {name}. Nice to make your acquaintance. My name is... Well, I doubt you need to know that. You may call me Fiaguhop for now. \nDialogue options: \nA: Is that a title? \nB: C'mon, you can tell me your name. It can't be that bad, right? \nC: Oookay, Fiaguhop. Where am I? \n{name}: ").strip().capitalize()
        if dialogueChoice=="A":
            print("Of sorts. I guess you could call it more of a... nickname")
        elif dialogueChoice=="B":
            print("Nosy, I see. And here I was so looking forward to you. I thought you'd be different. Ah well, no matter. ")
            time.sleep(5)
            lose()
        elif dialogueChoice=="C":
            print("Why, you're in your bedroom. Don't you recognize it? ")
            time.sleep(1)
            print("You do. Strangely enough, though you've never been here, it feels familiar.")
        else:
            print("Now, now, I told you. No typos. No misspellings.")
            time.sleep(1)
            lose()
        print("Now, enough questions. It's time for business. Go over to the hearth. ")
        time.sleep(1.5)
        print("You walk over to the hearth, and notice that the flames are swirling. They spin faster and faster, and reach out to you. You reach your hand out to it, and it sucks you in. ")
    else:
        chapterTitles=['']
        print(f"Chapter {chapter}: {chapterTitles[chapter-1]}")
        if chapter==1:
            if chapter == 1:
                print("The flames spit you out ")
    if chapter==0:
        chapname="the prologue"
    else:
        chapname="chapter",chapter+1
    if "n" in input(f"Congratulations. You've {random.choice(['completed','finished'])} {chapname}. Would you like to continue? \n").strip().lower():
        lose()
    else:
        chapterRun(chapter+1)
def lose():
    print("Bye!")
    time.sleep(1)
    pyautogui.FAILSAFE=False
    pythonRunner=pygetwindow.getActiveWindow()
    restorationAttempted=False
    while True:
        if restorationAttempted==True:
            print("Gotcha.")
            time.sleep(1)
            #pyautogui.hotkey('alt', 'tab')
            #pyautogui.hotkey('alt', 'f4')
        elif restorationAttempted==False:
            if not pythonRunner.isMinimized:
                pythonRunner.minimize()
                #restorationAttempted=True
door=input("\n\n\nWelcome to my game! Please use proper spelling or you will die. (Or I guess your computer will die but same difference) \n\nYou enter a room. It is empty of all furniture, and the walls are stark white. There are two doors, marked: \n Life. \n Death. \n A sign reads: \nChoose neither. \nTurn back. \n\nNow. \nWhat would you like to do? \n").strip().lower()
if door!="life": 
    if "turn back" in door or "neither" in door:
        chapterRun(0)
    else:
        print("So you have chosen death.")
        time.sleep(1)
        lose()
else:
    print("You walk in the door. A beautiful garden is outside, and there are butterflies fluttering around. The sun is shining, the birds are chirping, and everything is great.")
    time.sleep(3)
    print("Everything is peaceful. You're safe.")
    time.sleep(1)
    print("Or so you think.")
    time.sleep(1)
    lose()