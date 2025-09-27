import importlib.util
import subprocess
import sys
import os
for package in ['pyautogui','pyperclip','selenium']:
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
import pyautogui
import pyperclip
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
catNames=[]
catLinks=[]
run=False
while run==False:
    cats=int(input("How many tests? "))
    speed=str((cats*6)//60)+" minutes and "+str(cats*6-((cats*6)//60)*60)+ " seconds"
    run="y" in input(f"This will take a minimum of {speed}. Proceed? ").strip().lower()
pyautogui.hotkey("win","5")
options=Options()
options.add_argument("--headless=new")
driver=webdriver.Chrome(options=options)
while len(catNames)<cats:
    pyautogui.hotkey("ctrl","t")
    time.sleep(1)
    pyautogui.click(1880,160)
    time.sleep(1)
    pyautogui.hotkey("tab")
    pyautogui.hotkey("enter")
    time.sleep(3)
    pyautogui.hotkey("ctrl","l")
    pyautogui.hotkey("ctrl","c")
    link=pyperclip.paste()
    driver.get(link)
    try:
        catNameList=driver.find_element(By.CLASS_NAME,"pet-name").text.strip().split()
    except:
        catNameList=["Lil 404"]
        link="https://tabbycats.club/cat/lil404"
    catNameLists=[]
    for name in catNameList:
        catNameLists.append(name.capitalize())
    catName=" ".join(catNameLists)
    if not catName=="Lil 404":
        catNames.append(catName)
        catLinks.append(link)
    pyautogui.hotkey("ctrl","w")
    pyautogui.hotkey("ctrl","w")
driver.quit()
pyautogui.hotkey("win","3")
os.system('cls')
legends=["Fiery", "Evil", "Pinky", "Space", "Night", "Rainbow", "Tiny"]
legendcats=[]
legendlinks=[]
for x in range(cats):
    print(f"Cat {x+1}: \n └> Name: {catNames[x]}. \n └> Link: {catLinks[x]}.")
    if any(word in catNames[x] for word in legends):
        legendcats.append(catNames[x])
        legendlinks.append(catLinks[x])
for x in range(len(legendcats)):
    print(f"Legendary Cat {x+1}: \n └> Name: {legendcats[x]}. \n └> Link: {legendlinks[x]}.")