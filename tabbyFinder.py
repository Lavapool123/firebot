import subprocess
import sys
import threading
import time
import random
import string
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
import webbrowser
import os

# auto-install selenium if missing
try:
    import selenium
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "selenium"])
    import selenium

# constants
baseUrl="https://tabbycats.club/cat/"
targetSpecies={"cat", "bunny"}
foundCats=[]
failLinks=[]
noteworthyCount=0
foundCatCount=0
foundBunnyCount=0
lock=threading.Lock()

# setup chrome options
chromeOptions=Options()
chromeOptions.add_argument("--headless=new")
chromeOptions.add_argument("--log-level=3")
chromeOptions.add_experimental_option("excludeSwitches", ["enable-logging"])

def randomCode():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

def checkTabbyLink():
    global foundCatCount, foundBunnyCount, noteworthyCount
    while True:
        with lock:
            if foundCatCount>=6 and foundBunnyCount>=3:
                return
        code=randomCode()
        link=baseUrl+code
        print(f"checking link: {link}")
        try:
            driver=webdriver.Chrome(options=chromeOptions)
            driver.get(link)
            time.sleep(1)

            if "Lil 404" in driver.page_source:
                with lock:
                    failLinks.append(link)
                driver.quit()
                continue

            try:
                name=driver.find_element(By.CLASS_NAME, "css-1rs21hf").text
                species=driver.find_element(By.CLASS_NAME, "css-1id5uvs").text.lower()
            except:
                driver.quit()
                with lock:
                    failLinks.append(link)
                continue

            if species in targetSpecies:
                with lock:
                    if species=="cat":
                        foundCatCount+=1
                    elif species=="bunny":
                        foundBunnyCount+=1
                    foundCats.append((name, species, link))
                print(f"cat found! name: {name}, species: {species}")
                webbrowser.open(link)
            else:
                with lock:
                    noteworthyCount+=1
                print(f"non-target animal found at {link}")
            driver.quit()
        except WebDriverException:
            print(f"error accessing {link}")

threads=[]
for _ in range(3):
    thread=threading.Thread(target=checkTabbyLink)
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

# print results
print("\n--- Search Complete ---")
print(f"Found {len(foundCats)} target animals ({foundCatCount} cats, {foundBunnyCount} bunnies)")
print(f"Found {noteworthyCount} other animals\n")

print("--- Successes ---")
for name,species,link in foundCats:
    print(f"{name} ({species}) - {link}")

print("\n--- Failures ---")
for link in failLinks:
    print(link)
