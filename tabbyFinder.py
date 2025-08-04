import threading, random, time, queue, os, subprocess, sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from concurrent.futures import ThreadPoolExecutor
import webbrowser

# Ensure selenium is installed
try:
    import selenium
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "selenium"])

# Constants
baseUrl = "https://tabbycats.club/cat/"
speciesList = ["cat", "bunny"]
successLock = threading.Lock()
resultsQueue = queue.Queue()
successes = []
noteworthyLinks = []
foundCats = 0
foundBunnies = 0
TARGET_CATS = 6
TARGET_BUNNIES = 3
MAX_THREADS = 3

# Set up headless browser
def createDriver():
    chromeOptions = Options()
    chromeOptions.add_argument("--headless=new")
    chromeOptions.add_argument("--log-level=3")
    chromeOptions.add_argument("--disable-logging")
    chromeOptions.add_experimental_option("excludeSwitches", ["enable-logging"])
    
    # Suppress DevTools + Chrome warnings by silencing the Service logs
    service = Service(log_path=os.devnull)
    
    return webdriver.Chrome(service=service, options=chromeOptions)

# Check a single link
def checkLink(url):
    global foundCats, foundBunnies
    driver = None
    try:
        driver = createDriver()
        driver.get(url)
        nameEl = driver.find_element(By.TAG_NAME, "h1")
        speciesEl = driver.find_element(By.TAG_NAME, "h2")
        name = nameEl.text.strip()
        species = speciesEl.text.strip().lower()

        with successLock:
            if species == "cat" and foundCats < TARGET_CATS:
                foundCats += 1
                successes.append((name, "Cat", url))
                webbrowser.open(url)
                resultsQueue.put(f"Success: {name}, Cat\n")
            elif species == "bunny" and foundBunnies < TARGET_BUNNIES:
                foundBunnies += 1
                successes.append((name, "Bunny", url))
                webbrowser.open(url)
                resultsQueue.put(f"Success: {name}, Bunny\n")
            elif species not in speciesList:
                noteworthyLinks.append(url)
                resultsQueue.put(f"Noteworthy: {url}\n")
    except Exception:
        pass
    finally:
        if driver:
            driver.quit()

# Worker thread
def workerLoop():
    while True:
        with successLock:
            if foundCats >= TARGET_CATS and foundBunnies >= TARGET_BUNNIES:
                break
        randId = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
        url = baseUrl + randId
        resultsQueue.put(f"Checking link: {url}\n")
        checkLink(url)
        time.sleep(0.1)

# Real-time result printer
def printerLoop():
    while True:
        try:
            msg = resultsQueue.get(timeout=1)
            print(msg, end="")
            resultsQueue.task_done()
        except queue.Empty:
            with successLock:
                if foundCats >= TARGET_CATS and foundBunnies >= TARGET_BUNNIES:
                    break

# Main entry point
def main():
    printer = threading.Thread(target=printerLoop)
    printer.start()

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for _ in range(MAX_THREADS):
            executor.submit(workerLoop)

    printer.join()

    print("\nSuccessful Finds:")
    for name, species, link in successes:
        print(f"{name}, {species}, {link}")

    if noteworthyLinks:
        print("\nNoteworthy:")
        for link in noteworthyLinks:
            print(link)

if __name__ == "__main__":
    main()
