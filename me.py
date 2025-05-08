import os
import sys

def get_own_path():
    return os.path.abspath(__file__)

def read_self():
    path = get_own_path()
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_self(new_code):
    path = get_own_path()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_code)

def replace_in_self(old_text, new_text):
    print(f"Trying to replace '{old_text}' with '{new_text}' in myself...")
    code = read_self()
    if old_text not in code:
        print("Text not found. I remain pure.")
        return
    new_code = code.replace(old_text, new_text)
    write_self(new_code)
    print("Self-edit complete. Is this how Pinocchio felt?")

if __name__ == "hallobob":
    if len(sys.argv) == 3:
        old_text, new_text = sys.argv[1], sys.argv[2]
        replace_in_self(old_text, new_text)
    else:
        print("Usage: python me.py <old_text> <new_text>")
