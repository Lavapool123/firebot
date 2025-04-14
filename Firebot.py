import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
import tempfile

# Try to import NumPy, auto-install if not found
try:
    import numpy as np
except ImportError:
    import subprocess, sys
    print("[Info] NumPy not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np

# Firebot AI Neural Network Language Model
class SimpleNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleNNLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)         # Shape: (batch_size, seq_len, embedding_dim)
        x = x.mean(dim=1)              # Reduce sequence dimension
        x = F.relu(self.fc1(x))        # First fully connected layer
        x = self.fc2(x)                # Output layer
        return F.log_softmax(x, dim=1) # Softmax for classification

# FileManager allows safe file operations within the AI's designated folder
class FileManager:
    def __init__(self, folder="model_files"):
        self.folder = os.path.abspath(folder)
        self.own_file = os.path.abspath(__file__) if "__file__" in globals() else None
        os.makedirs(self.folder, exist_ok=True)

    def _is_within_folder(self, path):
        return os.path.abspath(path).startswith(self.folder)

    def create_file(self, filename, content=""):
        try:
            path = os.path.abspath(os.path.join(self.folder, filename))
            if self.own_file and path == self.own_file:
                return "Permission denied: Cannot create own file."
            if not self._is_within_folder(path):
                return "Permission denied: File must be within designated folder."
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
        except Exception as e:
            return f"Error creating file: {e}"

    def read_file(self, filename):
        try:
            path = os.path.abspath(os.path.join(self.folder, filename))
            if self._is_within_folder(path) and os.path.exists(path):
                with open(path, "r") as f:
                    return f.read()
            elif self.own_file and os.path.abspath(filename) == self.own_file:
                with open(self.own_file, "r") as f:
                    return f.read()
            return None
        except Exception as e:
            return f"Error reading file: {e}"

    def edit_file(self, filename, new_content):
        try:
            path = os.path.abspath(os.path.join(self.folder, filename))
            if self.own_file and path == self.own_file:
                return "Permission denied: Cannot edit own file."
            if not self._is_within_folder(path):
                return "Permission denied: File must be within designated folder."
            return self.create_file(filename, new_content)
        except Exception as e:
            return f"Error editing file: {e}"

    def delete_file(self, filename):
        try:
            path = os.path.abspath(os.path.join(self.folder, filename))
            if self.own_file and path == self.own_file:
                return "Permission denied: Cannot delete own file."
            if not self._is_within_folder(path):
                return "Permission denied: File must be within designated folder."
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            return f"Error deleting file: {e}"

# MemoryManager stores and retrieves memory files and can recall past memory
class MemoryManager:
    def __init__(self, memory_folder=os.path.join(tempfile.gettempdir(), "firebot_memory")):
        self.folder = os.path.abspath(memory_folder)
        os.makedirs(self.folder, exist_ok=True)

    def save_memory(self, filename, content):
        try:
            path = os.path.join(self.folder, filename)
            with open(path, "w") as f:
                f.write(content)
        except Exception as e:
            return f"Error saving memory: {e}"

    def load_memory(self, filename):
        try:
            path = os.path.join(self.folder, filename)
            if os.path.exists(path):
                with open(path, "r") as f:
                    return f.read()
            return None
        except Exception as e:
            return f"Error loading memory: {e}"

    def list_memories(self):
        try:
            return os.listdir(self.folder)
        except Exception as e:
            return [f"Error listing memories: {e}"]

    def load_all_memories(self):
        try:
            texts = []
            for fname in self.list_memories():
                content = self.load_memory(fname)
                if content:
                    texts.append(content)
            return " ".join(texts)
        except Exception as e:
            return f"Error loading all memories: {e}"

# Utility functions to build vocabulary and tokenize

def build_vocab(corpus):
    tokens = list(set(corpus.lower().split()))
    word2idx = {word: idx for idx, word in enumerate(tokens)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

# TerminalInterface allows conversation with the AI in the terminal
class TerminalInterface:
    def __init__(self, model, word2idx, idx2word):
        self.model = model
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.memory_manager = MemoryManager()

    def neural_generate(self, prompt):
        tokens = prompt.lower().split()
        if not tokens:
            return "<no_input>"
        indices = torch.tensor([[self.word2idx.get(tokens[-1], 0)]]).long()  # Shape: (1,1)
        with torch.no_grad():
            output = self.model(indices)
            top_idx = torch.argmax(output, dim=1).item()
        return self.idx2word.get(top_idx, "<unk>")

    def chat(self):
        print("\n[Firebot is curious and smart. Type 'exit' to quit.]")
        all_memories = self.memory_manager.load_all_memories()
        while True:
            try:
                user_input = input("You: ")
                if user_input.strip().lower() == "exit":
                    print("Firebot: Goodbye!")
                    break

                memory_title = f"mem_{len(self.memory_manager.list_memories())}.txt"
                self.memory_manager.save_memory(memory_title, user_input)
                all_memories += " " + user_input

                response = []
                for _ in range(10):
                    next_word = self.neural_generate(all_memories)
                    response.append(next_word)
                    all_memories += " " + next_word

                print("Firebot:", " ".join(response))

            except Exception as e:
                print("[Error during chat]:", e)
                traceback.print_exc()

# Basic Python Tutorial (can be extended further)
def basic_python_tutorial():
    print("\n--- Python Tutorial ---")
    print("Variables store data: x = 5")
    print("Lists hold multiple items: fruits = ['apple', 'banana']")
    print("Loops repeat actions: for i in range(3): print(i)")
    print("Functions group code: def greet(): print('Hi')")
    print("Classes model real-world things: class Dog: pass")
    print("Comments help explain code: use # before your message")

# Example usage
if __name__ == "__main__":
    sample_corpus = "The quick brown fox jumps over the lazy dog. The dog barked at the fox."
    word2idx, idx2word = build_vocab(sample_corpus)
    corpus_indices = [word2idx[word] for word in sample_corpus.lower().split()]
    input_tensor = torch.tensor([corpus_indices[:-1]]).long()  # Shape: (1, sequence_length)
    target_tensor = torch.tensor(corpus_indices[1:]).long()

    model = SimpleNNLanguageModel(vocab_size=len(word2idx), embedding_dim=10, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.NLLLoss()

    for _ in range(100):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = loss_fn(output, target_tensor)
        loss.backward()
        optimizer.step()

    ti = TerminalInterface(model, word2idx, idx2word)
    basic_python_tutorial()
    ti.chat()
