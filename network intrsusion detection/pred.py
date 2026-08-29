import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import pickle
import pyttsx3


# LOAD MODEL FILES
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))


# LOAD DATASET
df = pd.read_csv("02-16-2018.csv",low_memory=False)

# SAME CLEANING AS TRAINING
drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# REMOVE LEAKAGE FEATURES
leak_cols = [
    'Flow Bytes/s',
    'Flow Packets/s',
    'Fwd Packets/s',
    'Bwd Packets/s'
]
df.drop(columns=[col for col in leak_cols if col in df.columns], inplace=True)

df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.dropna(inplace=True)


# VOICE ENGINE
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()


# MAIN FUNCTION
def run_detection():
    try:
        tree.delete(*tree.get_children())

        user_input = entry.get()

        if "-" in user_input:
            start, end = map(int, user_input.split("-"))
        else:
            start = int(user_input)
            end = start + 1   # just one row

        data = df.iloc[start:end]

        X = data.drop("Label", axis=1)

        # SAME PROCESSING AS TRAINING
        X = X.apply(pd.to_numeric, errors='coerce')
        X.fillna(X.median(), inplace=True)

        X_scaled = scaler.transform(X)

        preds = model.predict(X_scaled)
        labels = le.inverse_transform(preds)

        results = []

        attack_count = 0

        for i, label in enumerate(labels):
            row_no = start + i

            tree.insert("", "end", values=(row_no, label))

            if label.lower() != "benign":
                attack_count += 1
                speak(f"Intrusion detected {label} at row {row_no}")

            results.append({"Row": row_no, "Prediction": label})

        # SAVE RESULTS
        pd.DataFrame(results).to_excel("results.xlsx", index=False)

        status_label.config(
            text=f"Completed! {attack_count} attacks detected",
            fg="red" if attack_count > 0 else "green"
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))


# UI DESIGN
root = tk.Tk()
root.title("Network Intrusion Detection System")
root.geometry("900x600")
root.configure(bg="#121212")

# HEADER
header = tk.Label(
    root,
    text="NETWORK INTRUSION DETECTION SYSTEM",
    font=("Helvetica", 22, "bold"),
    bg="#0f3460",
    fg="white",
    pady=15
)
header.pack(fill="x")

# INPUT FRAME
frame = tk.Frame(root, bg="#121212")
frame.pack(pady=20)

tk.Label(frame, text="Enter Row Range (e.g., 10-50):",
         font=("Arial", 14),
         bg="#121212", fg="white").grid(row=0, column=0, padx=10)

entry = tk.Entry(frame, font=("Arial", 14), width=15)
entry.grid(row=0, column=1, padx=10)

btn = tk.Button(frame,
                text="Run Detection",
                font=("Arial", 13, "bold"),
                bg="#e94560",
                fg="white",
                command=run_detection)
btn.grid(row=0, column=2, padx=10)

# TABLE FRAME
table_frame = tk.Frame(root)
table_frame.pack(pady=20, fill="both", expand=True)

scroll_y = tk.Scrollbar(table_frame)
scroll_y.pack(side="right", fill="y")

tree = ttk.Treeview(
    table_frame,
    columns=("Row", "Prediction"),
    show="headings",
    yscrollcommand=scroll_y.set
)

tree.heading("Row", text="Row Number")
tree.heading("Prediction", text="Prediction")

tree.column("Row", width=100, anchor="center")
tree.column("Prediction", width=200, anchor="center")

tree.pack(fill="both", expand=True)
scroll_y.config(command=tree.yview)

# STATUS
status_label = tk.Label(root,
                        text="Waiting for input...",
                        font=("Arial", 12),
                        bg="#121212",
                        fg="yellow")
status_label.pack(pady=10)

# FOOTER
footer = tk.Label(root,
                  text="Developed for Intrusion Detection Project",
                  font=("Arial", 10),
                  bg="#0f3460",
                  fg="white")
footer.pack(fill="x", side="bottom")

root.mainloop()