# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:49:13 2024

@author: hugop
"""

import tkinter as tk
from tkinter import messagebox
import random
import string

def generate_password():
    try:
        length = int(length_entry.get())
        if length <= 0:
            raise ValueError("La longueur doit être un nombre positif.")
        
        characters = string.ascii_letters + string.digits + string.punctuation
        password = ''.join(random.choice(characters) for _ in range(length))
        password_entry.delete(0, tk.END)
        password_entry.insert(0, password)
    except ValueError as e:
        messagebox.showerror("Erreur de saisie", f"Entrée invalide : {e}")

app = tk.Tk()
app.title("Générateur de Mot de Passe")

frame = tk.Frame(app)
frame.pack(pady=20, padx=20)

length_label = tk.Label(frame, text="Longueur du mot de passe:")
length_label.grid(row=0, column=0, padx=5, pady=5)

length_entry = tk.Entry(frame)
length_entry.grid(row=0, column=1, padx=5, pady=5)

generate_button = tk.Button(frame, text="Générer", command=generate_password)
generate_button.grid(row=0, column=2, padx=5, pady=5)

password_label = tk.Label(frame, text="Mot de passe généré:")
password_label.grid(row=1, column=0, padx=5, pady=5)

password_entry = tk.Entry(frame, width=30)
password_entry.grid(row=1, column=1, padx=5, pady=5, columnspan=2)

app.mainloop()
