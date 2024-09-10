import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import hashlib

# 비식별화 함수들
def column_replace(df, column_name, num_chars):
    df[column_name] = df[column_name].apply(lambda x: str(x)[:num_chars] + "**")

def sha256Text(salt_value, text):
    return hashlib.sha256(salt_value.encode() + text.encode()).hexdigest()

def column_encrypt(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: sha256Text('password', str(x)))

# CSV 파일 병합
def load_csv_1():
    global df1
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df1 = pd.read_csv(file_path)
        listbox_csv1.delete(0, tk.END)
        for column in df1.columns:
            listbox_csv1.insert(tk.END, column)
        highlight_common_keys()

def load_csv_2():
    global df2
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df2 = pd.read_csv(file_path)
        listbox_csv2.delete(0, tk.END)
        for column in df2.columns:
            listbox_csv2.insert(tk.END, column)
        highlight_common_keys()

def highlight_common_keys():
    for i in range(listbox_csv1.size()):
        listbox_csv1.itemconfig(i, bg="white")
    for i in range(listbox_csv2.size()):
        listbox_csv2.itemconfig(i, bg="white")

    if df1 is not None and df2 is not None:
        common_columns = set(df1.columns).intersection(set(df2.columns))
        for i, column in enumerate(df1.columns):
            if column in common_columns:
                listbox_csv1.itemconfig(i, bg="yellow")
        for i, column in enumerate(df2.columns):
            if column in common_columns:
                listbox_csv2.itemconfig(i, bg="yellow")

        key_menu['values'] = list(common_columns)
        key_menu.current(0)

def merge_csv():
    if df1 is not None and df2 is not None:
        selected_key = key_menu.get()
        if not selected_key:
            messagebox.showerror("Error", "Please select a key column for merging.")
            return

        df_to_merge = pd.merge(df1, df2, how='outer', on=selected_key)
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            df_to_merge.to_csv(save_path, sep=",", index=False)
            messagebox.showinfo("Success", f"CSV files merged and saved successfully as {save_path}")
    else:
        messagebox.showerror("Error", "Please load two CSV files first.")

# 비식별화 기능
def load_csv_for_anonymize():
    global df_to_anonymize, anonymization_settings
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df_to_anonymize = pd.read_csv(file_path)
        listbox_anonymize.delete(0, tk.END)
        anonymization_settings = {}
        for column in df_to_anonymize.columns:
            listbox_anonymize.insert(tk.END, column)
        method_menu.set('')
        update_display()

def on_column_select(event):
    if listbox_anonymize.curselection():
        selected_column = listbox_anonymize.get(listbox_anonymize.curselection())
        if selected_column in anonymization_settings:
            method_menu.set(anonymization_settings[selected_column])
        else:
            method_menu.set('')

def on_method_select(event):
    if listbox_anonymize.curselection():
        selected_column = listbox_anonymize.get(listbox_anonymize.curselection())
        selected_method = method_menu.get()
        if selected_column:
            anonymization_settings[selected_column] = selected_method
            update_display()

def update_display():
    settings_listbox.delete(0, tk.END)
    for column, method in anonymization_settings.items():
        settings_listbox.insert(tk.END, f"{column}: {method}")

def anonymize_csv():
    if df_to_anonymize is not None:
        for column_name, method in anonymization_settings.items():
            if method == "Replace with **":
                column_replace(df_to_anonymize, column_name, 2)
            elif method == "Replace with ***":
                column_replace(df_to_anonymize, column_name, 3)
            elif method == "SHA-256 Encrypt":
                column_encrypt(df_to_anonymize, column_name)

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            df_to_anonymize.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"CSV file anonymized and saved successfully as {save_path}")
    else:
        messagebox.showerror("Error", "Please load a CSV file first.")

# GUI 설정
root = tk.Tk()
root.title("CSV Tool")
root.geometry("600x600")

tabControl = ttk.Notebook(root)
merge_tab = ttk.Frame(tabControl)
anonymize_tab = ttk.Frame(tabControl)
tabControl.add(merge_tab, text="CSV Merge")
tabControl.add(anonymize_tab, text="Anonymization")
tabControl.pack(expand=1, fill="both")

# Merge Tab
df1 = None
df2 = None
ttk.Label(merge_tab, text="CSV File 1:").pack(pady=5)
btn_load_csv1 = ttk.Button(merge_tab, text="Load CSV 1", command=load_csv_1)
btn_load_csv1.pack(pady=5)
listbox_csv1 = tk.Listbox(merge_tab)
listbox_csv1.pack(pady=5, fill=tk.X)

ttk.Label(merge_tab, text="CSV File 2:").pack(pady=5)
btn_load_csv2 = ttk.Button(merge_tab, text="Load CSV 2", command=load_csv_2)
btn_load_csv2.pack(pady=5)
listbox_csv2 = tk.Listbox(merge_tab)
listbox_csv2.pack(pady=5, fill=tk.X)

ttk.Label(merge_tab, text="Select Key Column:").pack(pady=5)
key_menu = ttk.Combobox(merge_tab)
key_menu.pack(pady=5)

btn_merge_csv = ttk.Button(merge_tab, text="Merge CSV Files", command=merge_csv)
btn_merge_csv.pack(pady=20)

# Anonymization Tab
df_to_anonymize = None
anonymization_settings = {}

ttk.Label(anonymize_tab, text="CSV File to Anonymize:").pack(pady=5)
btn_load_anonymize = ttk.Button(anonymize_tab, text="Load CSV", command=load_csv_for_anonymize)
btn_load_anonymize.pack(pady=5)
listbox_anonymize = tk.Listbox(anonymize_tab)
listbox_anonymize.pack(pady=5, fill=tk.X)
listbox_anonymize.bind('<<ListboxSelect>>', on_column_select)

ttk.Label(anonymize_tab, text="Select Method:").pack(pady=5)
method_menu = ttk.Combobox(anonymize_tab, values=["None", "Replace with **", "Replace with ***", "SHA-256 Encrypt"])
method_menu.pack(pady=5)
method_menu.bind("<<ComboboxSelected>>", on_method_select)

ttk.Label(anonymize_tab, text="Anonymization Settings:").pack(pady=5)
settings_listbox = tk.Listbox(anonymize_tab)
settings_listbox.pack(pady=5, fill=tk.X)

btn_anonymize_csv = ttk.Button(anonymize_tab, text="Anonymize CSV", command=anonymize_csv)
btn_anonymize_csv.pack(pady=20)

root.mainloop()
