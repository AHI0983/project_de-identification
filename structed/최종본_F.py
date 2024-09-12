import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
import pandas as pd
import hashlib
import chardet

# 전역 변수
df1 = None
df2 = None
df_to_anonymize = None
anonymization_settings = {}
anonymized_columns = []

# 인코딩 감지 함수
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

# CSV 파일 로드 함수
def load_csv(file_path):
    encodings = ['utf-8', 'cp949', 'latin1']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    try:
        encoding = detect_encoding(file_path)
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load CSV file: {e}")
        return None

# CSV 병합 기능 함수들
def load_csv_1():
    global df1
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df1 = load_csv(file_path)
        if df1 is not None:
            listbox_csv1.delete(0, tk.END)
            for column in df1.columns:
                listbox_csv1.insert(tk.END, column)
            highlight_common_keys()

def load_csv_2():
    global df2
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df2 = load_csv(file_path)
        if df2 is not None:
            listbox_csv2.delete(0, tk.END)
            for column in df2.columns:
                listbox_csv2.insert(tk.END, column)
            highlight_common_keys()

# 공통 키 강조 함수 (노란색 강조)
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
        if common_columns:
            key_menu.current(0)

# CSV 파일 병합
def merge_csv():
    if df1 is not None and df2 is not None:
        selected_key = key_menu.get()
        if not selected_key:
            messagebox.showerror("Error", "Please select a key column to merge.")
            return

        df_merged = pd.merge(df1, df2, how='outer', on=selected_key)
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            try:
                df_merged.to_csv(save_path, index=False, encoding='utf-8-sig')
                messagebox.showinfo("Success", f"CSV files merged and saved at {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save the merged CSV: {e}")
    else:
        messagebox.showerror("Error", "Please load both CSV files first.")

# 익명화 기능 함수들
def load_csv_for_anonymize():
    global df_to_anonymize, anonymization_settings, anonymized_columns
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df_to_anonymize = load_csv(file_path)
        if df_to_anonymize is not None:
            anonymization_settings = {}
            anonymized_columns = []
            settings_listbox.delete(0, tk.END)

            # UI 요소 활성화
            toggle_ui_elements(state=tk.NORMAL)

            clear_anonymization_frame()
            for column in df_to_anonymize.columns:
                add_column_with_button(column)

            btn_anonymize.config(state=tk.NORMAL)
            btn_change_csv_anonymize.config(state=tk.NORMAL)

def change_csv_for_anonymize():
    global df_to_anonymize, anonymization_settings, anonymized_columns
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df_to_anonymize = load_csv(file_path)
        if df_to_anonymize is not None:
            clear_anonymization_frame()
            anonymization_settings = {}
            anonymized_columns = []
            settings_listbox.delete(0, tk.END)
            for column in df_to_anonymize.columns:
                add_column_with_button(column)

# UI 활성화/비활성화 함수
def toggle_ui_elements(state):
    settings_listbox.config(state=state)
    btn_anonymize.config(state=state)
    btn_change_csv_anonymize.config(state=state)

# 열을 추가하여 콤보박스를 추가하는 함수
def add_column_with_button(column):
    frame = tk.Frame(scrollable_frame, bg="white", padx=10, pady=5)  
    frame.pack(fill=tk.X, pady=5, padx=10)

    label = tk.Label(frame, text=column, font=("Arial", 10), bg="white", width=20, anchor='w')
    label.pack(side=tk.LEFT, padx=5)

    method_menu = ttk.Combobox(frame, state="readonly", values=[
        "Replace with **", "Replace with ***", "SHA-256 Encrypt", "Mask Phone",
        "Categorize Age", "Mask Address", "Round Up Square Footage", "Round Up Monthly Payment"], width=30)
    method_menu.pack(side=tk.LEFT, padx=5)

    button_add = tk.Button(frame, text="Add", command=lambda c=column: add_column_for_anonymization(c, method_menu),
                           bg="lightblue", relief=tk.RAISED, width=8)
    button_add.pack(side=tk.LEFT, padx=5)

    button_remove = tk.Button(frame, text="Remove", command=lambda: remove_column_for_anonymization(column),
                              bg="lightcoral", relief=tk.RAISED, width=8)
    button_remove.pack(side=tk.LEFT, padx=5)

# 익명화 열 추가
def add_column_for_anonymization(column_name, method_menu):
    method = method_menu.get()
    if column_name and method:
        anonymization_settings[column_name] = method
        anonymized_columns.append(column_name)
        refresh_settings_listbox()

# 익명화 열 제거
def remove_column_for_anonymization(column_name):
    if column_name in anonymization_settings:
        del anonymization_settings[column_name]
        anonymized_columns.remove(column_name)
        refresh_settings_listbox()

# 익명화 리스트 업데이트
def refresh_settings_listbox():
    settings_listbox.delete(0, tk.END)
    for column, method in anonymization_settings.items():
        settings_listbox.insert(tk.END, f"{column}: {method}")

def clear_anonymization_frame():
    for widget in scrollable_frame.winfo_children():
        if isinstance(widget, tk.Frame):
            widget.destroy()

# 익명화 수행
def anonymize_csv():
    if df_to_anonymize is not None and anonymization_settings:
        df_anonymized = df_to_anonymize.copy()
        for column, method in anonymization_settings.items():
            if method == "Replace with **":
                df_anonymized[column] = df_anonymized[column].apply(lambda x: '**' if pd.notna(x) else x)
            elif method == "Replace with ***":
                df_anonymized[column] = df_anonymized[column].apply(lambda x: '***' if pd.notna(x) else x)
            elif method == "SHA-256 Encrypt":
                df_anonymized[column] = df_anonymized[column].apply(sha256_text)

        save_path_anonymized = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path_anonymized:
            try:
                df_anonymized.to_csv(save_path_anonymized, index=False, encoding='utf-8-sig')
                messagebox.showinfo("Success", f"Anonymized CSV saved successfully as {save_path_anonymized}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save the anonymized CSV file: {e}")

def sha256_text(text):
    text_str = str(text)
    return hashlib.sha256(text_str.encode()).hexdigest()

# GUI 설정
window = tk.Tk()
window.title("CSV Merger and Anonymizer")
window.geometry("1200x800")  # 창 크기를 키움
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

tab_parent = ttk.Notebook(window)

# CSV 병합 탭 설정
merge_tab = ttk.Frame(tab_parent)
tab_parent.add(merge_tab, text="CSV Merge")

# 익명화 탭 설정
anonymize_tab = ttk.Frame(tab_parent)
tab_parent.add(anonymize_tab, text="Anonymize CSV")
tab_parent.pack(expand=1, fill='both')

### CSV Merge UI ###
frame_csv1 = tk.Frame(merge_tab)
frame_csv1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
frame_csv1.grid_rowconfigure(1, weight=1)
frame_csv1.grid_columnconfigure(0, weight=1)

frame_csv2 = tk.Frame(merge_tab)
frame_csv2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
frame_csv2.grid_rowconfigure(1, weight=1)
frame_csv2.grid_columnconfigure(0, weight=1)

label_csv1 = tk.Label(frame_csv1, text="CSV 1 Columns", anchor="w")
label_csv1.grid(row=0, column=0, sticky="nsew")

label_csv2 = tk.Label(frame_csv2, text="CSV 2 Columns", anchor="w")
label_csv2.grid(row=0, column=0, sticky="nsew")

# 리스트박스 크기 설정 (최대 크기)
listbox_csv1 = tk.Listbox(frame_csv1, height=25, width=50)  # height와 width를 최대한으로 설정
listbox_csv1.grid(row=1, column=0, sticky="nsew")

listbox_csv2 = tk.Listbox(frame_csv2, height=25, width=50)  # height와 width를 최대한으로 설정
listbox_csv2.grid(row=1, column=0, sticky="nsew")

scrollbar_csv1 = tk.Scrollbar(frame_csv1, orient=tk.VERTICAL, command=listbox_csv1.yview)
scrollbar_csv1.grid(row=1, column=1, sticky="ns")
listbox_csv1.configure(yscrollcommand=scrollbar_csv1.set)

scrollbar_csv2 = tk.Scrollbar(frame_csv2, orient=tk.VERTICAL, command=listbox_csv2.yview)
scrollbar_csv2.grid(row=1, column=1, sticky="ns")
listbox_csv2.configure(yscrollcommand=scrollbar_csv2.set)

# 리스트박스가 창 크기에 맞춰 커지도록 확장
frame_csv1.grid_rowconfigure(1, weight=1)
frame_csv1.grid_columnconfigure(0, weight=1)

frame_csv2.grid_rowconfigure(1, weight=1)
frame_csv2.grid_columnconfigure(0, weight=1)

btn_load_csv1 = tk.Button(merge_tab, text="Load CSV 1", command=load_csv_1)
btn_load_csv1.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")

btn_load_csv2 = tk.Button(merge_tab, text="Load CSV 2", command=load_csv_2)
btn_load_csv2.grid(row=2, column=1, padx=10, pady=5, sticky="nsew")

key_menu_label = tk.Label(merge_tab, text="Select Key Column:", anchor="w")
key_menu_label.grid(row=3, column=0, columnspan=2, sticky="nsew")

key_menu = ttk.Combobox(merge_tab, state="readonly")
key_menu.grid(row=4, column=0, columnspan=2, sticky="nsew")

btn_merge = tk.Button(merge_tab, text="Merge CSV", command=merge_csv)
btn_merge.grid(row=5, column=0, columnspan=2, pady=10, sticky="nsew")

# Configure resizing behavior for all widgets
merge_tab.grid_rowconfigure(1, weight=1)
merge_tab.grid_columnconfigure(0, weight=1)
merge_tab.grid_columnconfigure(1, weight=1)

### Anonymize CSV UI ###
frame_anonymize = tk.Frame(anonymize_tab)
frame_anonymize.pack(side=tk.TOP, fill=tk.X)

btn_load_csv_anonymize = tk.Button(frame_anonymize, text="Load CSV to Anonymize", command=load_csv_for_anonymize, font=("Arial", 10))
btn_load_csv_anonymize.pack(side=tk.LEFT, padx=10, pady=10)

button_frame = tk.Frame(anonymize_tab)
button_frame.pack(side=tk.BOTTOM, pady=20)

btn_anonymize = tk.Button(button_frame, text="Perform Anonymization", command=anonymize_csv, state=tk.DISABLED, font=("Arial", 10))
btn_anonymize.grid(row=0, column=0, padx=10)

btn_change_csv_anonymize = tk.Button(button_frame, text="Change CSV", command=change_csv_for_anonymize, state=tk.DISABLED, font=("Arial", 10))
btn_change_csv_anonymize.grid(row=0, column=1, padx=10)

# 익명화 부분의 두 박스 크기를 맞춤 (크게 설정)
settings_listbox = tk.Listbox(anonymize_tab, font=("Arial", 10), height=20, width=50, state=tk.DISABLED)  # height와 width 설정
settings_listbox.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

canvas = tk.Canvas(anonymize_tab, width=500, height=400)  # 크기를 설정
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(anonymize_tab, orient="vertical", command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill="y")

scrollable_frame = tk.Frame(canvas, bg="white")
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

window.mainloop()
