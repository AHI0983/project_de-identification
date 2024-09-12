import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
import pandas as pd
import hashlib
import chardet
import re

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
        "Mask Name", "Mask Address", "Mask Phone", "Mask Age", "SHA-256 Encrypt", "Custom Mask"], width=30)
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

# 이름 익명화 처리
def mask_name(name):
    if pd.isna(name):
        return name
    if len(name) > 3:
        return name[0] + '*' * (len(name) - 2) + name[-1]
    else:
        return name[0] + '*' + name[-1]

# 주소 익명화 처리 (도 지역 및 특례시 포함)
def mask_address(address):
    if pd.isna(address):
        return address

    # 특별시, 광역시 패턴
    city_patterns = {
        "서울": "서울특별시",
        "대구": "대구광역시",
        "인천": "인천광역시",
        "부산": "부산광역시",
        "광주": "광주광역시",
        "대전": "대전광역시",
        "울산": "울산광역시",
        "세종": "세종특별자치시",
    }

    # 도 패턴 (도 이름만 남기고 나머지 제거)
    province_patterns = {
        "경기도": "경기도",
        "강원도": "강원도",
        "충청북도": "충청북도",
        "충청남도": "충청남도",
        "전라북도": "전라북도",
        "전라남도": "전라남도",
        "경상북도": "경상북도",
        "경상남도": "경상남도",
        "제주특별자치도": "제주특별자치도",
    }

    # 특별시, 광역시 비식별화
    for city, masked_city in city_patterns.items():
        if city in address:
            if masked_city == "세종특별자치시":
                return "세종시"
            return masked_city

    # 도 지역 비식별화
    for province, masked_province in province_patterns.items():
        if province in address:
            return masked_province

    return address

# 전화번호 익명화 처리 (지역 번호 및 휴대폰 번호 모두 처리)
def mask_phone(phone_number):
    if pd.isna(phone_number):
        return phone_number
    phone_number = str(phone_number)
    
    # 휴대폰 번호 처리
    if phone_number.startswith('010'):
        return phone_number[:4] + '****' + phone_number[-4:]
    
    # 지역 번호 처리
    area_codes = ['02', '031', '032', '033', '041', '042', '043', '044', '051', '052', '053', '054', '055', '061', '062', '063', '064']
    for code in area_codes:
        if phone_number.startswith(code):
            return code + '-' + '****-' + phone_number.split('-')[-1]

    # 그 외 번호는 070으로 처리 (인터넷 전화 등)
    return '070-' + '****-' + phone_number[-4:]

# 나이 비식별화 처리
def mask_age(age):
    if pd.isna(age):
        return age
    age = int(age)
    return f"{age//10*10}대"

# 임의의 컬럼에 마스킹 처리
def custom_mask(value):
    if pd.isna(value):
        return value
    return '*' * len(str(value))

# SHA-256 암호화 처리
def sha256_text(text):
    text_str = str(text)
    return hashlib.sha256(text_str.encode()).hexdigest()

# 익명화된 값과 원본 값을 매핑하는 함수
def anonymize_column_with_mapping(column_data, anonymize_function):
    mapping = []
    anonymized_column = []

    for value in column_data:
        anonymized_value = anonymize_function(value)
        mapping.append((value, anonymized_value))
        anonymized_column.append(anonymized_value)

    return anonymized_column, mapping

# 매핑 테이블을 데이터프레임으로 생성
def create_mapping_dataframe(mapping_table):
    mapping_data = []
    for column, mappings in mapping_table.items():
        for original, anonymized in mappings:
            mapping_data.append([column, original, anonymized])
    
    return pd.DataFrame(mapping_data, columns=['Column', 'Original Value', 'Anonymized Value'])

# 익명화 수행
def anonymize_csv():
    if df_to_anonymize is not None and anonymization_settings:
        df_anonymized = df_to_anonymize.copy()
        mapping_table = {}  # 익명화된 값과 원본 값의 매핑을 저장할 테이블
        
        for column, method in anonymization_settings.items():
            if method == "Mask Name":
                df_anonymized[column], mapping = anonymize_column_with_mapping(df_anonymized[column], mask_name)
                mapping_table[column] = mapping
            elif method == "Mask Address":
                df_anonymized[column], mapping = anonymize_column_with_mapping(df_anonymized[column], mask_address)
                mapping_table[column] = mapping
            elif method == "Mask Phone":
                df_anonymized[column], mapping = anonymize_column_with_mapping(df_anonymized[column], mask_phone)
                mapping_table[column] = mapping
            elif method == "Mask Age":
                df_anonymized[column], mapping = anonymize_column_with_mapping(df_anonymized[column], mask_age)
                mapping_table[column] = mapping
            elif method == "Custom Mask":
                df_anonymized[column], mapping = anonymize_column_with_mapping(df_anonymized[column], custom_mask)
                mapping_table[column] = mapping
            elif method == "SHA-256 Encrypt":
                df_anonymized[column], mapping = anonymize_column_with_mapping(df_anonymized[column], sha256_text)
                mapping_table[column] = mapping

        # 익명화된 데이터 저장
        save_path_anonymized = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path_anonymized:
            try:
                # 익명화된 데이터 저장
                df_anonymized.to_csv(save_path_anonymized, index=False, encoding='utf-8-sig')

                # 매핑 테이블 저장
                mapping_df = create_mapping_dataframe(mapping_table)
                save_path_mapping = save_path_anonymized.replace('.csv', '_mapping.csv')
                mapping_df.to_csv(save_path_mapping, index=False, encoding='utf-8-sig')

                messagebox.showinfo("Success", f"Anonymized CSV saved successfully as {save_path_anonymized} and mapping table as {save_path_mapping}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save the anonymized CSV or mapping table: {e}")

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

btn_change_csv_anonymize = tk.Button(button_frame, text="Change CSV", command=load_csv_for_anonymize, state=tk.DISABLED, font=("Arial", 10))
btn_change_csv_anonymize.grid(row=0, column=1, padx=10)

settings_listbox = tk.Listbox(anonymize_tab, font=("Arial", 10), height=20, width=50, state=tk.DISABLED)
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
