import tkinter as tk
from tkinter import filedialog
import os
import re
 
def select_folder():
    folder_path = filedialog.askdirectory()
    entry_folder.delete(0, tk.END)
    entry_folder.insert(0, folder_path)
 
def search_and_remove():
    folder_path = entry_folder.get()
    if not os.path.exists(folder_path):
        tk.messagebox.showerror("错误", "文件夹不存在")
        return
 
    files = os.listdir(folder_path)
    for file in files:
        if re.search(r'\[[^]]*\]', file):
            # 构建新的文件名
            new_filename = re.sub(r'\[.*?\]', '', file)
            # 重命名文件
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_filename))
            print(f"已成功处理文件: {file} -> {new_filename}")
 
root = tk.Tk()
root.title("搜索并删除含有[]的文件名")
 
label_folder = tk.Label(root, text="请选择文件夹:")
label_folder.pack()
 
entry_folder = tk.Entry(root)
entry_folder.pack()
 
button_folder = tk.Button(root, text="选择文件夹", command=select_folder)
button_folder.pack()
 
button_search = tk.Button(root, text="搜索并删除", command=search_and_remove)
button_search.pack()
root.mainloop()
