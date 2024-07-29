import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

folder_path = ""


def select_folder():
    global folder_path
    folder_path = filedialog.askdirectory()
    folder_label.config(text=folder_path)
    return folder_path


def input_chars():
    chars_to_delete = delete_entry.get()
    if chars_to_delete:
        root.after(100, rename_files, chars_to_delete)
    else:
        messagebox.showwarning("警告", "请输入要删除的字符！")


def rename_files(chars_to_delete):
    global folder_path
    if folder_path:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if chars_to_delete in file:
                    # 构建新的文件名
                    new_filename = file.replace(chars_to_delete, '')
                    # 获取旧文件名和新的文件名的完整路径
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.join(root, new_filename)
                    # 重命名文件
                    os.rename(old_file_path, new_file_path)
        messagebox.showinfo("完成", "所有文件名中包含指定字符的文件都已重命名。")
    else:
        messagebox.showwarning("警告", "请选择文件夹！")


root = tk.Tk()
root.title("批量重命名文件名中字符")

folder_label = tk.Label(root, text="请选择文件夹：")
folder_label.pack()

folder_button = tk.Button(root, text="选择文件夹", command=select_folder)
folder_button.pack()

delete_label = tk.Label(root, text="请输入要删除的字符：")
delete_label.pack()

delete_entry = tk.Entry(root)
delete_entry.pack()

input_button = tk.Button(root, text="重命名", command=input_chars)
input_button.pack()

root.mainloop()
