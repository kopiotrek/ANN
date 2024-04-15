import os
def rename_files(folder_path, new_name_prefix):
    for count, filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            new_filename = f"{new_name_prefix}_{count}.jpg"  # Zakładamy format .jpg
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(file_path, new_file_path)
            print(f"Renamed {file_path} to {new_file_path}")

# Zmień nazwy plików w folderze No_findings
rename_files('/home/koczka/Documents/ANN/\\No_findings', 'no_findings')
