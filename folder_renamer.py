import os


def rename_folders(directory):
    for folder_name in os.listdir(directory):
        new_folder_name = folder_name.replace(' ', '_')

        old_path = os.path.join(directory, folder_name)
        new_path = os.path.join(directory, new_folder_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {folder_name} -> {new_folder_name}")


current_directory = os.getcwd()
test_directory = os.path.join(current_directory, "test")
train_directory = os.path.join(current_directory, "train")
rename_folders(test_directory)
rename_folders(train_directory)
