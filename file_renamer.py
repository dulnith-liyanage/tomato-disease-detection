import os


def rename_files(directory, main_dir):
    for root, dirs, files in os.walk(directory):
        for count, file in enumerate(files, start=1):
            # Extract subfolder name from the path
            sub_folder_name = os.path.basename(root)

            # Generate the new filename
            new_name = f"{main_dir}_{sub_folder_name}_{count}.jpg"

            # Create the new path for the file
            new_path = os.path.join(root, new_name)

            # Check if the file already exists
            while os.path.exists(new_path):
                count += 1
                new_name = f"{main_dir}_{sub_folder_name}_{count}.jpg"
                new_path = os.path.join(root, new_name)

            # Rename the file
            os.rename(os.path.join(root, file), new_path)
            print(f"Renamed: {file} to {new_name}")
            print(file)


# Specify the directory path where you want to rename files
current_directory = os.getcwd()
test_directory = os.path.join(current_directory, "test")
rename_files(test_directory, "test")

train_directory = os.path.join(current_directory, "train")
rename_files(train_directory, "train")
