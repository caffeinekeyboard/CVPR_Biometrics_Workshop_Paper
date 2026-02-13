import os
import shutil

def flatten_png_files(target_folder):
    target_folder = os.path.abspath(target_folder)
    if not os.path.exists(target_folder):
        print(f"Error: The folder '{target_folder}' does not exist.")
        return
    print(f"Processing folder: {target_folder}")
    for root, dirs, files in os.walk(target_folder):
        if root == target_folder:
            continue
        for file in files:
            if file.lower().endswith('.png'):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(target_folder, file)
                if os.path.exists(destination_path):
                    base, extension = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(destination_path):
                        new_name = f"{base}_{counter}{extension}"
                        destination_path = os.path.join(target_folder, new_name)
                        counter += 1
                try:
                    shutil.move(source_path, destination_path)
                    print(f"Moved: {file} -> {os.path.basename(destination_path)}")
                except Exception as e:
                    print(f"Error moving {file}: {e}")
    for root, dirs, files in os.walk(target_folder, topdown=False):
        if root == target_folder:
            continue      
        try:
            shutil.rmtree(root)
            print(f"Deleted folder: {root}")
        except Exception as e:
            print(f"Could not delete {root}: {e}")
if __name__ == "__main__":
    finger_type = "Double_Loop_Only"
    folder_paths = [f"/home/caffeinekeyboard/Codex/ICML_Workshop_Paper/data/{finger_type}/Noise_Level_0/", f"/home/caffeinekeyboard/Codex/ICML_Workshop_Paper/data/{finger_type}/Noise_Level_5/", f"/home/caffeinekeyboard/Codex/ICML_Workshop_Paper/data/{finger_type}/Noise_Level_10/", f"/home/caffeinekeyboard/Codex/ICML_Workshop_Paper/data/{finger_type}/Noise_Level_15/", f"/home/caffeinekeyboard/Codex/ICML_Workshop_Paper/data/{finger_type}/Noise_Level_20/"] 
    
    for folder_path in folder_paths:
        confirm = input(f"This will move all PNGs to the top level of '{folder_path}' and DELETE all subfolders (and any other files inside them). Are you sure? (y/n): ")
        if confirm.lower() == 'y':
            flatten_png_files(folder_path)
            print("Operation complete.")
        else:
            print("Operation cancelled.")
