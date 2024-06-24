import os
import time
import re

def ensure_folder_created_with_overwrite_prompt(folder_path):
    if os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            # ask if user wants to: overwrite, create new directory, or cancel
            while True:
                choice = input(f'Folder already exists: {folder_path}\nOptions: [O]verwrite, [C]reate new directory, [A]bort: ').lower()
                if choice == 'o':
                    break
                elif choice == 'c':
                    folder_path = input('Enter new directory path: ')
                    if not os.path.exists(folder_path):
                        break
                    else:
                        print(f'Error: Directory already exists: {folder_path}')
                elif choice == 'a':
                    raise FileExistsError(f'Folder already exists: {folder_path}')
                else:
                    print('Invalid choice. Please enter O, C, or A.')
    else:
        os.makedirs(folder_path)

    return folder_path

def ensure_folder_created(folder_path):
    print(f'Creating folder: {folder_path}')
    if os.path.exists(folder_path):
        print(f'Folder already exists: {folder_path}')
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def find_last_model(path, key: str) -> (str, str):
    if os.path.isdir(path):
        model_names = [model for model in os.listdir(path) if model.endswith('.pth')]
        print(f'Found {len(model_names)} models in {path}')
        if model_names:
            last_model = sorted(model_names, key=lambda x: int(re.match(rf'\S+_(\d+){key}', x).groups(1)[0]))[-1]
            return path, last_model

    if os.path.isfile(path) and path.endswith('.pt'):
        return os.path.dirname(path), os.path.basename(path)

    return None, None

def timeit(start_time) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

def class_exists(class_name):
    return class_name in globals() and isinstance(globals()[class_name], type)

def dict_except(d, *keys):
    return {k: v for k, v in d.items() if k not in keys}
