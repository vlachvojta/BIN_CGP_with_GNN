import os

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
