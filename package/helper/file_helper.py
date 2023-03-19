import glob
import os
import matplotlib.pyplot as plt

def change_extension_of_file(path,
                             search_extension, 
                             replace_extension
                            ):
    '''
    Change the extension of files
    
    :param path: the path of file where it is located
    :param search_extension: extension to be searched
    :param replace_extension: extension to be replaced
    '''
    
    print('\n\n ----- Starting the change in extension ---- \n\n')
    
    new_path = os.path.join(path, f'*{search_extension}')

    for image in glob.glob(new_path):
        parts = image.split(os.sep)[-1]
        name = parts.split('.')[0]
        new_name = name + '.' +  replace_extension
        new_name_full = os.path.join(path, new_name)
    
        os.rename(image, new_name_full)
        
    print('\n\n ----- Ending the change in extension ---- \n\n')
    
    
def show_image(image):
    '''
    Shows image file
    
    :param image: Shows image
    '''
    
    im = plt.imread(image)
    
    plt.imshow(im)
    plt.show()
    
    
def write_to_file_row_each_line(path, file_name=None, 
                                rows=None):
    '''
    Writes each row as one line in file
    :param path: path where file is to be saved
    :param file_name: name of new file
    :param rows: list of row to be written
    '''
    if file_name != None:
        path_with_name_of_file = os.path.join(path, file_name)
    else:
        path_with_name_of_file = path

    with open(f'{path_with_name_of_file}', 'w') as f:
        for line in rows:
            f.write(f"{line}\n")
            
            
def create_folder(output_path, folder_name):
    '''
    Creates folder in output path if folder does not exists
    
    :param output_path: path where folder existence to be
    checked
    :param folder_name: name of folder
    '''
    
    path = os.path.join(output_path, folder_name)
    
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)
        
    return path
        
def read_file(file):
    '''
    This function read files and return the content
    :param file: file name with location
    '''
    with open(f'{file}') as f:
        lines = f.read().splitlines()

        return lines
    
    
    
def remove_row_in_file(path, org_list, save_path=None):
    '''
    Removes unwanted rows from file
    :param path: Path where files are
    :param save_path: Path where the files to be saved
    :param org_list: List of original data without unwanted rows
    '''
    for file in glob.glob(path):
        filename = file.split(os.sep)[-1]
        ids = set(read_file(file))
        org_list = set(org_list)
        
        remaining = ids - org_list #unwanted rows
        
        if len(remaining) == 0:
            print(0)
            continue
            
        ids = ids - remaining
        temp_file = open(f'{file}', 'r+')
        
        temp_file.truncate(0)
        
        if save_path == None:
            save_path = path
            filename = None
            
        write_to_file_row_each_line(save_path, 
                                            filename, 
                                            ids)

def empty_the_file(filename):
    '''
    Empty the text file
    :param filename: name of file with path
    '''
    temp_file = open(f'{filename}', 'r+')
        
    temp_file.truncate(0)

        
            