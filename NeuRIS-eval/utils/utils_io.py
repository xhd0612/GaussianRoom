from datetime import datetime
import os, sys, logging
import shutil
import subprocess
from pathlib import Path
import glob


# Path
def checkExistence(path):
    if not os.path.exists(path):
        return False
    else:
        return True

def ensure_dir_existence(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            logging.info(f"Dir is already existent: {dir}")
    except Exception:
        logging.error(f"Fail to create dir: {dir}")
        exit()

def get_path_components(path):
    path = Path(path)
    ppath = str(path.parent)
    stem = str(path.stem)
    ext = str(path.suffix)
    return ppath, stem, ext

def add_file_name_suffix(path_file, suffix):
    ppath, stem, ext = get_path_components(path_file)
    
    path_name_new = ppath + "/" + stem + str(suffix) + ext
    return path_name_new

def add_file_name_prefix(path_file, prefix, check_exist = True):
    '''Add prefix before file name
    '''
    ppath, stem, ext = get_path_components(path_file)
    path_name_new = ppath + "/" + str(prefix) + stem  + ext

    if check_exist:
        ensure_dir_existence(ppath + "/" + str(prefix))
      
    return path_name_new

def add_file_name_prefix_and_suffix(path_file, prefix, suffix, check_exist = True):
    path_file_p = add_file_name_prefix(path_file, prefix, check_exist = True)
    path_file_p_s = add_file_name_suffix(path_file_p, suffix)
    return path_file_p_s

def get_files_stem(dir, ext_file):
    '''Get stems of all files in directory with target extension
    Return:
        vec_stem
    '''
    vec_path = sorted(glob.glob(f'{dir}/**{ext_file}'))
    vec_stem = []
    for i in range(len(vec_path)):
        pparent, stem, ext = get_path_components(vec_path[i])
        vec_stem.append(stem)
    return vec_stem

def get_files_path(dir, ext_file):
    return sorted(glob.glob(f'{dir}/**{ext_file}'))

# IO
def readLines(path_txt):
    fTxt = open(path_txt, "r")
    lines = fTxt.readlines()
    return lines

def copy_file(source_path, target_dir):
    try: 
        ppath, stem, ext = get_path_components(target_dir)
        ensure_dir_existence(ppath)
        shutil.copy(source_path, target_dir)
    except Exception:
        logging.error(f"Fail to copy file: {source_path}")
        exit(-1)

def remove_dir(dir):
    try:
        shutil.rmtree(dir)
    except Exception as ERROR_MSG:
        logging.error(f"{ERROR_MSG}.\nFail to remove dir: {dir}")
        exit(-1)

def copy_dir(source_dir, target_dir):
    try:
        if not os.path.exists(source_dir):
            logging.error(f"source_dir {source_dir} is not exist. Fail to copy directory.")
            exit(-1)
        shutil.copytree(source_dir, target_dir)
    except Exception as ERROR_MSG:
        logging.error(f"{ERROR_MSG}.\nFail to copy file: {source_dir}")
        exit(-1)

def INFO_MSG(msg):
    print(msg)
    sys.stdout.flush()
    
def changeWorkingDir(working_dir):
    try:
        os.chdir(working_dir)
        print(f"Current working directory is { os.getcwd()}.")
    except OSError:
        print("Cann't change current working directory.")
        sys.stdout.flush()
        exit(-1)

def run_subprocess(process_args):
    pProcess = subprocess.Popen(process_args)
    pProcess.wait()
    
def find_target_file(dir, file_name):
    all_files_recur = glob.glob(f'{dir}/**{file_name}*', recursive=True)
    path_target = None
    if len(all_files_recur) == 1:
        path_target = all_files_recur[0]

    assert not len(all_files_recur) > 1
    return path_target

def copy_files_in_dir(dir_src, dir_target, ext_file, rename_mode = 'stem'):
    '''Copy files in dir and rename it if needed
    '''
    ensure_dir_existence(dir_target)
    vec_path_files = sorted(glob.glob(f'{dir_src}/*{ext_file}'))
    for i in range(len(vec_path_files)):
        path_src = vec_path_files[i]
    
        if rename_mode == 'stem':
            pp, stem, _ = get_path_components(path_src)
            path_target = f'{dir_target}/{stem}{ext_file}'
        elif rename_mode == 'order':
            path_target = f'{dir_target}/{i}{ext_file}'
        elif rename_mode == 'order_04d':
            path_target = f'{dir_target}/{i:04d}{ext_file}'
        else:
            NotImplementedError

        copy_file(path_src, path_target)
    return len(vec_path_files)

# time-related funcs
def get_consumed_time(t_start):
    '''
    Return:
        time: seconds
    '''
    t_end = datetime.now()
    return (t_end-t_start).total_seconds()

def get_time_str(fmt='HMSM'):
    if fmt == 'YMD-HMS':
        str_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    elif fmt == 'HMS':
        str_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    elif fmt == 'HMSM':
        str_time = datetime.now().strftime("%H_%M_%S_%f")
    return str_time

def write_list_to_txt(path_list, data_list):
    num_lines = len(data_list)
    with open(path_list, 'w') as flis:
        for i in range(len(data_list)):
            flis.write(f'{data_list[i]}\n')
            
if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=sys.stdout,
        filename='example.log'
    )