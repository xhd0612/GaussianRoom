#!/usr/bin/env python
# -*- coding: utf-8 -*- 

from argparse import ArgumentParser
from datetime import datetime
import os, json

# --------------------------------------------extract txt--------------------------------------------
# 定义函数来提取文件中的第三行数据
def extract_metrics_from_file(file_path):
    metrics = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        third_line = lines[2].strip()  # 获取第三行数据
        parts = third_line.split('&')
        scene = parts[0].strip()
        values = [float(part.strip().rstrip('\\')) for part in parts[1:]]  # 去除末尾的空格和反斜杠
        metrics.append((scene, values))
    return metrics

# 定义函数来计算均值
def calculate_average_sdf(metrics):
    num_scenes = len(metrics)
    if num_scenes == 0:
        return []

    num_metrics = len(metrics[0][1])
    averages = [0] * num_metrics

    for _, values in metrics:
        for i in range(num_metrics):
            averages[i] += values[i]

    averages = [value / num_scenes for value in averages]
    return averages

# 定义函数来格式化输出结果
def format_output(scene_metrics, average_metrics):
    # formatted_average_metrics = [f"{value:.3f}" for value in average_metrics]
    formatted_average_metrics = [f"{value:.3f}" for value in average_metrics]
    
    output = ""
    for scene, values in scene_metrics:
        output += f"{scene} & {' & '.join(map(str, values))} \\\\ \n"
    output += f"Mean & {' & '.join(formatted_average_metrics)} \\\\ \n"
    return output

# 定义主函数来处理多个文件
def process_files(file_paths):
    all_metrics = []
    for file_path in file_paths:
        metrics = extract_metrics_from_file(file_path)
        all_metrics.extend(metrics)

    average_metrics = calculate_average_sdf(all_metrics)
    output = format_output(all_metrics, average_metrics)
    return output

# --------------------------------------------extract json--------------------------------------------

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_result_json(directory):
    parameters = {}
    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "results.json":
                file_path = os.path.join(root, file)
                return file_path

    return

# def read_json_file(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

def calculate_average_gs(parameters):
    num_files = len(parameters)
    if num_files == 0:
        return {}

    # 初始化均值字典
    average_values = {
        "SSIM": 0,
        "PSNR": 0,
        "LPIPS": 0
    }

    # 累加每个文件中的指标值
    for parameter in parameters:
        for model, values in parameter.items():
            for key, value in values.items():
                average_values[key] += value

    # 计算均值
    for key, value in average_values.items():
        average_values[key] /= num_files

    return average_values

def process_json(json_file_paths):
    # 读取所有 JSON 文件中的参数
    all_parameters = []
    for file_path in json_file_paths:
        parameters = read_json_file(file_path)
        all_parameters.append(parameters)

    # 输出结果字典
    output_dict = {}

    # 输出每个 JSON 文件中存储的三个指标按行排列
    for parameters, file_path in zip(all_parameters, json_file_paths):
        directory_name = os.path.basename(os.path.dirname(file_path))
        model_output = {}
        for model, values in parameters.items():
            model_output[directory_name] = {}
            for key, value in values.items():
                model_output[directory_name][key] = value
        output_dict.update(model_output)

    # 计算均值
    average_values = calculate_average_gs(all_parameters)
    output_dict["Average"] = average_values
    return output_dict

# ---------------------------------------------------------------------------------------------------
def eval_log(message="0000000", file_path = "./0-log/log.txt"):
    # 将结果集合到path文件夹
    # log.txt 加行书写 实验名称 所使用的权重 mesh所在位置 gt所在位置 eval 结果 等信息
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 如果文件不存在，创建文件
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w"):
            pass
    with open(file_path, "a") as f:
        f.write(f"{timestamp}\n {message}\n")

def save_to_json(output_dict, output_file_path):
    # 将结果写入 JSON 文件
    with open(output_file_path, 'w') as file:
        json.dump(output_dict, file, indent=4)

def save_to_txt(output, file_path):
    # 检查目录是否存在，如果不存在则创建目录
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 保存输出结果到文本文件
    with open(file_path, 'w') as file:
        file.write(output)


def find_txt_files(directory):
    txt_files = []
    # 列出目录下的所有文件和文件夹
    for root, dirs, files in os.walk(directory):
        # 遍历文件
        for file in files:
            # 如果文件以 ".txt" 结尾，则将其路径加入到列表中
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    # print("===============================\n",txt_files)
    if txt_files:
        return txt_files[0]
    else:
        return

# 对一个实验名统计8个场景的平均指标

def auto_eval(args):
    exp_name = args.exp_postfix
    print(f" -------------------calc {exp_name}-------------------")

    sdf_dir = args.sdf_dir
    gs_dir = args.gs_dir
    results_dir = args.results_dir
    scenes = os.listdir(sdf_dir)

    # 几何指标
    geo_results_path = [] 
    # /home/xhd/xhd/0-output/neuris_data_sdf/neus/scene0085_00/scene0085_00-an1/meshes/eval_neus_thres0.05_2024-05-07_12-27.txt
    for scene in scenes:
        # scene_dir = os.path.join(sdf_dir, scene)
        eval_dir = os.path.join(sdf_dir, scene, scene + "-" + exp_name, "meshes")
        if os.path.exists(eval_dir):
            fscore_txt_path = find_txt_files(eval_dir)
            if fscore_txt_path:
                geo_results_path.append(fscore_txt_path)

    
    output = process_files(geo_results_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = os.path.join(results_dir, "eval_neus_all_{}-{}.txt".format(timestamp, exp_name))
    save_to_txt(output, file_path)
    file_path = os.path.join(sdf_dir+"/result", "eval_neus_all_{}-{}.txt".format(timestamp, exp_name))
    save_to_txt(output, file_path)

    print(geo_results_path)
    print(output)
        
    # 渲染指标
    render_results_path = []
    for scene in scenes:
        # /home/xhd/xhd/0-output/neuris_data_gs/scene0085_00-an1/results.json
        eval_dir = os.path.join(gs_dir, scene + "-" + exp_name)
        if os.path.exists(eval_dir):
            psnr_json = get_result_json(eval_dir)
            if psnr_json:
                render_results_path.append(psnr_json)
            

    
    output = process_json(render_results_path)      
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = os.path.join(results_dir, "eval_render_all_{}-{}.json".format(timestamp, exp_name))
    save_to_json(output, file_path)
    file_path = os.path.join(sdf_dir+"/result", "eval_render_all_{}-{}.json".format(timestamp, exp_name))
    save_to_json(output, file_path)
    
    print("render_results_path", render_results_path)
    print(output)



if __name__ == '__main__':
    parser = ArgumentParser("auto eval geometry and rendering")
    parser.add_argument("--sdf_dir", default="/home/xhd/xhd/0-output/neuris_data_sdf/neus")
    parser.add_argument("--gs_dir", default="/home/xhd/xhd/0-output/neuris_data_gs")
    parser.add_argument("--results_dir", default="./0-log")
    parser.add_argument("--exp_postfix", default="an1")
    args = parser.parse_args()

    auto_eval(args)




    


