import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Temporal Collage Prompting - 驾驶事故视频识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py extract-frames --input data/videos --output data/data-frames/data-frames-3fps --interval 10
  python main.py create-collage --input data/data-frames/data-frames-3fps --output data/collages/collages-3fps-2-3 --layout 2-3
  python main.py analyze --input data/collages/collages-3fps-2-3 --model gpt-4o-low
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 视频帧提取命令
    extract_parser = subparsers.add_parser('extract-frames', help='从视频中提取帧')
    extract_parser.add_argument('--input', required=True, help='视频文件所在目录')
    extract_parser.add_argument('--output', required=True, help='帧保存目录')
    extract_parser.add_argument('--interval', type=int, default=10, help='帧提取间隔 (默认: 10)')
    
    # Collage生成命令
    collage_parser = subparsers.add_parser('create-collage', help='生成Collage图片')
    collage_parser.add_argument('--input', required=True, help='帧文件所在目录')
    collage_parser.add_argument('--output', required=True, help='Collage保存目录')
    collage_parser.add_argument('--layout', choices=['2-2', '2-3', '3-2'], default='2-3', help='Collage布局 (默认: 2-3)')
    
    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='使用GPT-4o分析视频')
    analyze_parser.add_argument('--input', required=True, help='Collage目录')
    analyze_parser.add_argument('--model', choices=['gpt-4o-low', 'gpt-4o-high'], default='gpt-4o-low', help='模型类型')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # 根据命令执行相应功能
    if args.command == 'extract-frames':
        extract_frames(args.input, args.output, args.interval)
    elif args.command == 'create-collage':
        create_collage(args.input, args.output, args.layout)
    elif args.command == 'analyze':
        analyze_collage(args.input, args.model)

def extract_frames(input_dir, output_dir, interval):
    """从视频中提取帧"""
    import cv2
    
    def get_total_frames(video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames
    
    def extract_single_video(video_path, output_dir, frame_interval=1):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        total_frames = get_total_frames(video_path)
        padding_length = len(str(total_frames))
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"无法打开视频文件: {video_path}")
        
        frame_count = 0
        saved_frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:0{padding_length}d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"已保存 {saved_frame_count} 帧到 '{output_dir}'")
    
    mp4_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    
    for video_path in mp4_files:
        relative_path = os.path.relpath(video_path, input_dir)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(output_dir, os.path.dirname(relative_path), video_name)
        extract_single_video(video_path, out_dir, interval)

def create_collage(input_dir, output_dir, layout):
    """生成Collage图片"""
    import cv2
    import numpy as np
    
    rows, cols = map(int, layout.split('-'))
    frames_per_collage = rows * cols
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for subfolder in os.listdir(input_dir):
        subfolder_path = os.path.join(input_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for inner_subfolder in os.listdir(subfolder_path):
                inner_subfolder_path = os.path.join(subfolder_path, inner_subfolder)
                if os.path.isdir(inner_subfolder_path):
                    frame_files = sorted(os.listdir(inner_subfolder_path))
                    num_collages = len(frame_files) // frames_per_collage + (1 if len(frame_files) % frames_per_collage > 0 else 0)
                    padding_width = len(str(num_collages))
                    
                    for collage_index in range(num_collages):
                        frames = []
                        for i in range(frames_per_collage):
                            if collage_index * frames_per_collage + i < len(frame_files):
                                frame_path = os.path.join(inner_subfolder_path, frame_files[collage_index * frames_per_collage + i])
                                frame = cv2.imread(frame_path)
                                frames.append(frame)
                        
                        if frames:
                            frame_height, frame_width = frames[0].shape[:2]
                            padding = 15
                            collage_height = rows * frame_height + (rows + 1) * padding
                            collage_width = cols * frame_width + (cols + 1) * padding
                            
                            collage = np.full((collage_height, collage_width, 3), 255, dtype=np.uint8)
                            
                            positions = []
                            for r in range(rows):
                                for c in range(cols):
                                    y = (r + 1) * padding + r * frame_height
                                    x = (c + 1) * padding + c * frame_width
                                    positions.append((y, x))
                            
                            for i, (frame, pos) in enumerate(zip(frames, positions)):
                                y, x = pos
                                collage[y:y+frame_height, x:x+frame_width] = frame
                                cv2.rectangle(collage, (x, y), (x + 80, y + 50), (0, 0, 0), -1)
                                cv2.putText(collage, f'{collage_index * frames_per_collage + i + 1}',
                                            (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            output_subfolder = os.path.join(output_dir, subfolder, inner_subfolder)
                            os.makedirs(output_subfolder, exist_ok=True)
                            output_path = os.path.join(output_subfolder, f'collage_{collage_index + 1:0{padding_width}d}.jpg')
                            cv2.imwrite(output_path, collage)
                            print(f"已保存: {output_path}")

def analyze_collage(input_dir, model_type):
    """使用GPT-4o分析Collage"""
    import time
    from datetime import datetime
    from openai import OpenAI
    from dotenv import load_dotenv
    from pathlib import Path
    import cv2
    import base64
    import json
    from sklearn.metrics import confusion_matrix, classification_report
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("错误: 未找到API密钥，请在.env文件中设置OPENAI_API_KEY")
        sys.exit(1)
    
    MODEL = "gpt-4o-2024-08-06" if model_type == 'gpt-4o-high' else "gpt-4o-2024-08-06"
    img_detail = "high" if model_type == 'gpt-4o-high' else "low"
    client = OpenAI(api_key=api_key)
    
    directories = {
        "norm": os.path.join(input_dir, "norm"),
        "ped": os.path.join(input_dir, "ped"),
        "col": os.path.join(input_dir, "col")
    }
    
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{MODEL}_{input_dir}_{img_detail}_log_{datetime.now():%Y-%m-%d-%H-%M}.log"
    
    def process_frames_from_directory(directory_path):
        base64Frames = []
        for root, _, files in os.walk(directory_path):
            for file in sorted(files):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    frame_path = os.path.join(root, file)
                    frame = cv2.imread(frame_path)
                    _, buffer = cv2.imencode(".jpg", frame)
                    if buffer.nbytes > 20 * 1024 * 1024:
                        print(f"警告: 帧尺寸超过20MB，跳过: {frame_path}")
                        continue
                    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        return base64Frames
    
    QUESTION = """Your task is to first identify whether an accident occurs in the video. You need to classify it as either "Normal" or "Accident". If it's "Normal", you don't need to take any action. However, if it's an "Accident", please also specify the type of accident with the reason in detail. There are only two types of accidents: Type A: a car crashes into people who are crossing the street. Type B: a car crashes with another vehicle. Let's think step-by-step"""
    
    def chat_with_gpt(base64Frames):
        retry_attempts = 5
        for attempt in range(retry_attempts):
            try:
                messages = [
                    {"role": "system", "content": """Use the video to answer the provided question. Respond in JSON with attributes: Class: {Normal|Accident}, Accident_type: {A|B|None}."""},
                    {"role":"user","content":[
                        "These are the frames from the video.",
                        *map(lambda x:{"type":"image_url","image_url":{"url":f'data:image/jpg;base64,{x}',"detail":f"{img_detail}"}}, base64Frames),
                        QUESTION
                    ]}
                ]
                
                qa_visual_response = client.chat.completions.create(model=MODEL, messages=messages)
                response = qa_visual_response.choices[0].message.content
                input_tokens = qa_visual_response.usage.prompt_tokens
                output_tokens = qa_visual_response.usage.completion_tokens
                
                return extract_response(response), input_tokens, output_tokens
            except Exception as e:
                print(f"API调用错误: {e}")
                if attempt < retry_attempts - 1:
                    print(f"重试中... ({attempt + 1}/{retry_attempts})")
                    time.sleep(2 ** attempt)
                else:
                    print("达到最大重试次数，跳过此视频")
                    return None, 0, 0
    
    def extract_response(response):
        response = response.lower()
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            start_idx = response.find("```json")
            if start_idx != -1:
                start_idx += 7
                end_idx = response.rfind("```")
                json_str = response[start_idx:end_idx].strip()
                return json.loads(json_str)
            else:
                raise ValueError("响应不包含有效的JSON")
    
    def get_label_index(label):
        return {"norm": 0, "ped": 1, "col": 2}[label]
    
    true_labels = []
    pred_labels = []
    prompt_tokens = 0
    completion_tokens = 0
    
    for label, dir_path in directories.items():
        print(f"处理标签: {label}")
        subfolders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        for subfolder in subfolders:
            print(f"处理子文件夹: {subfolder}")
            base64Frames = process_frames_from_directory(subfolder)
            if not base64Frames:
                print(f"警告: 未找到有效帧: {subfolder}")
                continue
            response, in_tokens, out_tokens = chat_with_gpt(base64Frames)
            if response is None:
                print(f"警告: 未收到有效响应: {subfolder}")
                continue
            
            prompt_tokens += in_tokens
            completion_tokens += out_tokens
            
            true_labels.append(get_label_index(label))
            if response["class"] == "normal":
                pred_labels.append(get_label_index("norm"))
            else:
                if response["accident_type"] == "a":
                    pred_labels.append(get_label_index("ped"))
                elif response["accident_type"] == "b":
                    pred_labels.append(get_label_index("col"))
    
    if true_labels and pred_labels:
        conf_matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])
        class_report = classification_report(true_labels, pred_labels, target_names=["Normal", "Pedestrian Accident", "Collision"])
        
        print("\n混淆矩阵:")
        print(conf_matrix)
        print("\n分类报告:")
        print(class_report)
        print(f"\n输入Token: {prompt_tokens}")
        print(f"输出Token: {completion_tokens}")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"混淆矩阵:\n{conf_matrix}\n\n分类报告:\n{class_report}\n\n输入Token: {prompt_tokens}\n输出Token: {completion_tokens}")
    else:
        print("错误: 没有预测结果或真实标签")

if __name__ == "__main__":
    main()