import json
import shutil
import os
from pathlib import Path


def copy_ng_files_keep_original(json_path, source_img_dir, output_dir):
    """
    复制包含 NG 标签的图像和 JSON（保留原始所有标签）
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查是否有 NG 标签
    has_ng = any(s['label'] == 'NG' for s in data['shapes'])

    if has_ng:
        ng_img_dir = Path(output_dir) / "images"
        ng_json_dir = Path(output_dir) / "json"
        ng_img_dir.mkdir(parents=True, exist_ok=True)
        ng_json_dir.mkdir(parents=True, exist_ok=True)

        # 复制图像
        img_name = data['imagePath']
        img_path = Path(source_img_dir) / img_name
        if img_path.exists():
            shutil.move(img_path, ng_img_dir / img_name)

        # 复制原始 JSON（不改内容）
        shutil.copy2(json_path, ng_json_dir / json_path.name)

        return True
    return False




# 批量处理
input_dir = r"D:\20260126_byt_汇总\images"
output_dir = r"C:\Users\Administrator\Desktop\ng_byt"
os.makedirs(output_dir, exist_ok=True)

for json_file in Path(input_dir).glob("*.json"):
    copy_ng_files_keep_original(json_file, input_dir, output_dir)
    # if img_path:
    #     print(f"✓ {json_file.name} -> 图片: {img_path}, NG标签: {count}")