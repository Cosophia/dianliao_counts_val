import os
import json


def remove_duplicate_points(points):
    """
    删除 polygon 中的重复点
    """
    new_pts = []
    seen = set()

    for p in points:
        pt = (round(p[0], 4), round(p[1], 4))

        # 去掉连续重复和全局重复
        if pt not in seen:
            new_pts.append([p[0], p[1]])
            seen.add(pt)

    return new_pts


def process_json(json_path, save_path):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])

    new_shapes = []

    for shape in shapes:

        if shape.get("shape_type") == "polygon":

            pts = shape.get("points", [])

            pts_clean = remove_duplicate_points(pts)

            # polygon 至少需要3个点
            if len(pts_clean) >= 3:
                shape["points"] = pts_clean
                new_shapes.append(shape)

        else:
            new_shapes.append(shape)

    data["shapes"] = new_shapes

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def batch_process(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for name in os.listdir(input_dir):

        if not name.endswith(".json"):
            continue

        in_path = os.path.join(input_dir, name)
        out_path = os.path.join(output_dir, name)

        process_json(in_path, out_path)

        print("processed:", name)


if __name__ == "__main__":

    input_dir = r"D:\ZX_AI_TRAIN\semseg\HD_Bubble_Det\data\labels"
    output_dir = r"D:\ZX_AI_TRAIN\semseg\HD_Bubble_Det\data\labels"

    batch_process(input_dir, output_dir)