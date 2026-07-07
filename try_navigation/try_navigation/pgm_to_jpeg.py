import os
from PIL import Image

# 対象のフォルダパスを指定
folder_path = os.path.expanduser('~/ros2_ws/src/map/test5_waypoint_map')

# フォルダ内のpgmファイルを検索し、jpegに変換
for filename in os.listdir(folder_path):
    if filename.endswith(".pgm"):
        pgm_path = os.path.join(folder_path, filename)
        jpeg_path = os.path.join(folder_path, os.path.splitext(filename)[0] + ".jpeg")

        # 画像を開いてjpegに変換して保存
        with Image.open(pgm_path) as img:
            img.save(jpeg_path, "JPEG")
        print(f"Converted {pgm_path} to {jpeg_path}")

print("All .pgm files have been converted to .jpeg.")
