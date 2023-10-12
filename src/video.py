import os

# img_path = "vis/graph-vis-2023-09-04-18-42-27-707743-dense/camera_1"
# frame_rate = 2
# height = 360
# width = 640
# out_path = "test.mp4"

# img_path = "vis/graph-vis-shirt/camera_1"
# frame_rate = 2
# height = 720
# width = 720
# out_path = "test-shirt.mp4"

# os.system(f"ffmpeg -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/*.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p {out_path}")

# img_path = "vis/postprocess-shirt"
# frame_rate = 1
# height = 360
# width = 640
# # out_path = "postprocess_orig.mp4"
# # out_path = "postprocess_gt.mp4"
# out_path = "postprocess_pred.mp4"
# 
# os.system(f"ffmpeg -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/*_1.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p {out_path}")

img_path = "vis/graph-vis-raw-2023-09-04-18-42-27-707743-dense/camera_1"
frame_rate = 15
height = 360
width = 640
out_path = "test-raw.mp4"

os.system(f"ffmpeg -r {frame_rate} -f image2 -s {width}x{height} -pattern_type glob -i '{img_path}/*.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p {out_path}")
