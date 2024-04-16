from moviepy.editor import VideoFileClip

# Load the video file
video = VideoFileClip("resize/newvid.mp4")

# Resize the video
resized_video = video.resize(newsize=(512, 512))

# Write the resized video to a new file
resized_video.write_videofile("resize/resized_video.mp4")
