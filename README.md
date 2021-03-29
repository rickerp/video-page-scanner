# Video page scanner
Simple video page scanner that applies an homography in all video frames of the video and joins them.  
It receives a video containing a paper ([example](./video.mp4)) and outputs a video only with the paper changing along with the original video. 
This project was made for the Computer Vision course.  

## Running

Install dependecies
```python3
pip3 install -r requirements.txt
```

Run
```python3
python3 main.py -v VIDEO -o OUTPUT -t TEMPLATE
```
- VIDEO - Video input file containing the paper (default: video.mp4)
- OUTPUT - Name of the output video file (default: output.mp4)
- TEMPLATE (optional) - Image file containing the wanted dimensions of output paper in the video (default dimensions: 1275x1650)