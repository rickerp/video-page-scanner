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

- `VIDEO` - Video input file containing the paper (default: video.mp4)
- `OUTPUT` - Name of the output video file (default: output.mp4)
- `TEMPLATE` (optional) - Image file containing the wanted dimensions of output paper in the video (default dimensions: 1275x1650)

## Example

You can run the program with the example video file

https://user-images.githubusercontent.com/32230933/163783160-c08f97c1-c1b8-41ad-9ce2-a4831bea70d7.mp4

* Expected output

https://user-images.githubusercontent.com/32230933/163961770-19f13613-b66c-4c6b-a847-0e8a8e048898.mp4

### Image preview example

| Input | Output |
|---|---|
| <img src="https://user-images.githubusercontent.com/32230933/163954831-b4a34480-2eac-4617-8965-68e82da580bd.jpg" height="400px"> ) | <img src="https://user-images.githubusercontent.com/32230933/163955590-bbc62ae4-e624-4b85-9cfb-376d4d51db5b.jpg" height="400px"> |

## Others
* Original project statement: [statement.pdf](https://github.com/rickerp/video-page-scanner/files/8511163/statement.pdf)
