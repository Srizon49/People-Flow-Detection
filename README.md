# People Flow Detection using Object Tracking & Heatmap Visualization

## Project Overview

This project implements a people flow detection system using YOLOv8 for human detection and ByteTrack for object tracking. The system detects people in a video, assigns unique IDs to each individual, tracks their movement across frames, and generates a heatmap showing areas with the highest activity.

The project demonstrates the practical application of computer vision techniques for crowd analysis, surveillance systems, pedestrian monitoring, and movement analytics.

## Features

- Human Detection using YOLOv8
- Object Tracking using ByteTrack
- Unique ID Assignment for each detected person
- Bounding Box Visualization
- Heatmap Generation
- Processed Video Output
- Real-time Person Tracking

## Technologies Used

- Python
- YOLOv8
- ByteTrack
- OpenCV
- NumPy
- Supervision

## Project Workflow

1. Input video is processed frame by frame.
2. YOLOv8 detects all people in the scene.
3. ByteTrack maintains unique identities across frames.
4. Bounding boxes and IDs are displayed.
5. Object positions are collected for heatmap generation.
6. A final heatmap is created showing movement density.

## Input

- people-walking.mp4

## Output

- tracked_people_flow.mp4
- people_heatmap.png

## Applications

- Crowd Monitoring
- Smart Surveillance
- Traffic Flow Analysis
- Pedestrian Analytics
- Public Space Monitoring

## Author

Ashraful Haque Srizon
