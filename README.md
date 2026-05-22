# Real-Time Bar Activity and Retention Analytics from a Public Livestream

### 1. Project overview

This project was developed for the assignment **Data Extraction from Livestreams** in the Postgraduate Program
for the cpurse IoT & Big Data.

The selected livestream is a public webcam of **Bondi Chaweng Samui** from Webcams24:

```text
https://webcams24.live/webcam/bondi-chaweng-samui-webcam
```
The project use:

```text
Python
OpenCV
YOLO object detection
CSV storage
SQLite storage
Time-series analysis
Event generation
```

### 2. Extracted information

The system extracts the following information from the livestream:

```text
Total people detected by the camera
People entering the bar
Estimated current bar occupancy
Bar retention percentage
Estimated bar entries per hour
Observed bar revenue
Projected daily bar revenue
Activity index
Vehicles detected
Bags detected
Events detected
```

## The most important metrics are:

Total people detected by camera

This represents the number of unique tracked people detected in the livestream during the current run.

Bar entries

A bar entry is counted when a detected person crosses the manually defined bar entrance line.

Bar retention

Bar retention measures how many detected people entered the bar.

```text
Bar retention (%) = bar entries / total people detected by camera × 100
```

Estimated bar occupancy

The system estimates how many people are currently inside the manually selected bar area. This is stored as a time-series measurement called:
```text
bar_seated_count
```

This value should be interpreted as an estimate of people sitting/stopping inside the bar area.

## Revenue estimation

The project assumes that each person entering the bar spends:
```text
7 EUR
```
Observed revenue:
```text
Observed revenue = total bar entries × 7 EUR
```

Projected daily revenue:
```text
Projected daily revenue = estimated entries per hour × business hours per day × 7 EUR
```
