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


- Total people detected by camera

```text
This represents the number of unique tracked people detected in the livestream during the current run.
```
- Bar entries
```text
A bar entry is counted when a detected person crosses the manually defined bar entrance line.
```
- Bar retention
  
```text
Bar retention measures how many detected people entered the bar in relation to the number of people detected by the camera
```

```text
Bar retention (%) = bar entries / total people detected by camera × 100
```

- Estimated bar occupancy

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
### 3. Why this information is useful

This data could be useful for:
```text
bar owners
restaurant managers
tourism analysts
marketing teams
event organizers
smart-city analysts
```
The system can answer questions such as:
```text
How many people pass through the camera view?
How many people enter the bar?
What is the conversion/retention rate?
When is the bar busiest?
How many people are estimated to be inside the bar?
What is the estimated revenue based on visible customer flow?
Are there periods with high traffic but low bar entry?
```

This is more useful than simple object detection because it converts raw video into business intelligence, and can be used to estimate the revenue, very important in geographic area where the usage of cash payment is sitll the highsest percentage, and can help the bar owners to monitor the trustfullness for the employess. Understand if possible attractio  can increase the rate of retation in the bar and so the cashflow.
Possible correlation with the crowdness of the area with the bar entrance

### 4. Big Data and IoT context

The livestream acts as an IoT data source.

```text
The pipeline is:

Public livestream
    ↓
OpenCV frame capture
    ↓
YOLO object detection
    ↓
Tracking and region-of-interest analytics
    ↓
Event generation
    ↓
CSV and SQLite storage
    ↓
Analysis and dashboard/reporting
```

In a larger Big Data system, this application could be scaled to many public webcams.

A possible scalable architecture would be:

Multiple livestreams
    ↓
Edge processing with Python/OpenCV
    ↓
MQTT or Kafka
    ↓
Cloud database or data lake
    ↓
Spark/Flink analytics
    ↓
Grafana, Power BI, or Streamlit dashboard

Instead of storing large raw video files, the system stores compact structured records such as counts, events, timestamps, and calculated metrics.
