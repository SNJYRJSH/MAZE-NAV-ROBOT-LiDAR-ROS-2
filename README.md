# MAZE NAV ROBO IN ROS-2 USING LiDAR AND FLOOD-FILL
A ROS-2 simulation of my maze navigation robot using LiDAR and a Flood-Fill Algorithm to solve any maze and navigate through it

**SOFTWARE USED**
- ROS-2
- PYTHON
- C++

**ARCHITECTURE**

<img width="1234" height="405" alt="Screenshot 2026-09-16 132208" src="https://github.com/user-attachments/assets/cf4b6757-97d4-4cb4-b679-7694e53548e7" />

**WORKING PRINCIPLE**

The robot incrementally constructs a grid-based representation of an initially unknown maze using LiDAR and wheel odometry. It then applies Flood Fill to the currently discovered map to find the optimal path. The robot then traverses the optimal path, using LiDAR to avoid walls until it reaches the exit.

**FLOOD - FILL VISUALIZATION**

<img width="1578" height="690" alt="Screenshot 2026-09-16 143444" src="https://github.com/user-attachments/assets/43faaa60-b785-4433-b214-ebd287f76760" />

**ROBOT**

<img width="919" height="553" alt="Screenshot 2026-09-16 145736" src="https://github.com/user-attachments/assets/7a3cfcc9-7e78-43aa-9536-6d79c1243261" />

**RESULTS**

https://www.youtube.com/watch?v=vaCZ9QVVZnY

**CHALLENGES**
- Custom wheel odometry and wheel integration along with calibration
- LiDAR object detection and avoidance
- Integration of Flood Fill with LiDAR and Odometry
- Robot Localization using Odometry and LiDAR



