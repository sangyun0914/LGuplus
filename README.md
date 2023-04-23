# Real-time AI Fitness Service
## 2022 LG U+ Corporate Collaboration Project  

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PyTorch](https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Flask-000000.svg?style=for-the-badge&logo=Flask&logoColor=white)

---
## Project Overview![](https://raw.githubusercontent.com/aregtech/areg-sdk/master/docs/img/pin.svg)
This repository contains the source code for the project 'Real-time action recognition-based AI fitness service'. This computer vision project recognizes the user's exercise and provides reps count and feedback, helping users to exercise in the right posture. 

<img src="https://user-images.githubusercontent.com/96368116/233837621-b0921165-f964-42af-9ac0-1cff89c68c0f.png">

---
## Real-time Exercise Recognition 💻
The pipeline for the project is in two parts: Skeleton Recognition and Exercise Recognition. 
- Skeleton Recognition: Acts as a preprocessing stage for extracting human joints, which we have used  [Mediapipe](https://google.github.io/mediapipe).
- Exercise Recognition: Uses given human joints to recognize the user's exercise. We have approached this task using GRU, LSTM, MHI, and Transformers to predict the given time-series joint features. Detailed code and explanation is given at main_project folder.

---
## Contributors 🙌 
- 유재준 (성균관대학교 소프트웨어학과 18학번)
- 김동빈 (성균관대학교 소프트웨어학과 20학번)
- 신상윤 (성균관대학교 소프트웨어학과 21학번)
- 안준성 (성균관대학교 소프트웨어학과 21학번)
- 채시헌 (성균관대학교 소프트웨어학과 21학번)

