# MedSwift: Autonomous Drone-Based Medical Delivery System

## 1. 프로젝트 개요 및 목표📋
### 개요
MedSwift는 GPS 신호가 없는 환경에서도 비전 기반 내비게이션을 활용하여 응급 의료 상황을 지원하기 위한 자율 드론 기반 의료용품 배달 시스템입니다.
기존 응급 물류 시스템의 한계를 극복하고, 빠른 시간 내에 의료 물품을 정확히 배달하는 것을 목표로 합니다.
### 목표
의료물품 긴급 배달: GPS가 약한 재난 지역, 터널, 실내 등에서 활용 가능.

## 2. 문제 정의  🚩

### 1. 기존 문제점
  응급 상황에서의 물류 지연
  GPS 의존성 문제


### 2. 해결 목표
  GPS 없이도 안정적인 비전 기반 경로 설정
  장애물 회피 및 정밀 착륙을 통한 의료물품 배달


## 3. 주요 기능 🚀

### 1. 출발지 마커 인식
![image](https://github.com/user-attachments/assets/b73d9c3a-c8b3-400d-9592-236309b4ac55)
ArUco 마커를 통해 출발지를 인식하고 비행 시작.

### 2. 경로 추적 (Way Point Navigation)
![image](https://github.com/user-attachments/assets/9e2df1ae-840e-4707-9631-07c85b14547f)
![image](https://github.com/user-attachments/assets/c0862062-602e-4ac2-ad9b-c9e3bd11c1af)

사전에 정의된 웨이포인트를 따라 이동하며 비전 기반 내비게이션을 수행.

### 3. 장애물 회피
장애물 탐지 및 경로 수정 알고리즘 적용.

### 4. 도착지 마커 인식 및 정밀 착륙
ArUco 마커를 활용해 정확한 착륙 지점에 의료용품 배달.


## 4. 시스템 아키텍처 🛠️

### 하드웨어
  DJI TELLO 드론
  소형 드론으로 시뮬레이션 및 비행 테스트 진행.

### 소프트웨어 스택
  Python: 메인 코드 및 알고리즘 구현
  OpenCV: 비전 기반 장애물 탐지 및 마커 인식
  ArUco Marker: 착륙 지점 인식
  Tello SDK: 드론 제어
Footer
© 2024 
