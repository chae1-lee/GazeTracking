# Data Analysis
## MPII Gaze
>https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild

구조: Data - Original<br>
　　　　　- Normalized<br>
+) 6 points-based face model

**Orginal**: face detector와 facial landmark detector로 잘린 눈 이미지<br>
　　　　참가자(15명)별 폴더가 있으며 그 안에 날짜별로 annotation.txt와 이미지가 들어있음<br>

　　　　　　annotation.txt:<br>
　　　　　　- Dimension 01-24: 감지된 눈의 landmark 좌표<br>
　　　　　　- Dimension 25-26: 화면상의 gaze target position<br>
　　　　　　- Dimension 27-29: 3D gaze target position related to camera<br>
　　　　　　- Dimension 30-35: 3D 얼굴 모델로 추정한 3D haed 6-points (4 eye corner points, 2 mouth corner points)<br>
　　　　　　- Dimension 36-38: 카메라로 추정된 3D 오른쪽 눈 중심 좌표<br>
　　　　　　- Dimension 39-41: 카메라로 추정된 3D 왼쪽 눈 중심 좌표<br>

　　　　　+) 각 참가자마다 Calibration 폴더도 있음:<br>
　　　　　　Camera.mat: 노트북 카메라 고유 매개변수.<br>
　　　　　　　- "cameraMatrix": 카메라 projection 행렬.<br>
　　　　　　　- "distCoeffs": 카메라 왜곡 계수.<br>
　　　　　　　- "retval": re-projection error(RMS, root mean square).<br>
　　　　　　　- "rvecs": the rotation vectors.<br>
　　　　　　　- "tvecs": the translation vectors.<br>
　　　　　　monitorPose.mat: 카메라 좌표에서 이미지 평면의 위치<br>
　　　　　　　- "rvecs": the rotation vectors.<br>
　　　　　　　- "tvecs": the translation vectors.<br>
　　　　　　creenSize.mat: 노트북 화면 크기<br>
　　　　　　　- "height_pixel": 화면 높이(pixel)<br>
　　　　　　　- "width_pixel": 화면 너비(pixel)<br>
　　　　　　　- "height_mm": 화면 높이(mm)<br>
　　　　　　　- "width_mm": 화면 너비(mm)<br>

**Normalized**: Sugano et al.[3]에서 원근 변환을 통해 크기 조정 및 회전을 취소한 정규화 후 안대 이미지<br>
　　　　　　Original 폴더와 마찬가지로 각 참가자마다 날짜별로 정리되어 있으며, file format은 ".mat"<br>

　　　　　　annotation.txt:<br>
　　　　　　　- 3D gaze head pose<br>
　　　　　　　- 3D gaze direction<br>
　　　　　　　- 논문에서 제안한 3D gaze directon -> **2D gaze target** 생성<br>

**Evaluation subset**: 이미지 목록

**Annotation subset**:<br>
　- 10,848개의 이미지<br>
　- (x, y) position of 6 facial landmarks (four eye corners, two mouth corners)<br>
　- (x, y) position of two pupil centers(왼쪽 눈, 오른쪽 눈)

* 잘린 눈 이미지 크기 (720 X 1280 px) -> Normalized eye patch image (36 X 60 px)

## Gaze360
>http://gaze360.csail.mit.edu/

- 실내 5곳(53명)과 실외 2곳(185명)에서 총 9회에 걸쳐 238명의 피험자를 수집
- 총 129K 훈련, 17K 검증 및 시선 주석이 있는 26K 테스트 이미지를 획득 
- 육안 검사를 통해 피험자의 연령, 민족 및 성별(58% 여성, 42% 남성)
