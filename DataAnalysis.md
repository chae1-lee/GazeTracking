# Data Analysis
## 01. MPII Gaze
> https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild

구조: Data - Original<br>
　　　　　- Normalized<br>
+) 6 points-based face model

### Original 폴더
face detector와 facial landmark detector로 잘린 눈 이미지, 참가자(15명)별 폴더가 있으며 그 안에 날짜별로 annotation.txt와 이미지가 들어있음<br>

- annotation.txt
  - Dimension 01-24: 감지된 눈의 landmark 좌표
  - Dimension 25-26: 화면상의 gaze target position
  - Dimension 27-29: 3D gaze target position related to camera
  - Dimension 30-35: 3D 얼굴 모델로 추정한 3D haed 6-points (4 eye corner points, 2 mouth corner points)
  - Dimension 36-38: 카메라로 추정된 3D 오른쪽 눈 중심 좌표
  - Dimension 39-41: 카메라로 추정된 3D 왼쪽 눈 중심 좌표

  - 각 참가자마다 Calibration 폴더도 있음:
    - Camera.mat: 노트북 카메라 고유 매개변수.
      - "cameraMatrix": 카메라 projection 행렬.
      - "distCoeffs": 카메라 왜곡 계수.
      - "retval": re-projection error(RMS, root mean square).
      - "rvecs": the rotation vectors.
      - "tvecs": the translation vectors.
    - monitorPose.mat: 카메라 좌표에서 이미지 평면의 위치
      - "rvecs": the rotation vectors.
      - "tvecs": the translation vectors.
    - creenSize.mat: 노트북 화면 크기
      - "height_pixel": 화면 높이(pixel)
      - "width_pixel": 화면 너비(pixel)
      - "height_mm": 화면 높이(mm)
      - "width_mm": 화면 너비(mm)

### Normalized 폴더
Sugano et al.[3]에서 원근 변환을 통해 크기 조정 및 회전을 취소한 정규화 후 안대 이미지, Original 폴더와 마찬가지로 각 참가자마다 날짜별로 정리되어 있으며, file format은 ".mat"<br>

- annotation.txt:
  - 3D gaze head pose
  - 3D gaze direction
  - 논문에서 제안한 3D gaze directon -> **2D gaze target** 생성

### Evaluation subset 폴더
이미지 목록

### Annotation subset 폴더
- 10,848개의 이미지<br>
- (x, y) position of 6 facial landmarks (four eye corners, two mouth corners)<br>
- (x, y) position of two pupil centers(왼쪽 눈, 오른쪽 눈)

###### 잘린 눈 이미지 크기 (720 X 1280 px) -> Normalized eye patch image (36 X 60 px)

## 02. Gaze360
> http://gaze360.csail.mit.edu/

- 실내 5곳(53명)과 실외 2곳(185명)에서 총 9회에 걸쳐 238명의 피험자를 수집 (총 1975588 frame)
- 총 129K 훈련, 17K 검증 및 시선 주석이 있는 26K 테스트 이미지를 획득 
- 육안 검사를 통해 피험자의 연령, 민족 및 성별(58% 여성, 42% 남성) 구분
- Ladybug5 360 degree camera 사용, 5+1 vertical sensor 장착
- 원본 이미지 크기 (2048 X 2448) -> 수정 후 이미지 크기 (3382 X 4096)
- 카메라가 겹쳐서 똑같은 장면이 두 장 생겨도 개별된 샘플로 저장

### 구조
The dataset consists of
- read me
- license
- ```metadata.mat``` with annotations
- 머리와 몸 부분이 잘린 JPEG images

## 03. Gaze Capture
> https://gazecapture.csail.mit.edu/

### json 설명:
- appleFace.json, appleLeftEye.json, appleRightEye.json
  - left: 이미지의 오른쪽에 나타나는 피사체의 물리적인 왼쪽
  - X, Y: bounding box의 왼쪽 상단 x, y 좌표 (pixel 단위)
  - W, H: bounding box의 너비와 높이 (pixel 단위)
  - IsValid: 실제 감지 여부, 1=감지, 0=감지되지 않음

- dotInfo.json
  - DotNum: 해당 프레임 동안 표시되는 점(0부터 시작)의 sequence 번호
  - XPts, YPts: 화면 왼쪽 상단 점의 중심 위치 (포인트 단위, 이 단위는 screen.json 참조)
  - XCam, YCam: 예측 공간에서 점의 중심 위치. 위치는 cm로 측정
  - Time: 표시된 점이 화면에 나타난 경과 시간(초)

- faceGrid.json: Apple face detection에서 생성된 "face grid"의 input feature. 25X25 grid
  - X, Y: 왼쪽 상단 x, y 좌표
  - W, H: face box의 너비와 높이
  - IsValid: apple*.json의 IsValid 값과 동일

- frames.json: frames 폴더에 있는 frame의 파일 이름. (info.json 참조)

- info.json
  - TotalFrames: 해당 주제의 총 프레임 수
  - NumFaceDetections: 얼굴이 감지된 프레임 수
  - NumEyeDetections: 눈이 감지된 프레임 수
  - Dataset: "train", "val", or "test"
  - DeviceName: 녹음에 사용된 장치의 이름

- motion.json: 프레임이 녹화되는 동안 60Hz로 기록된 모션 데이터 스트림

- screen.json
  - H, W: 앱의 활성 화면 영역의 높이와 너비
  - Orientation: The orientation of the interface, as described by the enumeration UIInterfaceOrientation, where:
    1. portrait
    2. portrait, upside down (iPad only)
    3. landscape, with home button on the right
    4. landscape, with home button on the left
