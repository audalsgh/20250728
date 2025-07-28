# 26일차

## 1. 로보플로우에서 라벨링하고 나온 모델 결과에 대한 평가 정리
<img width="340" height="474" alt="image" src="https://github.com/user-attachments/assets/4d50fb61-2439-4ae9-b10c-64ae39e88c20" />

```python
runs/
 └── detect/
      └── yolov11_custom/
           ├── train/       ← 학습용 이미지 및 라벨
           ├── valid/       ← 검증에 사용된 이미지 및 라벨
           ├── best.pt      ← 전이학습중, 가장 성능 좋았던 모델 (★ 가장 중요)
           ├── last.pt      ← 마지막 에폭에서 저장된 모델(가중치)
           └── data.yaml    ← 데이터셋의 클래스 이름, 경로, 갯수 등을 정의한 설정에대한 정보가 담긴 파일
```

| 용어             | 뜻                             |
| -------------- | ----------------------------- |
| **YOLO**       | 사진에서 사람·자동차·신호등 등 물체를 탐지하는 AI |
| **전이학습**       | 기존 모델을 내 데이터에 맞게 추가 훈련        |
| **.pt 파일**     | PyTorch 모델이 저장된 파일 (AI의 뇌)    |
| **yaml 파일**    | 데이터셋에 대한 설정 (클래스 수, 경로 등)     |
| **confidence** | 모델이 "이건 맞아!"라고 확신하는 정도 (0\~1) |

## 2. 다운받은 dataset을 구글 드라이브에 넣고, 코랩에서 YOLOv11 테스트해보기
Ultralytics YOLO는 PyTorch 모델이다. 기반이 파이토치라고 이해하기.<br>
런파드 도커 문제로 인해 수업은 더이상 하지않고, 코랩에서 dataset을 넣어 YOLOv11로 전이학습을 테스트해볼것.<br>
[코랩에서 실행한 코드들](0728_Transfer_Learning_Colab.ipynb)

1. 구글 드라이브에 dataset 폴더를 통쨰로 업로드하기.
  <img width="2560" height="970" alt="image" src="https://github.com/user-attachments/assets/9179e84f-431f-4a75-b8c2-c2840f4635ac" />

2. 코랩과 구글 드라이브 연동, mount해오기
  <img width="1148" height="508" alt="image" src="https://github.com/user-attachments/assets/d1ff1044-1e88-4a1e-b842-5d9371c816fd" />

3. dataset.yaml 야믈 파일내부에서 클래스가 몇개인지 알수있다.<br>
  데이터셋을 다운받은거라 원인은 모르겠으나, **경로를 dataset/vaild이 아니라 다른곳으로 잘못 참조하고 있음.**<br>
  <img width="197" height="202" alt="image" src="https://github.com/user-attachments/assets/b20b2952-65ee-497e-ae62-c4c8fd73ccd3" />

4. 경로상의 문제로, dataset.yaml 내부를 /content/dataset/dataset/train 등으로 수정해줘야한다.<br>
  train : 학습 이미지와 라벨이 들어 있는 경로<br>
  val:	검증 이미지와 라벨이 들어 있는 경로<br>
  names:	클래스 번호와 이름의 매핑이 보이고, 교수님이 주신 이 데이터셋은 두개의 클래스만 있다.
  <img width="1682" height="667" alt="image" src="https://github.com/user-attachments/assets/94288f88-83bd-48ac-b4ea-4e46f391517c" />

5. 수정후 재실행해보면, 3990장의 프레임이 분석되고, 오류없이 성능 평가까지 잘 마쳐진다.<br>
  -3990 프레임수 :	총 3990장의 영상 프레임을 처리했다는 의미. 영상 길이가 약 2분 = 프레임수 4000플마<br>
  -61.70초	: YOLO 모델이 전체 3990 프레임에 대해 추론하는 데 걸린 시간.<br>
  -FPS (Frame Per Second)	64.66 : (프레임수/처리시간)으로 구한 값, 초당 약 64.66장의 프레임을 YOLO가 처리했다는 뜻.<br>
   
  -> YOLO 모델이 꽤 빠르다는 뜻! 일반적으로 GPU (특히 Tesla T4 같은 모델)에서 60 FPS 이상이면 실시간 성능급임.<br>
  <img width="978" height="703" alt="image" src="https://github.com/user-attachments/assets/4373b72f-eee4-4e39-89a4-657d1266a6e9" />

7. 밑부분은 yaml 파일을 코랩의 코드셀에서 수정하는 부분이지만, 내가 직접 yaml 파일을 더블클릭하여 수정했으므로 의미는 없다.
