# Fish_Detection_Project
어종 식별 및 분류 알고리즘 개발을 주제로 인공지능팩토리에서 주관한 K-water AI 경진대회에 참가했다. 개인으로 진행했고, EDA 및 데이터 전처리, 데이터 증강, 모델링 등의 역할을 수행했다. 참가한 69팀 중 최종 순위 18위를 기록했다.

[프로젝트 내용 설명 영상](https://drive.google.com/file/d/1Ia6KdAJUfGKAQpYGjnAknZ9I6ZjGwWeq/view?usp=drive_link)

## 프로젝트 개요
- 인공지능팩토리에서 주관한 2023 제 3회 K-water AI 경진대회로 낙동강에 서식하는 물고기들의 어종을 분류하고 탐지하는 모델을 개발
- 약 2주동안 진행했으며 농어, 베스, 숭어, 강준치, 블루길, 잉어, 붕어, 누치 8개의 클래스에 대해 객체 탐지를 수행

<br>
<p align="center">

<img width="826" alt="image" src="https://github.com/Hyeonseung0103/Fish_Detection_Project/assets/97672187/3ad7a15e-1b35-4d33-8680-9b826bff3a6f">

<img width="727" alt="image" src="https://github.com/Hyeonseung0103/Fish_Detection_Project/assets/97672187/be4e5b26-8899-4a8f-9bb2-aba05c76fa03">


<br>

- 목표: 최종 순위 상위 30% 이내


## 데이터 설명
- 대회에서 제공한 데이터(학습 약 100,000개 / 테스트 약 45,000개)
- 학습에 사용한 데이터셋
  - Sample_v1: 학습 9,500개(객체 4,500 / 배경 500), 검증 1,000개(객체 500 / 배경 500), 데이터 증강 X
  - Sample_v2: 학습 18,000개(13,000 / 5,000), 검증 2,000개(1,500 / 500), 데이터 증강 O
  - Sample_v3: 학습 22,000개(17,000 / 5,000), 검증 2,500개(2,000 / 500), 데이터 증강 O
  - Sample_eq: 학습 26,000개(21,000 / 5,000), 검증 2,800개(2,300 / 500), 데이터 증강 O
  - 데이터 증강에는 blur, gaussian noise, clahe, gray, vertical/horizontal flip, resize & crop, hsv, brightness 변환 기법을 사용했다.

- 원본 이미지(왼쪽)와 horizontal flip(오른쪽) 증강이 적용된 이미지 예시
<p align = "center">
<img width="49%" alt="image" src="https://github.com/Hyeonseung0103/Fish_Detection_Project/assets/97672187/078d3614-291b-420f-ab8f-f73be377b7c1">
<img width="49%" alt="image" src="https://github.com/Hyeonseung0103/Fish_Detection_Project/assets/97672187/7049aa73-8754-4cbd-8489-3a8336fbec04">
</p>

<br>

- Blur(왼쪽)와 Clahe(오른쪽) 증강이 적용된 이미지 예시

<p align = "center">
<img width="49%" alt="image" src="https://github.com/Hyeonseung0103/Fish_Detection_Project/assets/97672187/49010a57-c8d9-437d-931e-ef5d8424d8c6">
<img width="49%" alt="image" src="https://github.com/Hyeonseung0103/Fish_Detection_Project/assets/97672187/a3e11f95-2fe9-4779-a848-8a33e176e514">
</p>

## 모델링
YOLOv8, YOLO-NAS, Faster R-CNN 모델을 사용했고 평가지표는 mAP50과 대회 평가지표인 F1-score를 사용했다.

### 1. 사용한 모델
1. YOLOv8
  - 2023년 1월에 출시된 ultralytics의 최신 YOLO 모델
  - 트랜스포머를 기반으로 객체 탐지 뿐만 아니라 분류, 세그멘테이션 등의 다양한 태스크를 수행할 수 있는 통합 프레임워크
  - 간단한 코드만으로도 학습과 추론이 가능하면서 이전버전의 YOLO 시리즈 보다 준수한 성능
  - 모델의 크기에 따라 n,s,m,l로 나눌 수 있고 본 프로젝트를 진행하며 모든 크기의 모델을 다 사용
   
2. YOLO-NAS
  - 2023년 5월에 출시됐고 Deci에서 개발한 YOLO의 가장 최신 모델
  - 이전 YOLO 시리즈와는 달리 quantization을 사용하여 정확도와 연산 속도를 개선시킴
  - 학습 중에 quantization block을 사용하여 quantization performance 향상
  - Post-training quantization으로 학습 후에 파라미터를 INT8 포멧으로 변환
  - 본 프로젝트에서 s,m,l 모두 사용

3. Faster R-CNN
- 다양한 실험을 위해 two stage 모델이면서 R-CNN 계열의 모델들 중 준수한 성능을 가지고 있는 Faster R-CNN 사용
- 이전 버전인 Fast R-CNN, R-CNN과는 달리 selective search 알고리즘 대신 anchor box를 사용하여 하나의 통일된 네트워크로 region proposals과 탐지를 수행
- Backbone으로는 ResNetX101에 FPN이 적용된 네트워크를 사용   

### 2. 성능
1. YOLOv8-nano
- Sample v1, v2, v3, eq 모든 데이터셋을 다 사용하고 n,s,m,l 모든 아키텍처를 다 사용했지만 sample_v2 데이터셋과 nano 모델을 사용했을 때 test F1-score가 0.635(confidence threshold 0.75)로 가장 높았음
- Sample_v1 dataset: test F1-score 0.529(에포크 20)
- Sample_v2 dataset: test F1-score 0.635(에포크 60 및 하이퍼파라미터 튜닝)
- Sample_v3 dataset: test F1-score 0.616(에포크 100 및 하이퍼파라미터 튜닝)
- Sample_eq dataset: test F1-score 0.622(에포크 120 및 하이퍼파라미터 튜닝)

2. YOLO-NAS
- sample_v2, v3 데이터셋을 사용하고 s,m,l 모든 아키텍처를 다 사용했지만 YOLOv8-nano 모델보다 성능이 많이 떨어져서 더 이상 실험을 진행하지 않음
- YOLO-NAS-S: test F1-score 0.567(에포크 40 및 하이퍼파라미터 튜닝)
- YOLO-NAS-M: test F1-score 0.42(에포크 20)
- YOLO-NAS-L: test F1-score 0.529(에포크 35 및 하이퍼파라미터 튜닝)

3. Faster R-CNN
- Sample_eq 데이터셋으로 학습했고 오랜 시간 학습했지만 학습 시간 대비 성능이 저조해서 더 이상 실험을 진행하지 않음
- Sample_eq dataset: test F1-score 0.507(에포크 25 및 하이퍼파라미터 튜닝)


### 3. Weighted Boxes Fusion
- 단일 모델로 예측을 하는 것보다 성능이 좋은 모델들을 결합하여 하나의 결과를 도출하는 것이 괜찮은 아이디어라고 판단
- 성능이 가장 좋았던 YOLOv8-nano 모델들 중 증강이 적용된 데이터셋(v2, v3, eq)에서 가장 좋은 test F1-score를 가진 모델을 하나 혹은 두개 선별하여 총 4개의 모델 사용
- Sample_v2(0.635, 0.625), Sample_v3(0.616), Sample_eq(0.622)
  - WBF weights = [2,2,1,1], iou_thr = 0.6, skip_box_thr = 0.81
  - 4개의 모델을 사용하여 WBF를 적용한 결과 Final test F1-score 0.637을 기록하며 성능 향상

## 한계점, 해결방안, 느낀점
- YOLO 외에 더 다양한 모델들을 돌려보지 못했고 데이터 증강과 YOLOv8-nano 모델의 하이퍼파라미터 튜닝에 의존<br>
  -> 다른 아키텍처와 알고리즘을 가진 모델들을 다양하게 돌려보는 것이 성능향상에 더 좋을 것
    
- 데이터가 충분하지 않아서 데이터 증강을 적용했지만 여전히 부족, 상대적으로 가벼운 모델의 성능이 더 좋았음<br>
  -> 학습 할 데이터가 충분하지 않을때 어떻게 성능을 높여야하는지에 대한 경험과 지식이 필요
     
- 모델 구축을 완료했는데도 GPU 메모리의 한계로 여러 모델들을 학습시키지 못함<br>
  -> 현재 환경에서 가용한 모델과 데이터셋의 크기가 어느 정도인지를 빠르게 파악하여 시간 낭비를 최소화

**개인적으로 참가했던 첫 대회이면서 총 69팀 중 최종 순위 18위(상위 25%)를 기록했다. 더 높은 순위를 달성하지못해서 조금 아쉽긴하지만 이전부터 적용해보고 싶었던 데이터 증강, WBF 등 다양한 기법들을 적용해볼 수 있어서 성적을 떠나 정말 많이 배웠고, 부족한 부분을 더 공부해보고 싶다는 생각이 들었던 의미있는 시간이었다!**
