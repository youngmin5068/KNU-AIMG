# KNU-AIMG

## 의료기기 탑재용 AI 모델 개발

- 유방촬영술 (Mammography) 데이터에서 유방 종양 부위 탐지 딥러닝 모델 개발
- Tumor mask의 부재로 인해 이미지 단위 classification을 통한 종양 탐지 (Weakly Supervised Learning) 
- KTL 데이터를 활용하여 Localization 성능 향상
  - KTL데이터종양 bounding box 제공
  - 종양 예측 부위가 bounding box 외부에 존재하지 않도록 loss function 설계
 
![스크린샷 2024-08-28 오후 1 22 29](https://github.com/user-attachments/assets/c50a8673-9d3a-441c-bcb0-03d88cf1e220)
- MLO(Mediolateral oblique) 데이터는 대흉근 부위가 존재하여 대흉근 부위를 잘못 예측하는 경우 발생
- 대흉근을 제거하는 알고리즘을 활용하여 불필요한 부위 제거 후 재학습
- 품질 향상

- 유방 촬영술 영상 내 종양 예측 프로그램 개발
- 기업(제노레이)에 모델 및 실행파일 제공, 업데이트 진행
- 기업(제노레이)와 협업하여 2등급 의료기기 등록 진행
