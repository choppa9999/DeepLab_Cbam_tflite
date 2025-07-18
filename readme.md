DeepLabV3+ with CBAM for Road Surface Segmentation
1. 프로젝트 개요
본 프로젝트는 DeepLabV3+ 아키텍처에 **CBAM (Convolutional Block Attention Module)**을 적용하여 도로 노면 상태(건조, 습윤, 슬러시 등)를 분할(Segmentation)하는 모델을 구현합니다.

학습된 모델은 TensorFlow Lite(.tflite) 형식으로 변환 및 양자화(INT8)하여, 모바일이나 임베디드 환경과 같은 리소스가 제한된 장치에서도 효율적으로 동작할 수 있도록 최적화하는 전체 파이프라인을 포함합니다.

2. 주요 기능
DeepLabV3+ & CBAM: 강력한 분할 성능을 자랑하는 DeepLabV3+ 모델에 어텐션 메커니즘(CBAM)을 추가하여 정확도를 향상시켰습니다.

전이 학습 (Transfer Learning): ImageNet으로 사전 학습된 MobileNetV2 백본을 사용하여 더 빠르고 안정적으로 모델을 학습합니다.

2단계 학습 전략:

특징 추출: 백본을 동결하고 새로 추가된 분할 헤드만 학습합니다.

미세 조정: 전체 모델을 낮은 학습률로 추가 학습하여 성능을 극대화합니다.

TFLite 변환 및 양자화: 학습된 모델을 Float32 및 INT8 정수형 TFLite 모델로 변환하여, 모델의 크기를 줄이고 추론 속도를 높입니다.

다양한 추론 결과 저장: 오버레이, 비교 이미지 등 다양한 시각화 옵션을 제공합니다.

3. 파일 구조
.
├── checkpoints/                # 학습된 모델(.keras)이 저장되는 폴더
│   └── labeling/
│       └── best_model.keras
├── data/                       # 데이터셋 폴더
│   └── labeling/
│       └── tfrecords/          # TFRecord 파일들이 위치하는 곳
├── exported_models/            # 내보내기된 모델이 저장되는 폴더
│   ├── labeling_cbam/          # SavedModel 형식
│   └── tflite_models/          # TFLite 형식 (.tflite)
├── main_cbam.py                # 학습, 평가, 모델 내보내기 메인 스크립트
├── inference_tflite.py         # TFLite 모델 추론 스크립트
├── deeplab_v3_plus_cbam.py     # 모델 아키텍처 정의
├── segmentation_dataset.py     # 데이터셋 로더 정의
└── README.md                   # 프로젝트 설명 파일 (현재 문서)

4. 환경 설정
필요한 라이브러리를 설치합니다.

pip install tensorflow Pillow matplotlib tqdm psutil

5. 데이터셋 준비
프로젝트 루트에 data/labeling/tfrecords 폴더를 생성합니다.

Roboflow 등에서 생성한 TFRecord 형식의 데이터셋 파일들을 위 폴더에 위치시킵니다.

학습 데이터: trainval-*.tfrecord

검증 데이터: valid-*.tfrecord (선택 사항, 없을 경우 양자화 시 학습 데이터 사용)

6. 사용 방법
모든 명령어는 프로젝트의 루트 디렉토리에서 실행합니다.

1단계: 모델 학습
main_cbam.py 스크립트를 train 모드로 실행하여 모델 학습을 시작합니다.

python main_cbam.py --mode train

학습이 완료되면 가장 성능이 좋은 모델이 checkpoints/labeling/best_model.keras 경로에 저장됩니다.

주요 학습 인자:

--initial_epochs: 특징 추출 단계의 에포크 수 (기본값: 10)

--fine_tune_epochs: 미세 조정 단계의 에포크 수 (기본값: 40)

--batch_size: 배치 크기 (기본값: 8)

--learning_rate: 초기 학습률 (기본값: 1e-3)

--no_transfer: 전이 학습을 사용하지 않고 처음부터 학습할 경우 사용

--resume_training: 저장된 체크포인트에서 학습을 이어서 진행할 경우 사용

2단계: 모델 내보내기
학습된 모델을 SavedModel 및 TFLite 형식으로 변환합니다.

python main_cbam.py --mode export

이 명령어를 실행하면 다음 파일들이 생성됩니다.

exported_models/labeling_cbam/<타임스탬프>/: SavedModel

exported_models/tflite_models/labeling_cbam_float32.tflite: Float32 TFLite 모델

exported_models/tflite_models/labeling_cbam_quant.tflite: INT8 양자화 TFLite 모델

3단계: TFLite 모델로 추론
inference_tflite.py 스크립트를 사용하여 변환된 TFLite 모델의 성능을 확인합니다.

# 예시: INT8 양자화 모델로 추론 실행
python inference_tflite.py \
  --model_path ./exported_models/tflite_models/labeling_cbam_quant.tflite \
  --input_path ./path/to/your/test_images \
  --output_dir ./inference_results \
  --save_mode comparison

주요 추론 인자:

--model_path: 사용할 .tflite 모델 파일의 경로 (필수)

--input_path: 추론할 이미지 또는 이미지가 담긴 폴더 경로 (필수)

--output_dir: 결과 이미지를 저장할 폴더 경로 (필수)

--save_mode: 결과 저장 방식 (기본값: overlay)

overlay: 원본 이미지에 마스크를 겹쳐서 저장

comparison: 원본, 마스크, 오버레이 이미지를 나란히 붙여서 저장

all: overlay와 comparison 결과를 모두 저장
