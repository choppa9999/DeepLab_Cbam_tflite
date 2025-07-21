# DeepLabV3+ with CBAM for Custom Dataset

이 프로젝트는 TensorFlow 2.x를 사용하여 **DeepLabV3+** 시맨틱 분할(Semantic Segmentation) 모델을 구현합니다. 특히, 모델의 헤드 부분에 **CBAM (Convolutional Block Attention Module)**을 적용하여 성능 향상을 시도하며, Roboflow에서 받은 **COCO 형식의 커스텀 데이터셋**을 학습시키는 전체 파이프라인을 제공합니다.

## 주요 기능

* **DeepLabV3+ with CBAM**: MobileNetV2를 백본으로 사용하고, ASPP와 Decoder에 CBAM 어텐션 모듈을 적용한 모델
* **커스텀 데이터셋 지원**: Roboflow COCO 형식의 데이터셋을 TFRecord로 자동 변환하여 학습
* **다양한 실행 모드**: `train`, `eval`, `inference`, `export` 모드를 통해 학습, 평가, 추론, 배포용 모델 변환을 손쉽게 실행
* **자동화된 설정**: 데이터셋의 이미지 수와 클래스 수를 자동으로 계산하여 설정 과정을 간소화
* **편의 기능**:
    * **다양한 결과 저장**: 추론 시 `--save_mode` 옵션을 통해 오버레이, 비교 이미지 등 다양한 형태로 결과 저장 가능
    * **범례(Legend) 생성**: 추론 시 클래스별 색상 정보를 담은 범례 이미지를 결과물에 병합하여 생성
    * **TensorBoard 안내**: 학습 시 TensorBoard 실행 명령어를 자동으로 안내
    * **진행률 표시**: 여러 이미지를 추론할 때 진행 상황을 시각적으로 표시
    * **혼합 정밀도 학습**: `--mixed_precision` 옵션으로 학습 속도 향상 및 메모리 사용량 절감 가능
    * **학습 이어하기**: `--resume_training` 옵션으로 중단된 학습을 매끄럽게 이어갈 수 있음

---

## 사전 준비 (Prerequisites)

프로젝트 실행에 필요한 라이브러리들을 설치합니다. `requirements.txt` 파일을 사용하면 한 번에 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```
이 프로젝트는 **TensorFlow 2.15 이상** 버전에서 테스트되었습니다. `requirements.txt` 파일에는 CPU 버전의 TensorFlow가 포함되어 있습니다. GPU를 사용하려면, 해당 파일의 `tensorflow` 라인을 주석 처리하고 사용 환경에 맞는 TensorFlow GPU 버전을 직접 설치해주세요.

---

## 사용 방법

### Step 1: 데이터셋 변환 (COCO → TFRecord)

`create_tfrecords_from_coco.py` 파일을 열고, 스크립트 상단에 있는 경로 변수를 자신의 환경에 맞게 수정한 후 실행합니다.

```bash
python3 create_tfrecords_from_coco.py
```

### Step 2: 모델 학습 및 내보내기

`main_cbam.py` 스크립트를 사용하여 모델을 학습하고, 라즈베리 파이에서 사용할 `.tflite` 모델을 생성합니다.

**1. 모델 학습**
```bash
python3 main_cbam.py --mode train --batch 8 --epoch 100
```

**2. TFLite 모델 생성**
학습이 완료되면, 아래 명령어를 실행하여 배포용 모델(`.tflite`)을 생성합니다.
```bash
# 양자화 없이 정확도가 높은 Float32 TFLite 모델 생성 (디버깅용)
python3 main_cbam.py --mode export --no_quant

# 라즈베리 파이 배포를 위한 INT8 양자화 모델 생성
python3 main_cbam.py --mode export
```
* 실행하면 `exported_models/tflite_models/` 폴더 안에 `.tflite` 파일이 생성됩니다.

### Step 3: 라즈베리 파이에서 추론 실행 (Inference)

이제 가벼워진 `inference_tflite.py`와 `.tflite` 모델을 사용하여 라즈베리 파이에서 추론을 실행합니다.

```bash
python3 inference_tflite.py --model ./exported_models/tflite_models/labeling_cbam_quant.tflite \
                            --input /path/to/your/images \
                            --output ./results \
                            --save_mode all
```

* `--model`: **Step 2에서 생성한 `.tflite` 파일**의 경로를 지정합니다.
* `--input`: 추론할 단일 이미지 파일 또는 이미지들이 들어있는 폴더 경로
* `--output`: 결과 이미지가 저장될 폴더 경로
* **`--save_mode`**: 결과 저장 방식을 선택합니다.
    * `overlay` (기본값): 오버레이 이미지와 범례를 합친 최종 결과만 저장합니다.
    * `comparison`: 원본, 마스크, 오버레이 이미지와 범례를 가로로 합친 비교 이미지를 저장합니다.
    * `all`: `overlay`와 `comparison` 방식의 모든 결과물을 저장합니다.
