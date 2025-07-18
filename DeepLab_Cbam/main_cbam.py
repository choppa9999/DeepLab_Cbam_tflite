import os
import datetime
import argparse
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import io
import re

from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

import psutil

# 사용자 정의 모듈 임포트 (동일한 디렉토리에 파일이 있어야 합니다)
from deeplab_v3_plus_cbam import DeeplabV3PlusWithCBAM, CBAM
from segmentation_dataset import SegmentationDataset


class EpochMetricsLogger(keras.callbacks.Callback):
    """각 에포크 종료 시 학습 및 검증 지표를 깔끔하게 출력하는 콜백입니다."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get('pixel_accuracy')
        val_acc = logs.get('val_pixel_accuracy')
        train_iou = logs.get('mean_iou_with_ignore')
        val_iou = logs.get('val_mean_iou_with_ignore')

        print(f"\n\nEpoch {epoch + 1} 완료" + "=" * 40)
        if train_acc is not None:
            print(f"  - 학습 정확도 (Train Accuracy)   : {train_acc:.4f}")
        if train_iou is not None:
            print(f"  - 학습 IoU (Train IoU)           : {train_iou:.4f}")
        if val_acc is not None:
            print(f"  - 검증 정확도 (Validation Accuracy): {val_acc:.4f}")
        if val_iou is not None:
            print(f"  - 검증 IoU (Validation IoU)      : {val_iou:.4f}")
        print("=" * 55 + "\n")


def setup_environment(args):
    """TensorFlow 환경 설정을 수행합니다."""
    tf.config.optimizer.set_jit(False)
    print("[정보] XLA JIT 컴파일러가 비활성화되었습니다.")

    if args.mixed_precision:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print("혼합 정밀도(Mixed Precision) 학습이 활성화되었습니다.")

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[정보] GPU 메모리 동적 할당이 활성화되었습니다.")
        except RuntimeError as e:
            print(f"GPU 메모리 설정 중 오류 발생: {e}")
    print(f"TensorFlow 버전: {tf.__version__}")
    print(f"사용 가능한 GPU 수: {len(physical_devices)}")


# --- 상수 정의 (사용자 수정 사항 반영) ---
INPUT_SHAPE = (512, 512, 3)
IGNORE_LABEL = 255
DATASET_NAME = 'images'
DATASET_DIR = './data/tfrecords'
OUTPUT_STRIDE = 16
MODEL_DIR = './checkpoints'
EXPORT_SAVEDMODEL_DIR = './exported_models/Model'


def create_pascal_label_colormap():
    """PASCAL VOC와 유사한 컬러맵을 생성합니다."""
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap


def label_to_color_image(label):
    """세그멘테이션 라벨(클래스 ID)을 컬러 이미지로 변환합니다."""
    if label.ndim != 2:
        raise ValueError('label은 2D 배열이어야 합니다. 현재 shape: ', label.shape)
    colormap = create_pascal_label_colormap()
    return colormap[label]


def create_legend_image(json_path):
    """COCO 주석 파일에서 범례 이미지를 생성합니다."""
    print("범례(legend) 생성 중...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"경고: 범례를 생성하기 위한 '{json_path}' 파일을 찾을 수 없습니다.")
        return None

    ALLOWED_CLASSES = ['dry', 'humid', 'slush', 'snow', 'wet']
    categories_original = {cat['id']: cat['name'] for cat in coco_data['categories']}
    categories = {k: v for k, v in categories_original.items() if v in ALLOWED_CLASSES}

    sorted_cat_ids = sorted(categories.keys())
    cat_id_map = {original_coco_id: i + 1 for i, original_coco_id in enumerate(sorted_cat_ids)}
    new_id_to_name = {new_id: categories[original_coco_id] for original_coco_id, new_id in cat_id_map.items()}
    legend_labels = {0: 'background', **new_id_to_name}
    colormap = create_pascal_label_colormap()

    fig, ax = plt.subplots(figsize=(3, len(legend_labels) * 0.35), dpi=120)
    ax.set_title("Legend", fontweight='bold')

    for i, (class_id, class_name) in enumerate(sorted(legend_labels.items())):
        color = colormap[class_id] / 255.0
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
        ax.text(1.2, i + 0.5, f"{class_id}: {class_name}", va='center', fontsize=10)

    ax.set_ylim(len(legend_labels), -0.5)
    ax.set_xlim(0, 4)
    ax.axis('off')

    try:
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='png', bbox_inches='tight', dpi=150)
        io_buf.seek(0)
        legend_img = np.array(Image.open(io_buf))
        plt.close(fig)
        print("범례 이미지 생성 완료.")
        if legend_img.shape[2] == 4:
            legend_img = legend_img[:, :, :3]
        return legend_img
    except Exception as e:
        print(f"오류: 범례 이미지를 생성하는 중 문제가 발생했습니다: {e}")
        plt.close(fig)
        return None


class MeanIoUWithIgnore(tf.keras.metrics.Metric):
    """특정 라벨(무시 라벨)을 제외하고 MeanIoU를 계산하는 사용자 정의 지표입니다."""
    def __init__(self, num_classes, name='mean_iou_with_ignore', dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.iou_metric = keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        
        mask = tf.not_equal(tf.squeeze(y_true, axis=-1), IGNORE_LABEL)
        
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        
        self.iou_metric.update_state(y_true_masked, y_pred_masked)

    def result(self):
        return self.iou_metric.result()

    def reset_state(self):
        self.iou_metric.reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config


def get_loss_and_metrics(num_classes):
    """손실 함수와 평가지표를 반환합니다."""
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                         reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    accuracy_metric = keras.metrics.SparseCategoricalAccuracy(name='pixel_accuracy')
    mean_iou_metric = MeanIoUWithIgnore(num_classes=num_classes)
    return loss_fn, accuracy_metric, mean_iou_metric


def train(args):
    """모델 학습을 수행합니다."""
    print(f"\n--- 학습 모드 시작 ---")

    checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
    
    train_loader_info = SegmentationDataset(DATASET_NAME, DATASET_DIR, 'trainval', INPUT_SHAPE[0], INPUT_SHAPE[1], is_training=True)
    num_classes = train_loader_info.num_classes
    loss_fn, acc_metric, iou_metric = get_loss_and_metrics(num_classes)

    if os.path.exists(checkpoint_path) and args.resume_training:
        print(f"\n[정보] 이전 학습에서 저장된 모델을 로드합니다: {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path, custom_objects={'MeanIoUWithIgnore': MeanIoUWithIgnore,
                                                                         'DeeplabV3PlusWithCBAM': DeeplabV3PlusWithCBAM,
                                                                         'CBAM': CBAM})
        print("모델과 옵티마이저 상태를 모두 로드하여 학습을 이어서 시작합니다.")
    else:
        print("\n[정보] 처음부터 새로운 학습을 시작합니다.")
        if args.resume_training:
            print("경고: --resume_training이 지정되었지만, 저장된 모델 파일이 없습니다.")

        if num_classes <= 1:
            print(f"오류: 데이터셋에서 유효한 클래스를 찾지 못했습니다. (num_classes: {num_classes})")
            return

        backbone_weights = 'imagenet' if not args.no_transfer else None
        if backbone_weights:
            print("\n[정보] 전이 학습을 위해 ImageNet으로 사전 학습된 백본 가중치를 사용합니다.")
        else:
            print("\n[정보] 전이 학습을 사용하지 않고, 무작위 가중치로 백본을 초기화합니다.")
        
        print(f"       모델은 총 {num_classes}개의 대상을 구분하도록 학습됩니다.")

        model = DeeplabV3PlusWithCBAM(
            num_classes=num_classes,
            input_shape=INPUT_SHAPE,
            output_stride=OUTPUT_STRIDE,
            backbone_weights=backbone_weights
        )
        model.build(input_shape=(None, *INPUT_SHAPE))
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
                      loss=loss_fn, metrics=[acc_metric, iou_metric])

    train_loader = SegmentationDataset(DATASET_NAME, DATASET_DIR, 'trainval', INPUT_SHAPE[0], INPUT_SHAPE[1], is_training=True)
    val_loader = SegmentationDataset(DATASET_NAME, DATASET_DIR, 'valid', INPUT_SHAPE[0], INPUT_SHAPE[1], is_training=False)
    total_train_samples = train_loader.get_num_data()
    total_val_samples = val_loader.get_num_data()
    steps_per_epoch = total_train_samples // args.batch
    validation_steps = total_val_samples // args.batch
    if steps_per_epoch == 0:
        raise ValueError(f"steps_per_epoch이 0입니다. 데이터셋 크기({total_train_samples})와 배치 크기({args.batch})를 확인하세요.")
    if validation_steps == 0 and total_val_samples > 0:
        validation_steps = 1

    train_ds = train_loader.make_batch(args.batch)
    val_ds = val_loader.make_batch(args.batch)

    def preprocess_data(image, label, weight):
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label, weight

    print("\n[정보] 학습 및 검증 데이터셋에 MobileNetV2 전처리(픽셀값 -1~1 스케일링)를 적용합니다.")
    train_ds = train_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    model.summary()
    log_dir = os.path.join(MODEL_DIR, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=False, monitor='val_mean_iou_with_ignore',
            mode='max', save_best_only=True, verbose=1
        ),
        EpochMetricsLogger()
    ]

    # [수정] 들여쓰기 오류 수정 및 인자 이름 변경 (args.epochs -> args.epoch)
    total_epochs = args.epoch
    initial_epochs = args.initial_epochs
    fine_tune_epochs = total_epochs - initial_epochs

    if fine_tune_epochs < 0:
        raise ValueError("initial_epochs는 전체 epoch보다 클 수 없습니다.")

    if args.resume_training:
        print("\n--- 저장된 지점부터 미세 조정을 계속합니다 ---")
        model.fit(train_ds, epochs=total_epochs, steps_per_epoch=steps_per_epoch,
                  validation_data=val_ds, validation_steps=validation_steps, callbacks=callbacks)
    elif args.no_transfer:
        print("\n--- 전이 학습 없이 전체 모델 학습 시작 ---")
        if total_epochs > 0:
            model.fit(train_ds, epochs=total_epochs, steps_per_epoch=steps_per_epoch,
                      validation_data=val_ds, validation_steps=validation_steps, callbacks=callbacks)
    else:
        # 1단계: 특징 추출
        if initial_epochs > 0:
            print("\n--- 1단계: 특징 추출 (백본 동결) ---")
            try:
                backbone = model.get_layer('backbone')
                backbone.trainable = False
                print("[정보] 백본 레이어를 동결했습니다. 분할 헤드만 학습합니다.")
            except ValueError:
                print("[경고] 'backbone'이라는 이름의 레이어를 찾을 수 없어 동결을 건너뜁니다.")
            
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
                          loss=loss_fn, metrics=[acc_metric, iou_metric])
            
            print(f"초기 {initial_epochs} 에포크 동안 헤드를 학습합니다...")
            model.fit(train_ds, epochs=initial_epochs, steps_per_epoch=steps_per_epoch,
                      validation_data=val_ds, validation_steps=validation_steps, callbacks=callbacks)

        # 2단계: 미세 조정
        if fine_tune_epochs > 0:
            print("\n--- 2단계: 미세 조정 (백본 동결 해제) ---")
            try:
                backbone = model.get_layer('backbone')
                backbone.trainable = True
                print("[정보] 백본 레이어의 동결을 해제했습니다. 전체 모델을 미세 조정합니다.")
            except ValueError:
                print("[경고] 'backbone' 레이어를 찾을 수 없어 동결 해제를 건너뜁니다.")
            
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate / 10),
                          loss=loss_fn, metrics=[acc_metric, iou_metric])
            
            print(f"추가 {fine_tune_epochs} 에포크 동안 전체 모델을 미세 조정합니다...")
            model.fit(train_ds, epochs=total_epochs,
                      initial_epoch=initial_epochs, steps_per_epoch=steps_per_epoch,
                      validation_data=val_ds, validation_steps=validation_steps, callbacks=callbacks)

    print("--- 학습 완료 ---")


def evaluate(args):
    print(f"\n--- 평가 모드 시작 ---")

    checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
    if not os.path.exists(checkpoint_path):
        print(f"오류: '{checkpoint_path}'에서 학습된 모델 파일을 찾을 수 없습니다.")
        return

    print(f"학습된 모델 로드 시도: {checkpoint_path}")
    try:
        model = keras.models.load_model(
            checkpoint_path,
            custom_objects={'MeanIoUWithIgnore': MeanIoUWithIgnore, 'DeeplabV3PlusWithCBAM': DeeplabV3PlusWithCBAM,
                            'CBAM': CBAM},
            compile=False
        )
        print("학습된 모델 로드 완료. 평가를 위해 다시 컴파일합니다.")
    except Exception as e:
        print(f"\n오류: 모델을 로드하는 중 문제가 발생했습니다: {e}")
        return

    val_loader = SegmentationDataset(DATASET_NAME, DATASET_DIR, 'valid', INPUT_SHAPE[0], INPUT_SHAPE[1],
                                     is_training=False)

    num_classes = model.layers[-1].filters
    total_val_samples = val_loader.get_num_data()
    if total_val_samples == 0:
        print(f"오류: 검증 데이터셋에 샘플이 없습니다.")
        return
    eval_steps = math.ceil(total_val_samples / 1)

    val_ds = val_loader.make_batch(1)

    loss_fn, acc_metric, iou_metric = get_loss_and_metrics(num_classes)
    model.compile(loss=loss_fn, metrics=[acc_metric, iou_metric])

    print(f"모델 평가 중... (Total steps: {eval_steps})")
    results = model.evaluate(val_ds, steps=eval_steps)
    print(f"평가 결과: {dict(zip(model.metrics_names, results))}")
    print(f"--- 평가 완료 ---")


def inference(args):
    process = psutil.Process(os.getpid())
    initial_ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\n--- 추론 모드 시작 (초기 RAM: {initial_ram_mb:.2f} MB) ---")
    if not args.input or not args.output:
        print("오류: --input과 --output 인자를 모두 지정해야 합니다.")
        return

    checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
    if not os.path.exists(checkpoint_path):
        print(f"오류: '{MODEL_DIR}'에서 학습된 모델 파일(*.keras)을 찾을 수 없습니다.")
        return

    print(f"학습된 최고 성능 모델 로드 시도: {checkpoint_path}")
    try:
        model = keras.models.load_model(
            checkpoint_path,
            custom_objects={'MeanIoUWithIgnore': MeanIoUWithIgnore, 'DeeplabV3PlusWithCBAM': DeeplabV3PlusWithCBAM,
                            'CBAM': CBAM},
            compile=False
        )
        print("학습된 모델 로드 완료 (추론 모드).")
    except Exception as e:
        print(f"\n오류: 모델을 로드하는 중 문제가 발생했습니다: {e}")
        return

    legend_image = None
    if args.coco_dir:
        for split in ['train', 'valid', 'test']:
            json_path = os.path.join(args.coco_dir, split, '_annotations.coco.json')
            if os.path.exists(json_path):
                legend_image = create_legend_image(json_path)
                break

    if legend_image is None:
        print("경고: 범례를 생성하지 못했지만, 추론은 계속 진행합니다.")

    input_path = args.input.strip()
    image_paths = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(input_path, filename))
    elif os.path.isfile(input_path) and input_path.lower().endswith(supported_extensions):
        image_paths.append(input_path)

    if not image_paths:
        print(f"오류: '{input_path}' 경로에서 처리할 이미지를 찾을 수 없습니다.")
        return

    os.makedirs(args.output, exist_ok=True)
    print(f"총 {len(image_paths)}개의 이미지를 처리합니다.")

    for image_path in tqdm(image_paths, desc="추론 진행률"):
        try:
            image_raw = tf.io.read_file(image_path)
            original_image_tensor = tf.image.decode_image(image_raw, channels=3)
            original_image_np = original_image_tensor.numpy()

            image_float = tf.cast(original_image_tensor, tf.float32)
            resized_image = tf.image.resize(image_float, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
            image_batch = tf.expand_dims(resized_image, 0)

            predictions = model.predict(image_batch, verbose=0)

            current_ram_mb = process.memory_info().rss / (1024 * 1024)
            tqdm.write(f"  - '{os.path.basename(image_path)}' 추론 후 RAM: {current_ram_mb:.2f} MB")

            original_height, original_width, _ = original_image_np.shape
            seg_map = tf.argmax(predictions[0], axis=-1)
            seg_map = tf.cast(seg_map, tf.uint8)
            seg_map_resized = tf.image.resize(tf.expand_dims(seg_map, -1), (original_height, original_width),
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            seg_map_resized = tf.squeeze(seg_map_resized, axis=-1)

            color_mask_image = label_to_color_image(seg_map_resized.numpy())
            overlayed_image = (original_image_np * 0.6 + color_mask_image * 0.4).astype(np.uint8)

            base_filename = os.path.basename(image_path)
            filename_no_ext, _ = os.path.splitext(base_filename)

            resized_legend = None
            if legend_image is not None:
                resized_legend = tf.image.resize(legend_image, (original_height, legend_image.shape[1])).numpy().astype(
                    np.uint8)

            if args.save_mode in ['comparison', 'all']:
                comparison_image = np.hstack((original_image_np, color_mask_image, overlayed_image))
                if resized_legend is not None:
                    final_comparison_image = np.hstack((comparison_image, resized_legend))
                else:
                    final_comparison_image = comparison_image

                comparison_path = os.path.join(args.output, f"{filename_no_ext}_comparison.png")
                tf.keras.utils.save_img(comparison_path, final_comparison_image)

            if args.save_mode in ['overlay', 'all']:
                if resized_legend is not None:
                    final_overlay_image = np.hstack((overlayed_image, resized_legend))
                else:
                    final_overlay_image = overlayed_image

                output_path = os.path.join(args.output, f"{filename_no_ext}_result.png")
                tf.keras.utils.save_img(output_path, final_overlay_image)

        except Exception as e:
            print(f"오류: '{image_path}' 처리 중 문제가 발생했습니다: {e}")

    final_ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\n[정보] 모든 추론 완료 후 RAM 사용량: {final_ram_mb:.2f} MB")
    print("\n--- 모든 추론 완료 ---")


def export_savedmodel(args):
    """모델을 TFLite 변환을 위한 SavedModel 형식으로 내보냅니다."""
    print(f"\n--- SavedModel 내보내기 모드 시작 ---")
    
    print("[정보] GPU 메모리 오류 방지를 위해 CPU에서 내보내기 작업을 수행합니다.")
    with tf.device('/CPU:0'):
        checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
        if not os.path.exists(checkpoint_path):
            print(f"오류: '{MODEL_DIR}'에서 학습된 모델 파일을 찾을 수 없습니다.")
            return None

        print(f"학습된 모델 로드: {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path, compile=False,
            custom_objects={'MeanIoUWithIgnore': MeanIoUWithIgnore,
                            'DeeplabV3PlusWithCBAM': DeeplabV3PlusWithCBAM,
                            'CBAM': CBAM})

        dummy_input = tf.zeros((1, *INPUT_SHAPE), dtype=tf.float32)
        _ = model(dummy_input)
        print("[정보] 모델 그래프 추적을 위해 더미 데이터로 모델을 호출했습니다.")

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        export_path = os.path.join(EXPORT_SAVEDMODEL_DIR, timestamp)
        os.makedirs(export_path, exist_ok=True)
        
        print(f"SavedModel을 다음 경로에 저장합니다: {export_path}")
        model.save(export_path)
        print("SavedModel 내보내기 완료.")
        
    return export_path


def export_tflite_model(saved_model_path):
    """SavedModel을 TFLite 형식으로 변환합니다 (Float32 및 INT8)."""
    print(f"\n--- TensorFlow Lite 모델 변환 모드 시작 ---")

    if not saved_model_path or not os.path.exists(saved_model_path):
        print(f"오류: 유효한 SavedModel 경로가 없습니다: {saved_model_path}")
        return

    base_model = tf.saved_model.load(saved_model_path)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, *INPUT_SHAPE], dtype=tf.float32)])
    def model_with_argmax(inputs):
        logits = base_model(inputs, training=False)
        return {'segmentation_mask': tf.argmax(logits, axis=-1, output_type=tf.int32)}

    concrete_func = model_with_argmax.get_concrete_function()

    # --- 1. Float32 모델 변환 (디버깅용) ---
    print("\n[1/2] Float32 TFLite 모델 변환 중 (양자화 없음)...")
    try:
        converter_float = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter_float.optimizations = []
        tflite_model_float = converter_float.convert()
        
        TFLITE_FLOAT_MODEL_PATH = './exported_models/tflite_models/Model_float32.tflite'
        os.makedirs(os.path.dirname(TFLITE_FLOAT_MODEL_PATH), exist_ok=True)
        with open(TFLITE_FLOAT_MODEL_PATH, "wb") as f:
            f.write(tflite_model_float)
        print(f"✅ Float32 TFLite 모델이 성공적으로 저장되었습니다: {TFLITE_FLOAT_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Float32 TFLite 모델 변환 중 오류 발생: {e}")

    # --- 2. INT8 정수 양자화 모델 변환 ---
    print("\n[2/2] INT8 정수 양자화 TFLite 모델 변환 중...")
    try:
        converter_quant = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_data_gen():
            """양자화를 위한 대표 데이터셋을 생성합니다. 'valid'가 없으면 'trainval'을 사용합니다."""
            print("  - 양자화를 위한 대표 데이터셋 생성 중...")
            subset_to_use = 'valid'
            try:
                dataset_loader = SegmentationDataset(DATASET_NAME, DATASET_DIR, subset_to_use, INPUT_SHAPE[0], INPUT_SHAPE[1], is_training=False)
                if dataset_loader.get_num_data() == 0:
                    raise ValueError(f"'{subset_to_use}' 데이터셋이 비어있습니다.")
            except (FileNotFoundError, ValueError) as e:
                print(f"경고: '{subset_to_use}' 서브셋을 사용할 수 없습니다 ({e}). 'trainval' 서브셋을 대신 사용합니다.")
                subset_to_use = 'trainval'
                dataset_loader = SegmentationDataset(DATASET_NAME, DATASET_DIR, subset_to_use, INPUT_SHAPE[0], INPUT_SHAPE[1], is_training=False)
            
            if dataset_loader.get_num_data() == 0:
                 raise ValueError(f"오류: 대표 데이터셋으로 사용할 '{subset_to_use}' 서브셋에 데이터가 없습니다.")

            print(f"  - 대표 데이터셋으로 '{subset_to_use}' 서브셋을 사용합니다.")
            for images, _, _ in dataset_loader.make_batch(1).take(150):
                preprocessed_images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
                yield [preprocessed_images]
            print("  - 대표 데이터셋 생성 완료.")

        converter_quant.representative_dataset = representative_data_gen
        converter_quant.inference_input_type = tf.uint8
        converter_quant.inference_output_type = tf.int8
        converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        tflite_quant_model = converter_quant.convert()
        
        TFLITE_QUANT_MODEL_PATH = './exported_models/tflite_models/Model_quant.tflite'
        os.makedirs(os.path.dirname(TFLITE_QUANT_MODEL_PATH), exist_ok=True)
        with open(TFLITE_QUANT_MODEL_PATH, "wb") as f:
            f.write(tflite_quant_model)
        print(f"✅ INT8 양자화 TFLite 모델이 성공적으로 저장되었습니다: {TFLITE_QUANT_MODEL_PATH}")
    except Exception as e:
        print(f"❌ INT8 양자화 TFLite 모델 변환 중 오류 발생: {e}")


def main():
    parser = argparse.ArgumentParser(description="DeepLabV3+ CBAM 모델 학습 및 배포 스크립트")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'eval', 'inference', 'export', 'export_tflite'],
                        help="실행할 모드 선택: train, eval, inference, export(savedmodel+tflite), export_tflite(tflite만)")
    parser.add_argument('--input', type=str, help="추론에 사용할 입력 이미지 또는 폴더 경로")
    parser.add_argument('--output', type=str, help="추론 결과를 저장할 폴더 경로")
    parser.add_argument('--coco_dir', type=str, help="클래스 정보 및 범례 생성을 위한 원본 Roboflow 데이터셋 폴더 경로")
    
    # [수정] 학습 관련 인자 이름 변경
    parser.add_argument('--epoch', type=int, default=50, help="전체 학습 에포크 수")
    parser.add_argument('--initial_epochs', type=int, default=10, help="전체 에포크 중, 백본 동결 상태에서 헤드만 학습시킬 초기 에포크 수 (전이 학습 시)")
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=4e-5)
    parser.add_argument('--mixed_precision', action='store_true', help="혼합 정밀도 학습 활성화")
    parser.add_argument('--resume_training', action='store_true', help="저장된 best_model.keras에서 학습을 이어합니다.")
    parser.add_argument('--no_transfer', action='store_true', help="전이 학습을 사용하지 않고 처음부터 학습합니다.")

    args = parser.parse_args()

    setup_environment(args)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'export':
        exported_path = export_savedmodel(args)
        if exported_path:
            export_tflite_model(exported_path)
    elif args.mode == 'export_tflite':
        try:
            latest_export = sorted(os.listdir(EXPORT_SAVEDMODEL_DIR))[-1]
            saved_model_to_convert = os.path.join(EXPORT_SAVEDMODEL_DIR, latest_export)
            print(f"가장 최근 SavedModel을 변환합니다: {saved_model_to_convert}")
            export_tflite_model(saved_model_to_convert)
        except (FileNotFoundError, IndexError):
            print(f"오류: '{EXPORT_SAVEDMODEL_DIR}'에서 SavedModel을 찾을 수 없습니다. 먼저 'export' 모드를 실행하세요.")


if __name__ == '__main__':
    main()

