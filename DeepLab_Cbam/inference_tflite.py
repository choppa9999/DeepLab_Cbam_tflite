import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

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
        raise ValueError(f'label은 2D 배열이어야 합니다. 현재 shape: {label.shape}')
    colormap = create_pascal_label_colormap()
    return colormap[label].astype(np.uint8)


def run_inference(args):
    """
    TFLite 모델을 사용하여 이미지에 대한 세그멘테이션 추론을 수행합니다.

    Args:
        args (argparse.Namespace): 스크립트 실행 시 전달된 인자.
    """
    model_path = args.model_path
    input_path = args.input_path
    output_dir = args.output_dir
    save_mode = args.save_mode
    
    print(f"TFLite 모델 로드 중: {model_path}")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"오류: TFLite 모델 파일을 로드할 수 없습니다. 경로를 확인하세요: {e}")
        return

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    _, height, width, _ = input_details['shape']
    print(f"모델 입력 크기: ({height}, {width})")

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

    os.makedirs(output_dir, exist_ok=True)
    print(f"총 {len(image_paths)}개의 이미지를 처리합니다. 결과는 '{output_dir}'에 저장됩니다.")

    for image_path in tqdm(image_paths, desc="TFLite 추론 진행률"):
        try:
            original_image = Image.open(image_path).convert('RGB')
            original_size = original_image.size
            resized_image = original_image.resize((width, height))

            input_type = input_details['dtype']
            
            if 'printed_info' not in locals():
                print(f"\n💡 정보: 모델이 예상하는 입력 타입은 '{np.dtype(input_type).name}' 입니다.")
                printed_info = True

            if input_type == np.uint8:
                input_data = np.array(resized_image, dtype=np.uint8)
                input_data = np.expand_dims(input_data, axis=0)
            else:
                image_np_float = np.array(resized_image, dtype=np.float32)
                image_batch = np.expand_dims(image_np_float, axis=0)
                input_data = tf.keras.applications.mobilenet_v2.preprocess_input(image_batch)
            
            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details['index'])
            seg_map = np.squeeze(output_data).astype(np.uint8)

            seg_map_pil = Image.fromarray(seg_map)
            resized_seg_map = seg_map_pil.resize(original_size, Image.NEAREST)
            
            # --- [수정] 저장 모드에 따라 다른 이미지 생성 및 저장 ---
            color_mask_image = label_to_color_image(np.array(resized_seg_map))
            original_image_np = np.array(original_image)
            overlayed_image = (original_image_np * 0.6 + color_mask_image * 0.4).astype(np.uint8)
            
            base_filename = os.path.basename(image_path)
            filename_no_ext, _ = os.path.splitext(base_filename)

            if save_mode in ['comparison', 'all']:
                comparison_image = np.hstack((original_image_np, color_mask_image, overlayed_image))
                comparison_path = os.path.join(output_dir, f"{filename_no_ext}_comparison.png")
                Image.fromarray(comparison_image).save(comparison_path)

            if save_mode in ['overlay', 'all']:
                output_path = os.path.join(output_dir, f"{filename_no_ext}_overlay.png")
                Image.fromarray(overlayed_image).save(output_path)
            # -----------------------------------------------------------

        except Exception as e:
            print(f"\n오류: '{image_path}' 처리 중 문제가 발생했습니다: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- 모든 추론 완료 ---")


def main():
    parser = argparse.ArgumentParser(description="TFLite DeepLabV3+ CBAM 모델 추론 스크립트")
    parser.add_argument('--model_path', type=str, required=True,
                        help="추론에 사용할 .tflite 모델 파일 경로")
    parser.add_argument('--input_path', type=str, required=True,
                        help="추론할 입력 이미지 또는 폴더 경로")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="추론 결과를 저장할 폴더 경로")
    # --- [추가] 저장 모드 인자 ---
    parser.add_argument(
        '--save_mode',
        type=str,
        default='overlay',
        choices=['overlay', 'comparison', 'all'],
        help="추론 결과 저장 방식: 'overlay' (덮어쓰기), 'comparison' (비교), 'all' (모두)"
    )
    # --------------------------
    args = parser.parse_args()

    run_inference(args)


if __name__ == '__main__':
    main()

