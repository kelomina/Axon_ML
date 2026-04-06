import os
import lightgbm as lgb
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType

def save_model(model, model_path):
    if str(model_path).lower().endswith('.onnx'):
        onnx_path = model_path
    else:
        onnx_path = os.path.splitext(model_path)[0] + '.onnx'
    train_txt_path = os.path.splitext(onnx_path)[0] + '.train.txt'
    model.save_model(train_txt_path)
    print(f"[+] Training LightGBM model saved to: {train_txt_path}")
    input_dim = int(model.num_feature())
    onnx_model = convert_lightgbm(model, initial_types=[('input', OnnxFloatTensorType([None, input_dim]))], target_opset=15)
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"[+] ONNX model saved to: {onnx_path}")
    legacy_txt_path = os.path.splitext(onnx_path)[0] + '.txt'
    if os.path.exists(legacy_txt_path):
        try:
            os.remove(legacy_txt_path)
            print(f"[+] Removed legacy txt model: {legacy_txt_path}")
        except Exception:
            pass

def load_existing_model(model_path):
    candidates = []
    if str(model_path).lower().endswith('.onnx'):
        candidates.append(os.path.splitext(model_path)[0] + '.train.txt')
        candidates.append(os.path.splitext(model_path)[0] + '.txt')
    candidates.append(model_path)
    seen = set()
    ordered = []
    for p in candidates:
        if p and p not in seen:
            ordered.append(p)
            seen.add(p)
    for candidate in ordered:
        if os.path.exists(candidate):
            print(f"[*] Loading existing model: {candidate}")
            try:
                model = lgb.Booster(model_file=candidate)
                print("[+] Existing model loaded successfully")
                return model
            except Exception as e:
                print(f"[!] Model loading failed: {e}")
    print(f"[-] Existing model not found: {model_path}")
    return None
