import torch
import cv2
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Small 모델 설정
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 
             'out_channels': [48, 96, 192, 384]},
}

def load_model():
    model = DepthAnythingV2(**model_configs['vits'])
    model.load_state_dict(torch.load(
        'checkpoints/depth_anything_v2_vits.pth', 
        map_location='cpu'
    ))
    model = model.to(DEVICE).eval()
    return model

def test_inference(model, image_path):
    raw_img = cv2.imread(image_path)
    depth = model.infer_image(raw_img)  # H x W numpy array
    return raw_img, depth

if __name__ == '__main__':
    model = load_model()
    print(f'Model loaded. Total params: {sum(p.numel() for p in model.parameters()):,}')
    
    # test image
    raw_img, depth = test_inference(model, 'test_images/sample.jpg')
    print(f'Input shape: {raw_img.shape}, Depth shape: {depth.shape}')
    print(f'Depth min: {depth.min():.3f}, max: {depth.max():.3f}')
    
    # save visualization
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    cv2.imwrite('results/raw/pytorch_output.png', depth_colored)
    print('Saved PyTorch output to results/raw/pytorch_output.png')