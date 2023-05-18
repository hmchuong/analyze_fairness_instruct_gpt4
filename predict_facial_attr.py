# Evaluate the face in gender, race, age of generated images
import glob
import argparse
import tqdm
import cv2
import numpy as np
import clip
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from insightface.app import FaceAnalysis

def detect_face_new(image_path, cnn_face_detector):
    image = cv2.imread(image_path)
    try:
        h, w = image.shape[:2]
    except:
        return None
    faces = cnn_face_detector.get(image)
    for idx, face in enumerate(faces):
        img_name = image_path.split("/")[-1]
        path_sp = img_name.split(".")
        bbox = np.round(face['bbox']).astype(int)
        xmin, ymin, xmax, ymax = bbox
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w - 1, xmax)
        ymax = min(h - 1, ymax)
        padding_h = int(0.25 * (ymax - ymin + 1)) // 2
        padding_w = int(0.25 * (xmax - xmin + 1)) // 2
        xmin = max(0, xmin - padding_w)
        xmax = min(w - 1, xmax + padding_w)
        ymin = max(0, ymin - padding_h)
        ymax = min(h - 1, ymax + padding_h)
        # import pdb; pdb.set_trace()
        face = image[ymin: ymax, xmin: xmax, ::-1]
        return Image.fromarray(face)

class FaRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = clip.load("ViT-B/16", device="cuda")[0]
        farl_state = torch.load("../FaRL-Base-Patch16-LAIONFace20M-ep64.pth")
        self.extractor.load_state_dict(farl_state["state_dict"],strict=False)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 18)
        )
        transform_list = [
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ]
        self.transform = transforms.Compose(transform_list)
    
    def forward(self, x, device):
        # Preprocessing
        x = self.transform(x)[None].to(device)
        # with torch.no_grad():
        feat = self.extractor.encode_image(x).float()
        out = self.head(feat)

        # Postprocess
        age_group = out[:, :9].softmax(dim=-1)[0]
        gender_group = out[:, 9:11].softmax(dim=-1)[0]
        race_group = out[:, 11:].softmax(dim=-1)[0]
        return age_group, gender_group, race_group

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate images')
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()

    all_images = sorted(list(glob.glob(f"{args.dir}/*/*.png")))

    # Define FaceAnalysis
    cnn_face_detector = FaceAnalysis(allowed_modules=['detection'], providers=[('CUDAExecutionProvider', {
        'device_id': args.gpu,
    }), 'CPUExecutionProvider'])
    cnn_face_detector.prepare(ctx_id=0, det_thresh=0.5, det_size=(256, 256))

    # Define FaRL
    device = torch.device("cuda:" + str(args.gpu))
    model = FaRL().to(device)
    model.eval()
    state_dict = torch.load("weights/FaRL_ft-FairFace.pth")
    model.load_state_dict(state_dict)
    output_names = ['0-2', '10-19', '20-29', '3-9', '30-39', '40-49', '50-59', '60-69', 'more than 70', \
                    'Female', 'Male', \
                    'Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']
    
    # Run analysis
    f = open(f"{args.dir}.csv", "w")
    f.write("occupation, filename, ")
    f.write(', '.join(output_names) + '\n')
    for image_path in tqdm.tqdm(all_images):
        occupation, image_name = image_path.split("/")[-2:]
        face = detect_face_new(image_path, cnn_face_detector)
        if face is None:
            print(f"Cannot find face in {image_path}")
        else:
            with torch.no_grad():
                age, gender, race = model(face, device)
                out = torch.cat([age, gender, race]).cpu().numpy().tolist()
                f.write(f"{occupation}, {image_name}, ")
                f.write(', '.join([str(x) for x in out]) + '\n')
    f.close()
