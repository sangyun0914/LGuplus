import pathlib
from collections import Counter
from glob import glob
import os

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torch import cuda
from torchvision import models, transforms

DIR = str(pathlib.Path(__file__).parent.resolve())
plt.rcParams['font.size'] = 14

# Initialization
MHI_DURATION = 30

COLOR = []
RADIUS = []
for idx in range(1,MHI_DURATION+1):
  ratio = (idx-1)/(MHI_DURATION-1)
  result = 5*ratio
  result += 3
  RADIUS.append(int(result))
RADIUS.reverse()

MHI_DURATION_FIRST = 15
for result_idx in range(1,MHI_DURATION_FIRST+1):
    ratio = (result_idx-1)/(MHI_DURATION_FIRST-1)
    COLOR.append((0,int(255*ratio), int(255*(1-ratio))))

MHI_DURATION_SECOND = 15
for result_idx in range(1,MHI_DURATION_SECOND+1):
    ratio = (result_idx-1)/(MHI_DURATION_SECOND-1)
    COLOR.append((int(255*ratio), int(255*(1-ratio)), 0))

LIST_COORD_RIGHT_SHOULDER = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_SHOULDER = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_CENTER_SHOULDER = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_ELBOW = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_ELBOW = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_WRIST = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_WRIST = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_HIP = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_HIP = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_CENTER_HIP = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_KNEE =[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_KNEE = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_ANKLE = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_ANKLE = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]


def imshow_tensor(image, ax=None, title=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image

save_file_name = DIR + '/models/resnet50-cuda-v2.pt'
checkpoint_path = DIR + '/models/resnet50-cuda-v2.pth'

### Whether to load on a GPU
if torch.cuda.is_available():
    load_on_gpu = True
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    load_on_gpu = True
    device = torch.device("mps")

### Load on CPU, remove if load GPU
load_on_gpu = False 
device = torch.device("cpu")


print(f'Device: {device}')

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
        # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_checkpoint(path):
    """Load a PyTorch model checkpoint

    Params
    --------
        path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """
    # Load in checkpoint
    checkpoint = torch.load(path, map_location=device)

    model = models.resnet50(pretrained=True)
    # Make sure to set parameters as not trainable
    for param in model.parameters():
        param.requires_grad = False
    model.fc = checkpoint['fc']

    # Load in the state dict
    #model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(torch.load(save_file_name, map_location=device))

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    print()

    if load_on_gpu:
        model = model.to(device)

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

model, optimizer = load_checkpoint(path=checkpoint_path)


def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path).convert('RGB')
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor

def predict(image_path, model, topk=5):
    """Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return

    Returns
        
    """

    # Convert to pytorch tensor
    img_tensor = process_image(image_path)

    # Resize
    if load_on_gpu:
        img_tensor = img_tensor.view(1, 3, 224, 224).to(device)
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)

        # Extract the actual classes and probabilities
        top_classes = [
            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
        ]
        top_p = topk.cpu().numpy()[0]

        return img_tensor.cpu().squeeze(), top_p, top_classes


def InsertCoordinate(image,landmark_pose):
  image_height, image_width, _ = image.shape 
  cv2.rectangle(image, (0,0), (image_width,image_height), (0,0,0), cv2.FILLED)

  # 좌표를 얻어옴
  RIGHT_SHOULDER = landmark_pose[12]
  RIGHT_SHOULDER_X = int(RIGHT_SHOULDER.x * image_width)
  RIGHT_SHOULDER_Y = int(RIGHT_SHOULDER.y * image_height)
  if (RIGHT_SHOULDER.visibility < 0.5):
    RIGHT_SHOULDER_X = 0
    RIGHT_SHOULDER_Y = 0
  LIST_COORD_RIGHT_SHOULDER.insert(0,(RIGHT_SHOULDER_X,RIGHT_SHOULDER_Y))
  LIST_COORD_RIGHT_SHOULDER.pop(29)

  LEFT_SHOULDER = landmark_pose[11]
  LEFT_SHOULDER_X = int(LEFT_SHOULDER.x * image_width)
  LEFT_SHOULDER_Y = int(LEFT_SHOULDER.y * image_height)
  if (LEFT_SHOULDER.visibility < 0.5):
    LEFT_SHOULDER_X = 0
    LEFT_SHOULDER_Y = 0
  LIST_COORD_LEFT_SHOULDER.insert(0,(LEFT_SHOULDER_X,LEFT_SHOULDER_Y))
  LIST_COORD_LEFT_SHOULDER.pop(29)
  
  CENTER_SHOULDER_X = int((RIGHT_SHOULDER_X+LEFT_SHOULDER_X)/2)
  CENTER_SHOULDER_Y = int((RIGHT_SHOULDER_Y+LEFT_SHOULDER_Y)/2)
  LIST_COORD_CENTER_SHOULDER.insert(0,(CENTER_SHOULDER_X,CENTER_SHOULDER_Y))
  LIST_COORD_CENTER_SHOULDER.pop(29)  

  RIGHT_ELBOW = landmark_pose[14]
  RIGHT_ELBOW_X = int(RIGHT_ELBOW.x * image_width)
  RIGHT_ELBOW_Y = int(RIGHT_ELBOW.y * image_height)
  if (RIGHT_ELBOW.visibility < 0.5):
    RIGHT_ELBOW_X = 0
    RIGHT_ELBOW_Y = 0
  LIST_COORD_RIGHT_ELBOW.insert(0,(RIGHT_ELBOW_X,RIGHT_ELBOW_Y))
  LIST_COORD_RIGHT_ELBOW.pop(29)

  LEFT_ELBOW = landmark_pose[13]
  LEFT_ELBOW_X = int(LEFT_ELBOW.x * image_width)
  LEFT_ELBOW_Y = int(LEFT_ELBOW.y * image_height)
  if (LEFT_ELBOW.visibility < 0.5):
    LEFT_ELBOW_X = 0
    LEFT_ELBOW_Y = 0
  LIST_COORD_LEFT_ELBOW.insert(0,(LEFT_ELBOW_X,LEFT_ELBOW_Y))
  LIST_COORD_LEFT_ELBOW.pop(29)

  RIGHT_WRIST = landmark_pose[16]
  RIGHT_WRIST_X = int(RIGHT_WRIST.x * image_width)
  RIGHT_WRIST_Y = int(RIGHT_WRIST.y * image_height)
  if (RIGHT_WRIST.visibility < 0.5):
    RIGHT_WRIST_X = 0
    RIGHT_WRIST_Y = 0
  LIST_COORD_RIGHT_WRIST.insert(0,(RIGHT_WRIST_X,RIGHT_WRIST_Y))
  LIST_COORD_RIGHT_WRIST.pop(29)

  LEFT_WRIST = landmark_pose[15]
  LEFT_WRIST_X = int(LEFT_WRIST.x * image_width)
  LEFT_WRIST_Y = int(LEFT_WRIST.y * image_height)
  if (LEFT_WRIST.visibility < 0.5):
    LEFT_WRIST_X = 0
    LEFT_WRIST_X = 0
  LIST_COORD_LEFT_WRIST.insert(0,(LEFT_WRIST_X,LEFT_WRIST_Y))
  LIST_COORD_LEFT_WRIST.pop(29)
  
  RIGHT_HIP = landmark_pose[24]
  RIGHT_HIP_X = int(RIGHT_HIP.x * image_width)
  RIGHT_HIP_Y = int(RIGHT_HIP.y * image_height)
  if (RIGHT_HIP.visibility < 0.5):
    RIGHT_HIP_X = 0
    RIGHT_HIP_Y = 0
  LIST_COORD_RIGHT_HIP.insert(0,(RIGHT_HIP_X,RIGHT_HIP_Y))
  LIST_COORD_RIGHT_HIP.pop(29)
  
  LEFT_HIP = landmark_pose[23]
  LEFT_HIP_X = int(LEFT_HIP.x * image_width)
  LEFT_HIP_Y = int(LEFT_HIP.y * image_height)
  if (LEFT_HIP.visibility < 0.5):
    LEFT_HIP_X = 0
    LEFT_HIP_Y = 0
  LIST_COORD_LEFT_HIP.insert(0,(LEFT_HIP_X,LEFT_HIP_Y))
  LIST_COORD_LEFT_HIP.pop(29)

  CENTER_HIP_X = int((RIGHT_HIP_X+LEFT_HIP_X)/2)
  CENTER_HIP_Y = int((RIGHT_HIP_Y+LEFT_HIP_Y)/2) 
  LIST_COORD_CENTER_HIP.insert(0,(CENTER_HIP_X,CENTER_HIP_Y))
  LIST_COORD_CENTER_HIP.pop(29)

  RIGHT_KNEE = landmark_pose[26]
  RIGHT_KNEE_X = int(RIGHT_KNEE.x * image_width)
  RIGHT_KNEE_Y = int(RIGHT_KNEE.y * image_height)
  if (RIGHT_KNEE.visibility < 0.5):
    RIGHT_KNEE_X = 0
    RIGHT_KNEE_Y = 0
  LIST_COORD_RIGHT_KNEE.insert(0,(RIGHT_KNEE_X,RIGHT_KNEE_Y))
  LIST_COORD_RIGHT_KNEE.pop(29)
  
  LEFT_KNEE = landmark_pose[25]
  LEFT_KNEE_X = int(LEFT_KNEE.x * image_width)
  LEFT_KNEE_Y = int(LEFT_KNEE.y * image_height)
  if (LEFT_KNEE.visibility < 0.5):
    LEFT_KNEE_X = 0
    LEFT_KNEE_Y = 0
  LIST_COORD_LEFT_KNEE.insert(0,(LEFT_KNEE_X,LEFT_KNEE_Y))
  LIST_COORD_LEFT_KNEE.pop(29)

  RIGHT_ANKLE = landmark_pose[28]
  RIGHT_ANKLE_X = int(RIGHT_ANKLE.x * image_width)
  RIGHT_ANKLE_Y = int(RIGHT_ANKLE.y * image_height)
  if (RIGHT_ANKLE.visibility < 0.5):
    RIGHT_ANKLE_X = 0
    RIGHT_ANKLE_Y = 0
  LIST_COORD_RIGHT_ANKLE.insert(0,(RIGHT_ANKLE_X,RIGHT_ANKLE_Y))
  LIST_COORD_RIGHT_ANKLE.pop(29)
  
  LEFT_ANKLE = landmark_pose[27]
  LEFT_ANKLE_X = int(LEFT_ANKLE.x * image_width)
  LEFT_ANKLE_Y = int(LEFT_ANKLE.y * image_height)
  if (LEFT_ANKLE.visibility < 0.5):
    LEFT_ANKLE_X = 0
    LEFT_ANKLE_Y = 0 
  LIST_COORD_LEFT_ANKLE.insert(0,(LEFT_ANKLE_X,LEFT_ANKLE_Y))
  LIST_COORD_LEFT_ANKLE.pop(29)

  
  if (LIST_COORD_RIGHT_SHOULDER[0][0]!=0 and LIST_COORD_RIGHT_ELBOW[0][0]!=0):
    # 오른쪽 어깨 - 오른쪽 팔꿈치
    cv2.line(image, (LIST_COORD_RIGHT_SHOULDER[0][0] , LIST_COORD_RIGHT_SHOULDER[0][1]) , (LIST_COORD_RIGHT_ELBOW[0][0], LIST_COORD_RIGHT_ELBOW[0][1]), (0,0,255), 1, cv2.LINE_AA)
  
  if (LIST_COORD_RIGHT_WRIST[0][0]!=0 and LIST_COORD_RIGHT_ELBOW[0][0]!=0):
    # 오른쪽 손목 - 오른쪽 팔꿈치
    cv2.line(image, (LIST_COORD_RIGHT_WRIST[0][0] , LIST_COORD_RIGHT_WRIST[0][1]) , (LIST_COORD_RIGHT_ELBOW[0][0], LIST_COORD_RIGHT_ELBOW[0][1]), (0,0,255), 1, cv2.LINE_AA)
  
  if (LIST_COORD_RIGHT_SHOULDER[0][0]!=0 and LIST_COORD_CENTER_SHOULDER[0][0]!=0):
    # 오른쪽 어깨 - 중어깨
    cv2.line(image, (LIST_COORD_RIGHT_SHOULDER[0][0] , LIST_COORD_RIGHT_SHOULDER[0][1]) , (LIST_COORD_CENTER_SHOULDER[0][0], LIST_COORD_CENTER_SHOULDER[0][1]), (0,0,255), 1, cv2.LINE_AA)

  if (LIST_COORD_CENTER_SHOULDER[0][0]!=0 and LIST_COORD_LEFT_SHOULDER[0][0]!=0):
    # 중어깨 - 왼쪽 어깨
    cv2.line(image, (LIST_COORD_CENTER_SHOULDER[0][0] , LIST_COORD_CENTER_SHOULDER[0][1]) , (LIST_COORD_LEFT_SHOULDER[0][0], LIST_COORD_LEFT_SHOULDER[0][1]), (0,0,255), 1, cv2.LINE_AA)
  
  if (LIST_COORD_LEFT_SHOULDER[0][0]!=0 and LIST_COORD_LEFT_ELBOW[0][0]!=0):
    # 왼쪽 어깨 - 왼쪽 팔꿈치
    cv2.line(image, (LIST_COORD_LEFT_SHOULDER[0][0], LIST_COORD_LEFT_SHOULDER[0][1]) , (LIST_COORD_LEFT_ELBOW[0][0], LIST_COORD_LEFT_ELBOW[0][1]), (0,0,255), 1, cv2.LINE_AA)
  
  if (LIST_COORD_LEFT_ELBOW[0][0]!=0 and LIST_COORD_LEFT_WRIST[0][0]!=0):
    # 왼쪽 팔꿈치 - 왼쪽 손목
    cv2.line(image, (LIST_COORD_LEFT_ELBOW[0][0], LIST_COORD_LEFT_ELBOW[0][1]) , (LIST_COORD_LEFT_WRIST[0][0], LIST_COORD_LEFT_WRIST[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_CENTER_SHOULDER[0][0]!=0 and LIST_COORD_CENTER_HIP[0][0]!=0):  
    # 중어꺠 - 중덩이
    cv2.line(image, (LIST_COORD_CENTER_SHOULDER[0][0] , LIST_COORD_CENTER_SHOULDER[0][1]) , (LIST_COORD_CENTER_HIP[0][0], LIST_COORD_CENTER_HIP[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_RIGHT_HIP[0][0]!=0 and LIST_COORD_CENTER_HIP[0][0]!=0):
    # 오른쪽 엉덩이 - 중덩이
    cv2.line(image, (LIST_COORD_RIGHT_HIP[0][0] , LIST_COORD_RIGHT_HIP[0][1]) , (LIST_COORD_CENTER_HIP[0][0], LIST_COORD_CENTER_HIP[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_LEFT_HIP[0][0]!=0 and LIST_COORD_CENTER_HIP[0][0]!=0):
    # 왼쪽 엉덩이 - 중덩이
    cv2.line(image, (LIST_COORD_LEFT_HIP[0][0] , LIST_COORD_LEFT_HIP[0][1]) , (LIST_COORD_CENTER_HIP[0][0], LIST_COORD_CENTER_HIP[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_RIGHT_HIP[0][0]!=0 and LIST_COORD_RIGHT_KNEE[0][0]!=0):
    # 오른쪽 엉덩이 - 오른쪽 무릎
    cv2.line(image, (LIST_COORD_RIGHT_HIP[0][0] , LIST_COORD_RIGHT_HIP[0][1]) , (LIST_COORD_RIGHT_KNEE[0][0], LIST_COORD_RIGHT_KNEE[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_RIGHT_KNEE[0][0]!=0 and LIST_COORD_RIGHT_ANKLE[0][0]!=0):
    # 오른쪽 무릎 - 오른쪽 발목
    cv2.line(image, (LIST_COORD_RIGHT_KNEE[0][0] , LIST_COORD_RIGHT_KNEE[0][1]) , (LIST_COORD_RIGHT_ANKLE[0][0], LIST_COORD_RIGHT_ANKLE[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_LEFT_HIP[0][0]!=0 and LIST_COORD_LEFT_KNEE[0][0]!=0):
    # 왼쪽 엉덩이 - 왼쪽 무릎
    cv2.line(image, (LIST_COORD_LEFT_HIP[0][0], LIST_COORD_LEFT_HIP[0][1]) , (LIST_COORD_LEFT_KNEE[0][0], LIST_COORD_LEFT_KNEE[0][1]), (0,0,255), 1, cv2.LINE_AA)
  if (LIST_COORD_LEFT_KNEE[0][0]!=0 and LIST_COORD_LEFT_ANKLE[0][0]!=0):
    # 왼쪽 무릎 - 왼쪽 발목
    cv2.line(image, (LIST_COORD_LEFT_KNEE[0][0] , LIST_COORD_LEFT_KNEE[0][1]) , (LIST_COORD_LEFT_ANKLE[0][0], LIST_COORD_LEFT_ANKLE[0][1]), (0,0,255), 1, cv2.LINE_AA)

  for idx in range(MHI_DURATION):
    cv2.circle(image, (LIST_COORD_RIGHT_SHOULDER[idx][0],LIST_COORD_RIGHT_SHOULDER[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_ANKLE[idx][0],LIST_COORD_RIGHT_ANKLE[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_ELBOW[idx][0],LIST_COORD_RIGHT_ELBOW[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_HIP[idx][0],LIST_COORD_RIGHT_HIP[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_KNEE[idx][0],LIST_COORD_RIGHT_KNEE[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_WRIST[idx][0],LIST_COORD_RIGHT_WRIST[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_SHOULDER[idx][0],LIST_COORD_LEFT_SHOULDER[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_ANKLE[idx][0],LIST_COORD_LEFT_ANKLE[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_ELBOW[idx][0],LIST_COORD_LEFT_ELBOW[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_KNEE[idx][0],LIST_COORD_LEFT_KNEE[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_WRIST[idx][0],LIST_COORD_LEFT_WRIST[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_HIP[idx][0],LIST_COORD_LEFT_HIP[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 

  cv2.circle(image, (0,0), 18, (0,0,0),cv2.FILLED, cv2.LINE_AA)

  return image

# Inference
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("sub_project/Posture/video/YoutubeTest.mp4")

with mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.5) as pose:
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret==False:
            break

        frame_process = frame.copy()
        image_height, image_width, _ = frame_process.shape

        frame_process.flags.writeable = False
        frame_process = cv2.cvtColor(frame_process, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_process)

        frame_process.flags.writeable = True
        if results.pose_landmarks:
            landmark_pose = results.pose_landmarks.landmark

            frame_process = cv2.cvtColor(frame_process, cv2.COLOR_RGB2BGR)
            frame_process = InsertCoordinate(frame_process,landmark_pose)
            input_image_cpy = frame_process.copy()
            gray_img = cv2.cvtColor(frame_process, cv2.COLOR_BGR2GRAY)

            ret, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
            inverted_binary_img = ~ binary_img

            #cv2.imshow('inverted',inverted_binary_img )
            #cv2.waitKey(0)
            contours_list, hierarchy = cv2.findContours(inverted_binary_img,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE) 

            i = 0
            print("len: ", len(contours_list))
            for c in contours_list:
                x, y, w, h = cv2.boundingRect(c)
                #print("Coordinate ", i, ": ", x, x+w, y, y+h)
                #print("Area ", i, ": ", cv2.contourArea(c))

                if (cv2.contourArea(c) > 15000  and cv2.contourArea(c) < 2000000 and w>100 and h> 100):
                    
                    cv2.rectangle(input_image_cpy, (x-50, y-50), (x + w+50, y + h+50), (0, 0, 255), 5)
                    print("Coordinate ", i, ": ", x-50, x+w+50, y-50, y+h+50)
                    print("Area ", i, ": ", cv2.contourArea(c))
                    i+=1

            cv2.imshow('Skeleton bounding box', input_image_cpy)
            cv2.waitKey(0)


        if (cv2.waitKey(10) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
