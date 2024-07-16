import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image, ImageDraw
import firebase_admin
from firebase_admin import credentials, messaging, db
from classifier import ParkingSpotClassifier


def convert_to_rgb(image):
    if image.mode == 'L':
        image = image.convert('RGB')
    return image

transform = transforms.Compose([
    transforms.Lambda(convert_to_rgb),
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([2.8643e-07, 2.7963e-07,2.4438e-07], [1.2124e-07, 1.1815e-07, 1.2416e-07])
])

current_dir = os.path.dirname(__file__ )

device = 1 if torch.cuda.is_available() else None

bounding_boxes_file = os.path.join(current_dir, 'camera1_slot.csv')
bounding_boxes_df = pd.read_csv(bounding_boxes_file)

# Initialize the model
model = ParkingSpotClassifier()

# Load the model 
model_path = os.path.join(current_dir, 'parking_slot_classifier_mobilenetv3_modified.pth')
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode
if torch.cuda.is_available():
    model = model.to('cuda')
    
    
# Initialize Firebase app 
cred = credentials.Certificate(os.path.join(current_dir, "espark-7ad35-firebase-adminsdk-a4nee-dad45e3537.json"))
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://espark-7ad35-default-rtdb.firebaseio.com'
})
ref = db.reference('/parking/p1/slot')
slots = ref.get()

cap = cv2.VideoCapture(os.path.join(current_dir, 'input_video_camera1.mp4'))

scale_factor_x = 1000 / 2592
scale_factor_y = 750 / 1944
batch_size = 35

paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        patches = {}
        batch_patches = []

        for idx, row in bounding_boxes_df.iterrows():
            slot_id = row['SlotId']
            x_large, y_large, w_large, h_large = int(row['X']), int(row['Y']), int(row['W']), int(row['H'])

            x_small = int(x_large * scale_factor_x)
            y_small = int(y_large * scale_factor_y)
            w_small = int(w_large * scale_factor_x)
            h_small = int(h_large * scale_factor_y)

            left = x_small
            upper = y_small
            right = x_small + w_small
            lower = y_small + h_small

            patch = image.crop((left, upper, right, lower))
            patch = transform(patch)
            patch = patch.unsqueeze(0)
            
            if torch.cuda.is_available():
                patch = patch.to('cuda')

            patches[slot_id] = (patch, (left, upper, right, lower))
            batch_patches.append(patch)

        batch_patches_tensor = torch.cat(batch_patches, dim=0)
        outputs = model(batch_patches_tensor).squeeze()

        draw = ImageDraw.Draw(image)
        i = 0
        statuses = {}

        for slot_id, (slot_data, (left, upper, right, lower)) in patches.items():
            output = outputs[i]
            i += 1
            prediction = torch.sigmoid(output).item()

            if prediction >= 0.5:
                status = 1
                label = 'Occupied'
                color = (255, 0, 0)
            else:
                status = 0
                label = 'Available'
                color = (0, 255, 0)

            draw.rectangle([(left, upper), (right, lower)], outline=color, width=2)

            text = f'{slot_id}: {label}'
            draw.text((left, upper), text, fill="white")
            
            statuses[f's{slot_id}'] = {'status': status, 'type': 'white', 'value': str(slot_id)}

        frame_with_annotations = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        cv2.imshow('Annotated Video', frame_with_annotations)
        
        ref = db.reference('/parking/p1/slot')
        ref.update(statuses)
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused

cap.release()
cv2.destroyAllWindows()


