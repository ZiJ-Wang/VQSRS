import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from model.vqsrs import Model_vqsrs
from PIL import Image
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = "/data/wzj/workend/result/vq2_srh_300_rgb_3xb16_151e_256idx/p/checkpoint0150.pth"
model = Model_vqsrs()
model_dict = model.state_dict()
state = torch.load(model_path)
state_dict = state['model']
for k in list(state_dict.keys()):
    # retain only encoder_q up to before the embedding layer
    if k.startswith("module.") :
        state_dict[k[len("module.") :]] = state_dict[k]
        del state_dict[k]
pre = {k: v for k, v in state_dict.items() if k in model_dict}
model_dict.update(pre)
model.load_state_dict(model_dict)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

transform = transforms.Compose([
    #transforms.Resize(300),  
    transforms.ToTensor()
])

image_folder = "/data/wzj/cytoself-torch/VQSRS/heatmap/t"  
image_files = os.listdir(image_folder)
idx = {0: 'hgg', 1: 'lgg', 2: 'mening', 3: 'metast', 4: 'normal', 5: 'pituita', 6: 'schwan'}

results = []

for filename in tqdm(image_files, desc="Processing images", ncols=100):
    if filename == 'SRS.png':
        continue

    image_path = os.path.join(image_folder, filename)
    
    image = Image.open(image_path)
    image = transform(image)  
    image = image.unsqueeze(0)  

    with torch.no_grad():  
        outputs = model(image.to(device))  

    # Get the predicted class and confidence with the maximum value
    confidence, predicted_class = torch.max(F.softmax(outputs[3], dim=1), 1)
    confidence = confidence.cpu().numpy()[0]  
    predicted_class = predicted_class.cpu().numpy()[0]  
    predicted_class_name = idx[predicted_class]

    results.append({
        'filename': filename,
        'predicted_class': predicted_class_name,
        'confidence': confidence
    })

df = pd.DataFrame(results)

df.to_csv('./predictions.csv', index=False)

print("Predictions saved to predictions.csv")