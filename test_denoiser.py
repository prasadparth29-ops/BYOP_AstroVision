import torch
import torchvision.transforms as transforms
from PIL import Image
from unet import Unet #keep unet.py in the same folder
import matplotlib.pyplot as plt

img_path = "noisy2.png"
weights_path = "epoch_best_loss34.pth"

model = Unet() 
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()

original_img = Image.open(img_path).convert('RGB')
orig_w, orig_h = original_img.size

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

input_tensor = preprocess(original_img).unsqueeze(0)

with torch.no_grad():
    b,c,h,w=input_tensor.shape
    etr = torch.empty(b).uniform_(5.0, 60.0).to('cpu')
    etr=etr.view((b,1,1,1)).expand((b,1,h//16,w//16))
    output_tensor = model(input_tensor, etr) 

output_img = transforms.ToPILImage()(output_tensor.squeeze(0))
final_output = output_img.resize((orig_w, orig_h))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(final_output)
plt.title("denoised")
plt.tight_layout()
plt.show()
