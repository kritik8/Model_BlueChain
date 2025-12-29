from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    return transform(img).unsqueeze(0)
