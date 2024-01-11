from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        # txt 파일을 한 줄씩 읽어서 이미지 경로와 레이블을 리스트에 저장합니다.
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        self.data = []
        for line in lines:
            parts = line.strip().split(',')
            image_path = parts[0]
            label = int(parts[2])
            self.data.append((image_path, label))

        print(self.data)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label