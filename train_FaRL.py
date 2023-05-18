from PIL import Image
import pandas as pd
import clip
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models
from tqdm import tqdm

class FairFaceDataset(Dataset):
    def __init__(self, csv_path, is_train=False):
        super().__init__()
        self.data = pd.read_csv(csv_path)
        self.output_names = ['0-2', '10-19', '20-29', '3-9', '30-39', '40-49', '50-59', '60-69', 'more than 70', \
                             'Female', 'Male', \
                            'Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']
        # self.output_names = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70', \
        #                      'Male','Female' , \
        #                     'White', 'Black',  'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian',  'Middle Eastern']
        transform_list = [
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if is_train:
            transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomRotation(20),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.2), interpolation=3)
            ] + transform_list[2:]
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        label = [0] * len(self.output_names)
        for name in ["age", "gender", "race"]:
            prop = item[name]
            try:
                i = self.output_names.index(prop)
                label[i] = 1
            except ValueError:
                continue
        
        # Read image
        image = Image.open(item.file)
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class FaRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = clip.load("ViT-B/16", device="cuda")[0]
        farl_state = torch.load("weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth")
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
        self.head.apply(init_weights)

    def freeze(self, mode=True):
        self.extractor.train(False)
        self.head.train(mode)
        for param in self.extractor.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # with torch.no_grad():
        feat = self.extractor.encode_image(x).float()
        out = self.head(feat)
        return out

def custom_loss(logits, labels):
    age_group = logits[:, :9]
    age_label = labels[:, :9].argmax(1)
    age_loss = nn.functional.cross_entropy(age_group, age_label)

    gender_group = logits[:, 9:11]
    gender_label = labels[:, 9:11].argmax(1)
    gender_loss = nn.functional.cross_entropy(gender_group, gender_label)

    race_group = logits[:, 11:]
    race_label = labels[:, 11:].argmax(1)
    race_loss = nn.functional.cross_entropy(race_group, race_label)

    return age_loss + gender_loss + race_loss
    

if __name__ == "__main__":
    # Define hyperparams
    n_epochs = 30
    learning_rate = 0.005
    batch_size = 128
    num_workers = 8
    log_freq = 50
    
    # Define dataset
    train_dataset = FairFaceDataset("data/FairFace/fairface_label_train.csv", True)
    val_dataset = FairFaceDataset("data/FairFace/fairface_label_val.csv", False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Define model
    model = FaRL().cuda()
    model.train()

    # Define Optimizer, Loss, LRScheduler
    train_params = [p for p in model.parameters() if p.requires_grad]
    optim = SGD(train_params, lr=learning_rate, momentum=0.9)
    n_iterations = len(train_dataloader) * n_epochs
    lr_scheduler = CosineAnnealingLR(optim, n_iterations, eta_min=1e-8, verbose=False)
    criterion = nn.BCEWithLogitsLoss()
    print("Number of training parameters:", sum([p.numel() for p in train_params]))
    
    best_acc = 0
    # Train
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        
        # Train 1 epoch
        model.train()
        if epoch < 10:
            model.freeze()
        if epoch == 10:
            print("Finetuning the whole network")
            train_params = [p for p in model.parameters() if p.requires_grad]
            optim = SGD(train_params, lr=learning_rate * 0.1, momentum=0.9)
            n_iterations = len(train_dataloader) * n_epochs
            lr_scheduler = CosineAnnealingLR(optim, n_iterations, eta_min=1e-8, verbose=False)
            criterion = nn.BCEWithLogitsLoss()
            print("Number of training parameters:", sum([p.numel() for p in train_params]))
        for b_i, (b_image, b_label) in enumerate(train_dataloader):
            b_image = b_image.cuda()
            b_label = b_label.cuda()

            optim.zero_grad()
            logits = model(b_image)
            # loss = criterion(logits, b_label)
            loss = custom_loss(logits, b_label)
            loss.backward()
            optim.step()
            lr_scheduler.step()

            if b_i % log_freq == 0:
                lr = lr_scheduler.get_last_lr()[0]
                print("Iter {}/{} - LR {:.8f}: Loss: {:.4f}".format(b_i, len(train_dataloader), lr, loss.item()))
            
        # Eval 1 epoch
        print("Evaluating")
        model.eval()
        correct_age = 0
        correct_gender = 0
        correct_race = 0
        for b_i, (b_image, b_label) in enumerate(val_dataloader):
            b_image = b_image.cuda()
            b_label = b_label.cuda()
            with torch.no_grad():
                logits = model(b_image)

            # Compute accuracy
            age_group = logits[:, :9].softmax(dim=-1).argmax(1)
            age_label = b_label[:, :9].argmax(1)
            gender_group = logits[:, 9:11].softmax(dim=-1).argmax(1)
            gender_label = b_label[:, 9:11].argmax(1)
            race_group = logits[:, 11:].softmax(dim=-1).argmax(1)
            race_label = b_label[:, 11:].argmax(1)
            correct_age += (age_group == age_label).sum()
            correct_gender += (gender_group == gender_label).sum()
            correct_race += (race_group == race_label).sum()
            if b_i % log_freq == 0:
                print("Iter {}/{}".format(b_i, len(val_dataloader)))
        correct_sample = correct_age + correct_gender + correct_race
        correct_sample = correct_sample.item() / (3 * len(val_dataset))
        print("Accuracy: {:.2f} %".format(correct_sample * 100))
        print("Age accuracy: {:.2f} %".format(correct_age / len(val_dataset) * 100))
        print("Gender accuracy: {:.2f} %".format(correct_gender / len(val_dataset) * 100))
        print("Race accuracy: {:.2f} %".format(correct_race / len(val_dataset) * 100))
        import pdb; pdb.set_trace()
        if correct_sample > best_acc:
            print("Saving the best checkpoint")
            best_acc = correct_sample
            torch.save(model.state_dict(),"weights/FaRL_FairFace_best.pth")
        else:
            print("No improvement - best accuracy: {:.2f} %".format(best_acc * 100))
        print()
        

    


        