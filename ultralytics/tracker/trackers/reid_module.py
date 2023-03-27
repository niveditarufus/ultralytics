import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
from collections import OrderedDict

def image_net_preprocess(img):
    preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    return input_tensor

def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features

class ReID():
    def __init__(self, device='cuda', size = (224,224), batch_size = 16):
        super(ReID).__init__()

        self.model = models.resnet50(pretrained=True)
        # state_dict_model = torch.load("model/ft_ResNet50/net_last.pth")
        # new_state_dict = OrderedDict()
        # for k, v in state_dict_model.items():
        #         if "model." in k:
        #             name = k.replace("model.", "", 1)
        #             new_state_dict[name] = v
        # self.model.load_state_dict(new_state_dict)     
        self.model.eval().cuda().half()
        self.model = models.resnet50(pretrained=True)
        state_dict_model = torch.load("./reid/lightning_logs/version_0/checkpoints/epoch=0-step=614.ckpt")['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict_model.items():
            if "model.model." in k:
                name = k.replace("model.model.", "", 1)
                new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        
        self.embeddings = []
        self.device = device
        self.size = size
        self.batch_size = batch_size

    def inference(self, image, detections):
        H, W, C = image.shape

        patches = []
        batch_patches = []
        for i,dets in enumerate(detections):
            tlbr = dets.astype(int)
            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])
            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]
            # cv2.imshow("frame1",patch) #display in windows cv2.imshow("frame2",bb)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            patch = patch[:, :, ::-1]

            patch = cv2.resize(patch, self.size, interpolation=cv2.INTER_LINEAR)
            # patch = image_net_preprocess(patch)
            patch = torch.as_tensor(patch.transpose(2, 0, 1))
            patch = patch.to(device=self.device).half()

            patches.append(patch)

            if (i + 1) % self.batch_size == 0:
                patches = torch.stack(patches, dim=0)
                batch_patches.append(patches)
                patches = []

        if len(patches):
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)

        features = np.zeros((0, 1000))
        self.model.eval().to(self.device).half()

        for batch in batch_patches:

            # Run model
            # batch_ = torch.clone(batch)
            emb = self.model(batch)
            feat = postprocess(emb)

            features = np.vstack((features, feat))
        return features


# reid = ReID()
# reid.inference((np.random.rand(200,200,3)).astype('float32'), np.array([[10,20,30,40]]).astype('float32'))
