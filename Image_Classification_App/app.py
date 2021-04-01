"""
画像を Streamlit 上で表示して、学習済みの mobilenet_v2(Pytorch)で分類
https://towardsdatascience.com/create-an-image-classification-web-app-using-pytorch-and-streamlit-f043ddf00c24

Usage:
    $ conda activate lightning
    $ streamlit run ./app.py
"""
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

sys.path.append("./gradcam_plus_plus-pytorch")
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


@st.cache(allow_output_mutation=True)
def load_model():
    # model = models.resnet101(pretrained=True)
    model = models.mobilenet_v2(pretrained=True)
    return model


def preprocessing_image(image_pil_array: "PIL.Image"):
    # https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose(
        [
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch_t = torch.unsqueeze(transform(image_pil_array), 0)
    return batch_t


def predict(image_path):
    img = Image.open(image_path)
    batch_t = preprocessing_image(img)

    model = load_model()
    model.eval()
    out = model(batch_t)

    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    top5 = [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

    result_pp = _gradcam(model, img)

    return top5, result_pp


def _gradcam(model, image_pil_array: "PIL.Image"):
    """grad-cam++"""
    target_layer = model.features  # mobilenet_v2

    img_cv2 = pil2cv(image_pil_array)

    img = img_cv2 / 255
    a_transform = A.Compose([A.Resize(224, 224, p=1.0), ToTensorV2(p=1.0)], p=1.0)
    img = a_transform(image=img)["image"].unsqueeze(0)

    a_transform = A.Compose(
        [
            A.Resize(224, 224, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
    batch_t = a_transform(image=img_cv2)["image"].unsqueeze(0)

    gradcam_pp = GradCAMpp(model, target_layer)
    mask_pp, logit = gradcam_pp(batch_t)
    heatmap_pp, result_pp = visualize_cam(mask_pp, img)
    return result_pp


def pil2cv(image):
    """ PIL型 -> OpenCV型
    https://qiita.com/derodero24/items/f22c22b22451609908ee"""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def main():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.title("Simple Image Classification App")
    st.write("")

    file_up = st.file_uploader("Upload an image", type="jpg")

    if file_up is not None:
        image = Image.open(file_up)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        st.write("")
        st.write("Just a second...")
        labels, result_pp = predict(file_up)
        st.image(
            result_pp.numpy().transpose(1, 2, 0),
            caption="Model Attention(Grad-CAM++).",
            use_column_width=True,
        )

        # print out the top 5 prediction labels with scores
        for i in labels:
            st.write("Prediction (index, name)", i[0], ",   Score: ", round(i[1], 2))
    else:
        img_url = "https://github.com/riron1206/streamlit_app/blob/master/Image_Classification_App/image/dog.jpg?raw=true"
        st.image(
            img_url,
            caption="Sample Image. Please download and upload.",
            use_column_width=True,
        )


if __name__ == "__main__":
    main()
