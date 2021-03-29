"""
画像を Streamlit 上で表示して、学習済みのresnet101(Pytorch)で分類
https://towardsdatascience.com/create-an-image-classification-web-app-using-pytorch-and-streamlit-f043ddf00c24

Usage:
    $ streamlit run ./app.py
"""
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image


def predict(image_path):
    resnet = models.resnet101(pretrained=True)

    # https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    resnet.eval()
    out = resnet(batch_t)

    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


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
        labels = predict(file_up)

        # print out the top 5 prediction labels with scores
        for i in labels:
            st.write("Prediction (index, name)", i[0], ",   Score: ", round(i[1], 2))


if __name__ == "__main__":
    main()
