import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.datasets import ImageNet

def senet_train():
    # wandb로 로그관리하기
    wandb.init(project='senet_study')
    wandb.login()






