import kagglehub
import shutil

path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
path2 = '/home/long/longdata//decentralized_dl'
shutil.move(path, path2)
print('done')