import kagglehub
import shutil

path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
path2 = '/Users/longle/study-at-epita/intro to blockchain/decentralized_dl'
shutil.move(path, path2)
print('done')