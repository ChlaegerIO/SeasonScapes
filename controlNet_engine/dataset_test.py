import matplotlib.pyplot as plt
from dataset import MyDataset

dataset = MyDataset(train_dir='r1SceneDepth')
print(len(dataset))

item = dataset[10]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

# plot images
# renormalize to 0, 1 from -1, 1
jpg = (jpg + 1) / 2
plt.figure()
plt.imshow(jpg)
plt.title('jpg')
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(hint)
plt.title('hint')
plt.axis('off')
plt.show()
