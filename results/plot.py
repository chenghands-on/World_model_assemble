import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# epsoids = 30
# col_name = 'Evaluation/AverageDiscountedReturn'
# path1='results/622/sgd.csv'
# path2='results/622/drsom.csv'
# path3='results/622/nsgd_0.001.csv'


# sgd = pd.read_csv(path1)
# drsom=pd.read_csv(path2)
# nsgd1=pd.read_csv(path3)


# plt.figure(dpi=300,figsize=(8,4))
# plt.title('')
# plt.plot(np.arange(1, epsoids + 1), sgd['all_train_loss'], label='sgd,lr=0.001')
# plt.plot(np.arange(1, epsoids + 1), drsom['all_train_loss'], label='drsom,gamma=1e-3')
# plt.plot(np.arange(1, epsoids + 1), nsgd1["all_train_loss"], label='nsgd,lr=0.001')


# plt.legend()
# plt.ylabel('Loss')
# plt.xlabel('Epoch #')
# plt.savefig('./results/fig/269218.jpg')
# plt.show()

# epsoids = 15
path1='./run-dmc_walker_walk_v2_2-tag-scalars_train_return.csv'
path2='./run-dmc_walker_walk_v3-tag-scalars_train_return.csv'



v2 = pd.read_csv(path1)
v3=pd.read_csv(path2)
# drsom2=pd.read_csv(path6)


plt.figure(dpi=300,figsize=(8,4))
plt.title('')
# plt.plot(np.arange(1, epsoids + 1), sgd1["all_val_mae"], label='sgd,lr=0.001')
# plt.plot(np.arange(1, epsoids + 1), nsgd2["all_val_mae"], label='nsgd,lr=0.001')
plt.plot(v2["Step"][:50], v2["Value"][0:50], label='DreamerV2')
plt.plot(v3["Step"][:50], v3["Value"][0:50], label='DreamerV3')
# plt.plot(np.arange(1, epsoids + 1), drsom1["all_val_mae"], label='drsom,new para')
# plt.plot(np.arange(1, epsoids + 1), drsom2["all_val_mae"], label='drsom,gamma=1e-3')



plt.legend()
plt.ylabel('Return')
plt.xlabel('Env Steps')
plt.savefig('./fig/dmc_walk_walker.jpg')
plt.show()