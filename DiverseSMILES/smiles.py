from gensim.models import word2vec
from rdkit.Chem import PandasTools
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
import numpy as np


df = PandasTools.LoadSDF('data/data/ames.sdf')
#df = pd.read_csv('data/data/ames.csv')
# print(df)
model = word2vec.Word2Vec.load('data/models/model_300dim.pkl')


print(df.keys())


df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], 1)), axis=1)

print(df['sentence'][0])
exit()

df['mol2vec'] = [DfVec(x) for x in sentences2vec(df['sentence'], model, unseen='UNK')]

data = np.array([x.vec for x in df['mol2vec']])
print(data)

print(data.shape)


from matplotlib import pyplot as plt
from DiverseSMILES.algorithms import mitchells_best_candidate

selection = 500
candidates = 5
num_samples = len(data)

k_inds = mitchells_best_candidate(data, selection, candidates=candidates)

#k_farthest, k_inds = incremental_farthest_search(data, selection)

farthest = data[k_inds]
mask = np.ones(num_samples, dtype=bool)
mask[k_inds] = False
leftover = data[mask]

print(farthest.shape)
print(leftover.shape)

print(k_inds)

# fig = plt.figure()
# ax = fig.add_axes([0, 0, 1, 1])
# ax.scatter(farthest[:, 0], farthest[:, 1], color='r', label='farthest')
# ax.scatter(leftover[:, 0], leftover[:, 1], color='b', label='leftover')
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_title(f'K = {selection} farthest points')
# plt.show()
