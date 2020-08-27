from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec
from gensim.models import word2vec
from rdkit import Chem


def embed_smiles(smiles):
    """
    Given a list of smiles string, computes a forward
    pass through the word2vec model pretrained on smiles
    strings and returns the embedding vectors.

    Parameters
    ----------
    smiles : list, str
        List of smiles strings or single smile string

    Returns
    -------
    list : embedded vectors, 1 for each element in smiles 
    """

    # Accept single smiles string
    if isinstance(smiles, str):
        smiles = [smiles]

    # Load pretrained model weigts
    model = word2vec.Word2Vec.load('data/models/model_300dim.pkl')

    # 
    mols = [Chem.MolFromSmiles(i) for i in smiles]

    print(mols)
    # TODO: see how mol2alt_sentence works
    sentences = [sentences2vec(MolSentence(mol2alt_sentence(m, 1)), model, unseen='UNK') for m in mols]
    return sentences


def embed_single_smiles(smiles):
    model = word2vec.Word2Vec.load('data/models/model_300dim.pkl')
    mol = Chem.MolFromSmiles(smiles)
    sentences = sentences2vec(MolSentence(mol2alt_sentence(mol, 1)), model, unseen='UNK')
    return sentences


if __name__ == '__main__':
    smiles_embedded = embed_smiles(["c1ccccc1", "O=C1CCCC2=C1C1(CCS(=O)(=O)C1)N=C(Nc1nc3ccccc3o1)N2",
                                    "CN[C@@H]1C[C@H]2O[C@@](C)([C@@H]1OC)n1c3ccccc3c3c4CNC(=O)c4c4c5ccccc5n2c4c13"])
    print(smiles_embedded[0].shape)
    print(len(smiles_embedded))
