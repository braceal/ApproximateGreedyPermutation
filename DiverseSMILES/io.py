import json
import pandas as pd


def smiles_from_kbase_fba_model(in_path, out_path=''):
    """
    Returns a list of smiles strings given the KBase
    FBA model object. Optionally writes the smiles to
    a csv file with a single column named 'smiles'.

    Parameters
    ----------
    in_path : str
            File path to KBase FBA object json file

    out_path : str, optional
            File path to output csv file containing smiles

    Returns
    -------
    list : list of smiles strings
    """

    # Read KBase FBA obj
    with open(in_path) as f:
        obj = json.load(f)

    # Retrieve smiles strings from FBA object
    smiles = [compound['smiles']
              for compound in obj['modelcompounds']
              if compound['smiles'] != 'none']

    if out_path:
        pd.Series(smiles).to_csv(out_path, header=['smiles'], index=False)

    return smiles


def read_smiles(in_path):
    """
    Returns a list of smiles strings. Assumes input csv file
    was written by smiles_from_kbase_fba_model or otherwise
    has a single column named 'smiles'.

    Parameters
    ----------
    in_path : str
            File path to smiles csv file

    Returns
    -------
    list : list of smiles strings
    """
    return pd.read_csv(in_path)['smiles'].tolist()


if __name__ == '__main__':
    #smiles_from_kbase_fba_model('data/31.json', 'data/MMSyn3_repair_smiles.csv')
    # read_smiles('data/MMSyn3_repair_smiles.csv')
    pass
