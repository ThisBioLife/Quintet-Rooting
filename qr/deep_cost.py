import torch
import numpy as np
import os
from qr.adr_theory import u2r_mapping
from qr.models import *
from functools import lru_cache
import torch.functional as F

script_path = os.path.realpath(__file__).rsplit("/", 1)[0]
u_enc = torch.load(script_path + "/u_encoder.pt")
tree_enc = torch.load(script_path + "/tree_encoder.pt")
rooted_topologies = torch.load(script_path + "/rooted_topologies.pt")

@lru_cache(None)
def obtain_classifer():
    classifier = ClassifierHead()
    classifier.load_state_dict(torch.load(script_path + "/weights/supcon_big_classifier.true.finetuned.pt"))
    classifier.eval()
    return classifier

@lru_cache(None)
def obtain_encoder():
    encoder = Encoder()
    encoder.load_state_dict(torch.load(script_path + "/weights/supcon_big_encoder.true.finetuned.pt"))
    encoder.eval()
    return encoder

def predict(u : np.ndarray):
    encoder = obtain_encoder()
    classifier = obtain_classifer()
    u_encoded = encoder(torch.tensor(u).float())
    return classifier(u_encoded)

@lru_cache(None)
def obtain_transl():
    return torch.load(script_path + "/weights/table5_to_contrastive.pt")

def cost_between(unrooted_id : int, u : np.ndarray, temperature : float = 1.) -> np.ndarray:
    criterion = torch.nn.CrossEntropyLoss()
    transl = obtain_transl()
    candidate_topologies_ix = transl[u2r_mapping[unrooted_id]]
    pred = predict(u) / temperature
    cost = np.asarray([criterion(pred, c).detach().item() for c in candidate_topologies_ix])
    return cost