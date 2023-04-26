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
r2u_mapping = torch.load(script_path + "/weights/r2u_mapping.pt")

WITH_LABEL = False

OLD_CLASSIFIER_PATH = "/weights/supcon_big_classifier.truegenetrees.5.pt"
@lru_cache(None)
def obtain_classifer(path="", dim=5):
    classifier = ClassifierHead(dim) if WITH_LABEL else ClassifierHead()
    classifier.load_state_dict(torch.load(script_path + path, "cpu"))
    classifier.eval()
    print(classifier)
    return classifier

OLD_ENCODER_PATH = "/weights/supcon.99.pt"
@lru_cache(None)
def obtain_encoder(path="/weights/supcon.99.pt", dim=15*11):
    encoder = Encoder(dim)
    encoder.load_state_dict(torch.load(script_path + path, "cpu"))
    encoder.eval()
    print(encoder)
    return encoder

@lru_cache(None)
def basis_vector(i):
    v = np.zeros(15)
    v[i] = 1
    return torch.tensor(v).float()

def predict_ils(u : np.ndarray):
    encoder = obtain_encoder("/weights/ils/encoder.pt")
    classifier = obtain_classifer("/weights/ils/classifier.pt")
    u_encoded = encoder(torch.tensor(u).float())
    return classifier(u_encoded)

@lru_cache(None)
def obtain_transl():
    return torch.load(script_path + "/weights/table5_to_contrastive.pt")

def ils_cost_between(unrooted_id : int, ils_signal : np.ndarray, temperature : float = 1.) -> np.ndarray:
    criterion = torch.nn.CrossEntropyLoss()
    transl = obtain_transl()
    candidate_topologies_ix = transl[u2r_mapping[unrooted_id]]
    pred = predict_ils(ils_signal) / temperature
    cost = np.asarray([criterion(pred, c).detach().item() for c in candidate_topologies_ix])
    return cost

def gdl_predict(co):
    encoder = obtain_encoder("/weights/gdl/encoder.pt", dim=15)
    classifier = obtain_classifer("/weights/gdl/classifier.pt")
    u_encoded = encoder(torch.tensor(co).float())
    return classifier(u_encoded)

def joint_predict(ils_signal, gdl_signal):
    encoder_ils = obtain_encoder("/weights/joint/encoder_ils.pt")
    encoder_gdl = obtain_encoder("/weights/joint/encoder_gdl.pt", dim=15)
    classifier = obtain_classifer("/weights/joint/classifier.pt", dim=10)
    u_encoded_ils = encoder_ils(torch.tensor(ils_signal).float())
    u_encoded_gdl = encoder_gdl(torch.tensor(gdl_signal).float())
    u_encoded = torch.cat([u_encoded_ils, u_encoded_gdl], dim=-1)
    return classifier(u_encoded)

def gdl_cost_between(unrooted_id : int, gdl_signal : np.ndarray) -> np.ndarray:
    criterion = torch.nn.CrossEntropyLoss()
    transl = obtain_transl()
    candidate_topologies_ix = transl[u2r_mapping[unrooted_id]]
    pred = gdl_predict(gdl_signal)
    cost = np.asarray([criterion(pred, c).detach().item() for c in candidate_topologies_ix])
    return cost

def joint_cost_between(unrooted_id : int, ils_signal : np.ndarray, gdl_signal: np.ndarray) -> np.ndarray:
    if ils_signal.sum() == 0:
        return gdl_cost_between(unrooted_id, gdl_signal)
    criterion = torch.nn.CrossEntropyLoss()
    transl = obtain_transl()
    candidate_topologies_ix = transl[u2r_mapping[unrooted_id]]
    pred = joint_predict(ils_signal, gdl_signal)
    cost = np.asarray([criterion(pred, c).detach().item() for c in candidate_topologies_ix])
    return cost