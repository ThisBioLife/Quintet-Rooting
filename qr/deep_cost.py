import torch
import numpy as np
import os
from qr.adr_theory import u2r_mapping

script_path = os.path.realpath(__file__).rsplit("/", 1)[0]
u_enc = torch.load(script_path + "/u_encoder.pt")
tree_enc = torch.load(script_path + "/tree_encoder.pt")
rooted_topologies = torch.load(script_path + "/rooted_topologies.pt")

def cost_between(unrooted_id : int, u : np.ndarray) -> np.ndarray:
    candidate_topologies = rooted_topologies[u2r_mapping[unrooted_id]]
    u_encoded = u_enc(torch.tensor(u).float())
    encoded_candidate_topologies = tree_enc(candidate_topologies)
    return ((encoded_candidate_topologies - u_encoded) ** 2).sum(axis=1).detach().numpy()