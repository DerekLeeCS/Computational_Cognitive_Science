import numpy as np
from scipy import sparse

na = 5
nb = 6
Wb = 7

WnewBunScram = sparse.kron(np.eye(na), Wb)
sInd = np.transpose(np.reshape(np.arange(1,na*nb+1), (nb,na), 'F'))
sInd = np.reshape(sInd,(na*nb,1),'F')
#print(WnewBunScram)
print(WnewBunScram[2:5,3:5])