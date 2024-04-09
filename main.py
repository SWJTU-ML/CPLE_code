from sklearn.preprocessing import StandardScaler
from src.cluster_score import cluster_accuracy
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from src.CPLE import cple
import scipy.io as scio

if __name__ == '__main__':
    # load data'
    data = scio.loadmat('data/abta.mat')
    x = data['data']
    y = data['labels'].flatten()

    ss_x = StandardScaler()
    x = ss_x.fit_transform(x)

    y_pre = cple(x,y,k=10,alpha=5,beta=5)

    NMI = normalized_mutual_info_score(y,y_pre)
    ARI = adjusted_rand_score(y,y_pre)
    ACC = cluster_accuracy(y, y_pre)
    print(NMI,ACC,ARI)

