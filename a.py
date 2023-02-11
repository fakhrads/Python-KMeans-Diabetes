import numpy as np
import pandas as pd

# membaca data
data = pd.read_excel("diabetes.xlsx")

# mengubah data menjadi numpy array
data = np.array(data)

# memisahkan atribut dan kelas
atribut = data[:,:-1]
kelas = data[:,-1]

# membuat centroid
centroid = np.zeros((2, 8))

# memberikan nilai random pada centroid
centroid[0,:] = np.random.randint(0, 10, 8)
centroid[1,:] = np.random.randint(10, 20, 8)

# melakukan proses kmeans
while True:
    # membuat cluster baru
    cluster = np.zeros((len(atribut),))
    for i in range(len(atribut)):
        jarak1 = np.linalg.norm(atribut[i,:] - centroid[0,:])
        jarak2 = np.linalg.norm(atribut[i,:] - centroid[1,:])
        
        if jarak1 < jarak2:
            cluster[i] = 0
        else:
            cluster[i] = 1
    
    # menghitung centroid baru
    centroid_baru = np.zeros((2, 8))
    for i in range(2):
        data_cluster = atribut[cluster == i]
        centroid_baru[i,:] = np.mean(data_cluster, axis = 0)
        print("Centroid : ", centroid_baru)
    
    # cek apakah centroid baru sama dengan centroid lama
    if np.array_equal(centroid, centroid_baru):
        break
    else:
        centroid = centroid_baru

# menghitung akurasi
akurasi = np.sum(cluster == kelas) / len(kelas)
print("Akurasi: ", akurasi)
