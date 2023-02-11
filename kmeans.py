import numpy as np
import pandas as pd

# membaca data dari file .xlsx
data = pd.read_excel("diabetes.xlsx")

# mengambil semua data yang akan digunakan sebagai data training
X = data.values

print("Data       : Diabetes")
print("Total Data : ",len(X), "data")
print("")
K = int(input("Cluster : "))

# validasi input
while K < 2:
    print("Invalid!")
    K = int(input("Cluster : "))


# memilih K titik awal sebagai centroid awal
centroids = X[np.random.choice(X.shape[0], K, replace=False), :]
print("CENTROIDS AWAL")
print(centroids)

# menyimpan centroid baru setiap iterasi selesai
new_centroids = np.zeros(centroids.shape)

# menyimpan titik/data ke dalam masing-masing kluster
clusters = np.zeros(X.shape[0], dtype=int)

# flag untuk memastikan bahwa proses algoritma k-means sudah selesai
done = False

# melakukan proses k-means selama centroid tidak centroidsama
iteration = 0
while not done:
    iteration += 1

    # menentukan jarak antara setiap titik dengan centroid
    distances = np.zeros((X.shape[0], K))
    for i in range(K):
        distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
    
    


    # menentukan kluster untuk setiap titik berdasarkan jarak terdekat ke centroid
    clusters = np.argmin(distances, axis=1)
    print("\n\PEMBENTUKAN KLUSTER BARU :")
    # menentukan kluster baru dari centroid baru
    for i in range(K):
        new_centroids[i] = np.mean(X[clusters == i], axis=0)
        print('C'+str(i),new_centroids[i])

    print("\n\nJARAK DENGAN CENTROIDS :\n")
    for i in range(len(distances)):
        print(i,distances[i])

    # mengecek apakah centroid baru sama dengan centroid sebelumnya
    done = np.array_equal(centroids, new_centroids)

    # mengupdate centroid
    centroids = new_centroids

    # menampilkan centroid baru setiap iterasi
    print("\nIterasi ke-{}:".format(iteration))
    print("\n\nJARAK DENGAN CENTROIDS :\n")
    for i in range(len(distances)):
        print(i,distances[i])
    print("Centroid baru:")
    for i in range(len(centroids)):
        print("C"+str(i),centroids[i])
    result = np.bincount(clusters)
    for i in range(K):
        print("Kluster ",i,": ", result[i], "data")
        
    print("")
    if done == True:
        print("Iterasi ke-{}:".format(iteration+1))
        print("\n\nJARAK DENGAN CENTROIDS :\n")
        for i in range(len(distances)):
            print(i,distances[i])
        print("Centroid baru:")
        for i in range(len(centroids)):
            print("C"+str(i),centroids[i])
        for i in range(K):
            print("Kluster ",i,": ", result[i], "data")
        print("")
        print("Centroid sudah sama dengan centroid sebelumnya, maka perulangan berhenti!")
        print("")

# menampilkan hasil akhir k-means
print("Hasil k-means:")
print("Centroid akhir:", centroids)
for i in range(K):
        print("Kluster ",i,": ", result[i], "data")
