import pandas as pd
import numpy as np

# --- FUNGSI MATH menghitung entrophy dan gain ---
def hitung_entropy(kolom_target):
    counts = kolom_target.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts))

def hitung_gain(df, fitur, target='Play'):
    entropy_total = hitung_entropy(df[target])
    values = df[fitur].unique()
    entropy_atribut = 0
    for v in values:
        subset = df[df[fitur] == v]
        entropy_atribut += (len(subset) / len(df)) * hitung_entropy(subset[target])
    return entropy_total - entropy_atribut

# --- LOGIKA ID3 (Membangun Pohon) ---
def buat_pohon(df, fitur_tersisa, target='Main'):
    target_data = df[target]
    
    # Base Case 1: Jika semua data punya jawaban yang sama (murni)
    if len(target_data.unique()) <= 1:
        return target_data.iloc[0]
    
    # Base Case 2: Jika fitur habis tapi data belum murni, ambil suara terbanyak
    if not fitur_tersisa:
        return target_data.mode()[0]
    
    # Pilih fitur terbaik berdasarkan Gain tertinggi
    gain_list = [hitung_gain(df, f, target) for f in fitur_tersisa]
    fitur_terbaik = fitur_tersisa[np.argmax(gain_list)]
    
    # Buat struktur pohon (Node)
    pohon = {fitur_terbaik: {}}
    
    # Buang fitur yang sudah dipakai dari daftar
    fitur_baru = [f for f in fitur_tersisa if f != fitur_terbaik]
    
    # Buat cabang untuk setiap nilai unik di fitur terbaik
    for nilai in df[fitur_terbaik].unique():
        subset = df[df[fitur_terbaik] == nilai]
        # Rekursif: panggil lagi fungsi ini untuk data di cabang tersebut
        pohon[fitur_terbaik][nilai] = buat_pohon(subset, fitur_baru, target)
        
    return pohon

# --- EKSEKUSI ---
# Membaca dari CSV
df = pd.read_csv('data\data.csv')

# Daftar fitur awal (semua kolom kecuali target)
fitur_awal = list(df.columns[:-1])

# Bangun Pohon
hasil_pohon = buat_pohon(df, fitur_awal)

# Tampilkan Struktur Pohon

def print_tree_cakep(pohon, indent="  "):
    if not isinstance(pohon, dict):
        print(f" --> {pohon}")
    else:
        fitur = list(pohon.keys())[0]
        print(f"\n{indent}[ {fitur} ]")
        for nilai, sub in pohon[fitur].items():
            print(f"{indent}  |-- {nilai}", end="")
            print_tree_cakep(sub, indent + "    ")

print("========================")
print("STRUKTUR POHON KEPUTUSAN:")
print_tree_cakep(hasil_pohon)
print("")
print("========================")
