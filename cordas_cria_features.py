
import os
import numpy as np
import librosa
import librosa.display

# ------------------------------
# Configurações principais
# ------------------------------
SR = 22050       # taxa de amostragem
N_FFT = 2048     # tamanho da FFT
HOP_LENGTH = 512 # avanço entre janelas
N_MELS = 40      # número de filtros Mel

# Pasta onde estão os arquivos .wav
# Exemplo: "dados_cordas/" ou "." se estiver no diretório atual
BASE_DIR = "/content/drive/MyDrive/cordas_amostras"

# ------------------------------
# Funções auxiliares
# ------------------------------

def carregar_limpar_audio(path, sr=SR, top_db=40):
    """
    Carrega o áudio, remove silêncio no início e fim,
    e normaliza o sinal para o intervalo aproximado [-1, 1].
    """
    y, sr = librosa.load(path, sr=sr)

    # remove silêncio (início e fim)
    y_trim, _ = librosa.effects.trim(y, top_db=top_db)

    # normaliza para [-1, 1]
    max_abs = np.max(np.abs(y_trim)) + 1e-9
    y_trim = y_trim / max_abs

    return y_trim, sr

def espectro_fft_medio(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Calcula o espectro médio em frequência usando STFT + média no tempo.
    Retorna um vetor em dB e o vetor de frequências correspondente.
    """
    # STFT
    X = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')

    # potência do espectro
    mag = np.abs(X)**2

    # média ao longo do tempo (eixo 1 = frames no tempo)
    mag_mean = np.mean(mag, axis=1)

    # converte para dB (relativo ao máximo da própria amostra)
    mag_db = librosa.power_to_db(mag_mean, ref=np.max)

    # vetor de frequências (0 até sr/2)
    freqs = np.linspace(0, sr/2, len(mag_mean))

    return freqs, mag_db

def espectro_mel_medio(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Calcula o espectrograma Mel médio (banco de filtros na escala Mel).
    Retorna um vetor em dB e as frequências centrais dos filtros Mel.
    """
    S_mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0  # potência da STFT
    )

    # média no tempo (eixo 1 = frames)
    S_mel_mean = np.mean(S_mel, axis=1)

    # converte para dB (relativo ao máximo)
    S_mel_db = librosa.power_to_db(S_mel_mean, ref=np.max)

    # frequências centrais dos filtros Mel
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr/2)

    return mel_freqs, S_mel_db

# ------------------------------
# Loop principal: gera o dataset
# ------------------------------

# Vamos guardar as features de todas as amostras
X_fft_list = []
X_mel_list = []
y_labels = []

# Supondo nomes do tipo: cordaN_M.wav, com N=1..6, M=1..30
for n in range(1, 7):        # corda 1 a 6
    for m in range(1, 31):   # amostra 1 a 30
        filename = f"corda{n}_{m}.wav"
        filepath = os.path.join(BASE_DIR, filename)

        if not os.path.isfile(filepath):
            print(f"AVISO: arquivo não encontrado -> {filepath}")
            continue

        print(f"Processando {filepath} ...")

        # 1) Carrega, remove silêncio e normaliza
        y, sr = carregar_limpar_audio(filepath, sr=SR)

        # 2) FFT média
        freqs_fft, fft_db = espectro_fft_medio(y, sr)

        # 3) Mel média
        mel_freqs, mel_db = espectro_mel_medio(y, sr)

        # Guarda as features em listas
        X_fft_list.append(fft_db)
        X_mel_list.append(mel_db)

        # Rótulo correspondente à corda n
        y_labels.append(n)

# Converte listas em arrays
X_fft = np.array(X_fft_list)    # shape: (N_amostras, N_fft/2+1)
X_mel = np.array(X_mel_list)    # shape: (N_amostras, N_mels)
y_labels = np.array(y_labels)   # shape: (N_amostras,)

print("Formatos das matrizes:")
print("X_fft:", X_fft.shape)
print("X_mel:", X_mel.shape)
print("y_labels:", y_labels.shape)

# ------------------------------
# Salvando os dados em disco
# ------------------------------

np.save("X_fft.npy", X_fft)
np.save("X_mel.npy", X_mel)
np.save("y_labels.npy", y_labels)

print("Arquivos salvos: X_fft.npy, X_mel.npy, y_labels.npy")
