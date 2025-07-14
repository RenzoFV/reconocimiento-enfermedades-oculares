# download_models.py
import os
import gdown

MODELOS = {
    "eye_disease_model.h5": "https://drive.google.com/uc?id=1ijFPQkxRTA_IpOcMtDoWIZgofKh-6uaT",
    "mobilenet_individual_model.h5": "https://drive.google.com/uc?id=11pCfoCEMHWNhwCvJJYJ_iSaEJGdaPjxW",
    "efficientnet_individual_model.h5": "https://drive.google.com/uc?id=1sXx9znd_GFPVaE9s9G5nb2HSCMKk59Y4",
    "resnet_individual_model.h5": "https://drive.google.com/uc?id=1HW6_4mWYjrWTjPP28PGeMZcDtovYV6ah",
    "ensemble_mobilenet_model.h5": "https://drive.google.com/uc?id=1h6ZgE5NtUdzJUjQcS0ilNQt9iokcNrs0",
    "ensemble_efficientnet_model.h5": "https://drive.google.com/uc?id=1w8_0hWIDlEEtHbtBEDkJCUqQkZJSH6XU",
    "ensemble_resnet_model.h5": "https://drive.google.com/uc?id=1v7PVOooZ2krSK4J24Fv0SrdxnajFVcmW",
}

def descargar_modelos():
    for nombre, url in MODELOS.items():
        if not os.path.exists(nombre):
            print(f"📥 Descargando {nombre}...")
            gdown.download(url, nombre, quiet=False)
        else:
            print(f"✅ {nombre} ya existe.")

if __name__ == "__main__":
    descargar_modelos()
