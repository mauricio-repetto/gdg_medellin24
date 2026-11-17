# ¿Cómo hago para que la IA me conozca? — GDG Medellín 2024

Taller de **fine-tuning de modelos de difusión (Stable Diffusion)** presentado por Mauricio Repetto (ML Engineer en Nearsure) en un evento de GDG Medellín (octubre–noviembre 2024).

El objetivo es introducir los modelos generativos de difusión y mostrar, con ejemplos prácticos, cómo a partir de 3 a 5 fotos propias podemos "enseñarle" un concepto nuevo (nosotros mismos, con el token `<mauri>`) a Stable Diffusion para luego generar imágenes donde somos protagonistas. Se comparan tres técnicas de fine-tuning: **Textual Inversion**, **DreamBooth** y **LoRA**.

## Contenido de las slides

| Bloque | Temas |
|---|---|
| AI vs ML vs DL | Definiciones (McCarthy), machine learning, deep learning |
| IA generativa | Qué es GenAI; ejemplos en NLP (ChatGPT, Gemini), audio (Stable Audio, Udio) y visión (Stable Diffusion, Midjourney, DALL·E); IA tradicional vs. generativa |
| Fundamentos | Autoencoders, VAEs y VQ-VAEs; GANs (generador vs. discriminador); modelos de difusión y la U-Net |
| Hands on: Stable Diffusion | Latent diffusion, cross-attention y condicionamiento de texto; inferencia; comparación de priors SD v1.5 / v2.1 / SDXL 1.0 |
| Técnicas de fine-tuning | Textual Inversion, DreamBooth y LoRA, cada una con su notebook |
| Extras | Novedades (Würstchen, DiTs, InstaFlow, PixArt-α, DALL·E 3, SD3, Flux.1-dev), HyperDreamBooth, Perfusion, SDXL Refiner, tesis de maestría (data augmentation con datos sintéticos) |

## Estructura del repo

```
gdg_medellin24/
├── README.md
├── drive_assets.json                 # IDs de Drive de los archivos que se bajan al repo
├── notebooks/
│   ├── 01_textual_inversion.ipynb    # Textual Inversion
│   ├── 02_dreambooth.ipynb           # DreamBooth sin priors / con priors
│   └── 03_lora.ipynb                 # DreamBooth + LoRA sin priors / con priors
├── slides/
│   └── como_hago_para_que_la_ia_me_conozca.pdf
└── scripts/
    └── fetch_assets.py               # Re-descarga notebooks y slides desde Drive
```

Los notebooks se renombraron respecto a Drive (allí se llaman `¿Cómo hago para que la AI me conozca? - <Técnica>.ipynb`); el mapeo está en `drive_assets.json`.

## Los notebooks

Los tres siguen la misma estructura y están pensados para **Google Colab con GPU** (probados en una T4):

1. **Setup**: subir 3–5 fotos propias a una carpeta de Drive (ver `sample_fotos/` como ejemplo), montar Drive, chequear la GPU e instalar `diffusers`, `xformers`, `bitsandbytes` (y `peft` para LoRA). Se descarga el script oficial de entrenamiento de `diffusers`.
2. **Utilidades**: manejo de archivos y una tabla HTML para ver las imágenes.
3. **Dándonos a conocer**: entrenamiento sobre Stable Diffusion v1.5. Ojo: StabilityAI pidió dar de baja `runwayml/stable-diffusion-v1-5` en Hugging Face, así que revisá el ID del modelo base antes de correr.
4. **Inference**: carga del modelo con un *pipeline* de `diffusers`, cambio de *scheduler* y generación de imágenes con prompts como *"A photo of a smiling `<mauri>` man posing with the Everest mountain in the background"*.
5. **Comparación con SD 1.5** sin fine-tuning.

| Notebook | Técnica | Detalles |
|---|---|---|
| `01_textual_inversion.ipynb` | Textual Inversion | Aprende un nuevo embedding de texto para `<mauri>`; ~1 h en una T4. Funciona mejor para estilos |
| `02_dreambooth.ipynb` | DreamBooth | Fine-tuning del modelo completo, **sin priors** y **con priors** (prior preservation para reducir overfitting y *language drift*); con priors ~3 h 20 min en una T4. Usa gradient checkpointing y optimizador de 8 bits |
| `03_lora.ipynb` | DreamBooth + LoRA | Solo entrena matrices de bajo rango agregadas a la U-Net; mismas variantes sin/con priors (~3 h con priors) |

Para usar los modelos de forma interactiva, los notebooks sugieren la WebUI de Automatic1111.

## Archivos pesados en Google Drive (no incluidos en el repo)

El repo no incluye pesos de modelos ni fotos. Están en la carpeta del evento:

**Carpeta raíz:** <https://drive.google.com/drive/folders/1HESxiuKD5WbdFNmn_TRtrbHY3MEqbyvQ>

| Recurso | Qué es | Link |
|---|---|---|
| `finetuned_models/` | Pesos resultantes de cada técnica | [carpeta](https://drive.google.com/drive/folders/1iw4GcLEbg9cWpZdi4NMTemb_E4FQuVmp) |
| ├─ `my_model_ti/` | Textual Inversion | [carpeta](https://drive.google.com/drive/folders/1DiLkjHaJtaV0kx3oYo-e2bFvGfe96KvX) |
| ├─ `my_model_db/` | DreamBooth sin priors | [carpeta](https://drive.google.com/drive/folders/13PNtGufDVP9opoqtjYbqHwwE6MEE0Cx-) |
| ├─ `my_model_dbp/` | DreamBooth con priors | [carpeta](https://drive.google.com/drive/folders/1-JEV-LyYdNdAq3ASfP_hMgZNl51iFj3U) |
| ├─ `my_model_db_lora/` | LoRA sin priors | [carpeta](https://drive.google.com/drive/folders/1EX88pcba0JyLYEftEompLYEP8qGxFUt7) |
| └─ `my_model_dbp_lora/` | LoRA con priors | [carpeta](https://drive.google.com/drive/folders/10zGUFcgwk5Q0PDQnL8E7G_OjfbSLFANc) |
| `sample_fotos/` | Las 7 fotos (`1.jpg`…`7.jpg`) usadas para entrenar | [carpeta](https://drive.google.com/drive/folders/1gr2IV6kyITBK83UW7jk4ILsRh7AGB9V7) |

Para saltear el entrenamiento, agregá un acceso directo a `finetuned_models/` en tu Drive y apuntá la ruta del modelo en la sección *Inference* del notebook a la subcarpeta correspondiente.

## Links externos

- Slides originales (Google Slides): <https://docs.google.com/presentation/d/1V1rnAXciGe4lISQf_9jQq18vKcERAndmfHYgwwV93lw/edit>
- Notebooks en Colab: [Textual Inversion](https://colab.research.google.com/drive/1vhRYbtNDSxdeW2pEHqTKAajOYhzMgoam) · [DreamBooth](https://colab.research.google.com/drive/1b-CdGcfdu_h2yfQ7uGb3VnGFS-AQ-pEy) · [LoRA](https://colab.research.google.com/drive/1gaIvIONm2VXiA9THBfmjEY2BKa1N053H)
- Links cortos del taller: carpeta `bit.ly/gdg24_ns`, Textual Inversion `bit.ly/gdgmed24_ns_ti`, DreamBooth `bit.ly/gdgmed24_ns_db`, LoRA `bit.ly/gdgmed24_ns_lora`
- Librería `diffusers`: <https://github.com/huggingface/diffusers>

## Actualizar notebooks y slides desde Drive

```bash
python scripts/fetch_assets.py          # baja lo que falte
python scripts/fetch_assets.py --force  # vuelve a bajar todo
```
