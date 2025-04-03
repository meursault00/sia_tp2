
# TP2 SIA - Algoritmos de Genéticos

## Introducción

Se busca implementar un compresor de imágenes un tanto peculiar. Deberemos implementar
un motor de Algoritmos Genéticos que pueda recibir una imágen, y lograr la mejor
aproximación a ella a través de triángulos sobre un canvas blanco.
Nuestros únicos parámetros (no confundir con hiperparámetros) entonces serán la imagen a
procesar, y T – la cantidad de triángulos que queremos utilizar para aproximar esa imágen.
Los triángulos deberán ser de un color uniforme pero pueden ser traslúcidos (RGBA, HSLA,
…).

[Enunciado](docs/SIA - TP2 - 2025 1Q.pdf)

### Requisitos

- Python3
- pip3
- [pipenv](https://pypi.org/project/pipenv/)

### Instalación

Parado en la carpeta del tp2 ejecutar

```sh
pipenv install
```

para instalar las dependencias necesarias en el ambiente virtual

## Ejecución

```
pipenv run python main.py [config_path] [image_path]
```

