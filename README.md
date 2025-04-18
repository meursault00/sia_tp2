
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

### Análisis de función de fitness

```
pipenv run python fitness_analysis.py configs/fitness_analysis.json
```

### Análisis de método de separación de población

```
pipenv run python pop_sep_analysis.py configs/pop_sep_analysis.json
```

### Análisis de estabilidad de método de selección
Correr con todas las configuraciones de la carpeta configs/selection_sability de la siguiente manera:
```
pipenv run python main.py configs/selection_sability/boltzmann.json images/bosch.jpg
```
Luego de volcar los datos en el archivo stability_analysis, ejecutarlo para ver gráficos:
```
pipenv run python stability_analysis.py
```

### Análisis de presión selectiva
Correr con todas las configuraciones de la carpeta configs/selection_pressure de la siguiente manera:
```
pipenv run python main.py configs/selection_pressure/ranking.json images/bosch.jpg
```
Luego de volcar los datos en el archivo selective_pressure_analysis, ejecutarlo para ver gráficos:
```
pipenv run python selective_pressure_analysis.py
```

### Análisis de diferente sampleo con permutación de métodos de mutación

```
pipenv run python analysis/sampling_analysis.py configs/analysis/sampling_analysis_config.json [image_path]
```

### Análisis de métodos de cada parámetro

```
pipenv run python analysis/selection_analysis.py configs/analysis/selection_analysis_config.json [image_path]
pipenv run python analysis/crossover_analysis.py configs/analysis/crossover_analysis_config.json [image_path]
pipenv run python analysis/mutation_analysis.py configs/analysis/mutation_analysis_config.json [image_path]
```

## Ejecución con Jupyter Notebook

Para ver los generaciones intermedias se puede usar Jupyter Notebook.

### Instalación

```
pipenv install jupyter
```

### Ejecución

Para abrir Jupyter Notebook –desde el virtual environment que se este usando- usar el siguiente comando:

```
jupyter notebook
```

Entonces se abrirá Jupyter Notebook en el navegador, abrir el archivo main.ipynb. 

Cuando se solicite elegir el kernel de Python 3 por default. 

De ser necesario modificar la segunda cell siguiendo el esquema abajo:

```
%run python main.py [config_path] [image_path]
```

Apretar el botón de :fast_forward: en la barra superior a la izquierda.