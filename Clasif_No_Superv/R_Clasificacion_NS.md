
# Técnicas de Clasificación I. Clasificación no supervisada

Jessica Bernal Borrego
27/03/2022


## Contenido:

1. [Introducción](#id1)
2. [Clasificación No Supervisada](#id2)
3. [Número de clústers](#id3)
4. [Escenas de primavera y verano](#id4)

<div id='id1' />


## 1. Introducción

Cargamos las librerías a utilizar (instalamos las que proceda si no las tenemos previamente):

```{r, results='hide', warning=FALSE, message=FALSE}
library(sp)
library(rgdal)
library(raster)
library(reshape)
library(grid)
library(gridExtra)
library(RStoolbox)
library(caret)
library(rasterVis)
library(corrplot)
library(doParallel)
library(NeuralNetTools)
library(tidyr)
library(stringr)
library(e1071)
library(sf)
library(mapview)
library(factoextra)

```
Creamos una variable donde indicamos el directorio donde tenemos las bandas de la escena de otoño. Podemos sentar también la ruta de trabajo mediante la función "setwd".


```{r}

setwd("C:/Geoforest/Tec_Clasif")
dir_in="./Material_practicas/Sentinel/O"

rasList=list.files(dir_in, pattern = "tif", full.names = TRUE)

```

A continuación, procedemos a crear un ráster stack que tenga almacenada la **escena de otoño**. Es decir, creamos una única variable (sentinel_o) que contenga el total de las bandas (las 10 imágenes Sentinel de nuestra carpeta de otoño)

```{r}
sentinel_o=stack(rasList)

```

Ploteamos, y vemos que se nos muestran de forma independiente cada una de las bandas

```{r}
plot(sentinel_o)

```
![image](https://user-images.githubusercontent.com/100314590/160289807-9018e4cb-dc05-48f2-977a-624ba7f632bc.png)

Las imágenes anteriores ya habían sido corregidas atmosféricamente de forma previa. Sabiendo esto, podemos intuir que es por esto que los valores que muestran las gráficas a la derecha de cada imagen, no representan valores de reflectancia (valores porcentuales). A menudo estos valores se someten a un factor de escala para mostrar valores enteros, dado que los valores decimales son un tipo de dato que ocupan mayor espacio, y por tanto se trata de evitar. En los metadatos, se suele indicar si se ha usado un factor de escala. 

Vamos a customizar la paleta de colores, indicando el rango de colores que queremos utlizar, en nuestro caso 256 colores.

```{r}
plot(sentinel_o, col=grey(0:255/255))

```
![image](https://user-images.githubusercontent.com/100314590/160289845-5a641ec8-4c41-49dc-86a5-2dd7eb0b783f.png)


Otra forma de representación sería:

```{r}
spplot(sentinel_o, col.regions=gray(0:255/255))

```
![image](https://user-images.githubusercontent.com/100314590/160289890-63a2eecc-e41c-4d91-8b63-ac2e92c42adb.png)



Si quisiéramos representar una combinación entre bandas, emplearemos:

```{r}
plotRGB(sentinel_o, r=7, g=2, b=3, stretch="lin")
```
![image](https://user-images.githubusercontent.com/100314590/160289919-6a58bde6-c57b-406f-96a7-bd0697595d8f.png)

Lo que hace el stretch lineal es ajustar el rango dinámico de la imagen que puede emplear una computadora para representar una imagen en la pantalla. En este caso hemos representado una imagen en falso color. 

Otra combinación sería:

```{r}
plotRGB(sentinel_o, r=2, g=4, b=2, stretch="lin")
```
![image](https://user-images.githubusercontent.com/100314590/160289932-38b85f4f-79be-4da1-8706-6124cec4078f.png)

Podemos ejecutar las dos líneas al mismo tiempo si queremos comparar (ventajas de markdown en R ;)

```{r}
plotRGB(sentinel_o, r=7, g=2, b=3, stretch="lin")
plotRGB(sentinel_o, r=2, g=4, b=2, stretch="lin")
```
![image](https://user-images.githubusercontent.com/100314590/160289967-60653324-5d40-4bb0-8bf2-529eaaa44f9d.png)

Aquí no tenemos herramientas para hacer un zoom, de modo que para ello utilizamos el paquete mapview (ej con la capa 1):

```{r}
mapview(sentinel_o [[1]])
```
![image](https://user-images.githubusercontent.com/100314590/160290017-c645a8e5-87d2-4d66-b3e4-d2b0078d7c88.png)


<div id='id2' />

## 2. Clasificación No Supervisada

En este tipo de clasificación, nosotros le indicamos directamente al algoritmo a emplear en cuantas clases queremos agrupar los píxeles de la escena. La determinación del número de clases adecuado requiere de lo que se denomina un entrenamiento. 

Para ello vamos a utilizar una función denominada *kmean* para hacer un clúster del contenido de los píxels:

```{r}
kmncluster<-kmeans(sentinel_o[], 
                   centers = 3,
                   iter.max = 500,
                   nstart = 2,
                   algorithm = "Lloyd")

```

Vemos que nos devuelve una estructura de datos compleja. Exploramos el objeto creado:

```{r}
str(kmncluster)

```

Procedemos a transformar los datos en imagen. Para ello, primero vamos a crear una imagen sin contenido, un lienzo en forma de variable en el que el número de filas, de columnas y extensión geográfica va a ser la misma que la de kmncluster (la escena sentinel de la que partimos)

```{r}
resultado=raster(sentinel_o)

```

Le damos contenido:

```{r}
resultado=setValues(resultado, kmncluster$cluster)
plot(resultado)
```
![image](https://user-images.githubusercontent.com/100314590/160290190-bb6777bf-795a-460c-b1b2-9dd50c70c6e9.png)


Si queremos pintarla en mapview:

```{r}
mapview(resultado)

```

![image](https://user-images.githubusercontent.com/100314590/160290205-752b7329-5de0-461c-8e08-b1a8ac263e6a.png)

Aunque aparezca un gradiente en la leyenda, realmente cada píxel es un valor entero correspondiente a la clase.

Vamos a crear una variable para controlar la paleta de color: 

```{r}
pal=mapviewPalette("mapviewTopoColors")
mapview(resultado, col.regions=pal(10), at=seq(0,6,1))

```

![image](https://user-images.githubusercontent.com/100314590/160290224-17a4feac-f564-48e1-ba52-fc880ab2ddc0.png)

Procedemos a guardar la imagen de la clasificación no supervisada en el disco duro: 

```{r}
writeRaster(resultado, paste(dir_in, "salida_3.tif", sep="/"),
            overwrite=TRUE)
```

La clasificación no supervisada sirve como paso previo a la clasificación supervisada, de modo que nos muestra qué podemos "pedirle" a la escena.

Hemos realizado una clasificación en 3 clústers o grupos, podemos testar con R si el número de clústers es el adecuado.

<div id='id3' />
## 3. Número de Clústers

En este punto cabe preguntarse cuál es el número adecuado de clases o clústers a definir. Esta determinación en el número óptimo de clases es fundamental en el clustering. Sin embargo, no existe una respuesta única y directa a esta cuestión. El número óptimo de clústers es en cierto modo subjetivo y depende del método utilizado para medir las similitudes, así como de los parámetros utilizados para la partición.

Los métodos que podemos emplear para la determinación de este número pueden ser métodos directos y métodos de prueba estadística:

* Métodos directos: consisten en optimizar un criterio, como las sumas de cuadrados dentro del clúster.

* Métodos de prueba estadística: consiste en comparar las pruebas con la hipótesis nula. 

Además de estos metodos descritos hay más de treinta  índices y métodos publicados para identificar el número óptimo de cluster. A continuación, veremos algunos de ellos.

### 3.1 Método de Elbow

El objetivo es definir los clústers de forma que la variación total intra-clúster (suma total de los cuadrados o WSS, que mide la compacidad de la agrupación) sea lo más pequeña posible. Así, se considera la WSS dentro de clúster en función del número de clúster de forma que hay que elegir un número de clústers tal que añadir otro clúster no mejore mucho el WSS total.

Usaremos la función *fviz_nbclust* contenida en el paquete *factoextra* que nos asistirá en el número de clusters a usar. Parte de este proceso podría hacerse también en QGIS pero llevaría mucho más tiempo.

Indicamos el set de datos, el algoritmo de clustering y el método de determinación del número de clúster.

```{r, error=TRUE}
fviz_nbclust(sentinel_o, kmeans, method = "wss") 
```
![image](https://user-images.githubusercontent.com/100314590/160290553-558e07e8-32a7-48e4-abe5-28bce03972c2.png)

Se ha obtenido un error indicando que se alcanzó el límite de memoria, por tanto *a priori* no sería posible aplicar estas técnicas en caso de que la imagen alcanzara unas determinadas dimensiones, tuviera un número elevado de bandas, o ambas situaciones.

Como alternativa, vamos a realizar un muestreo sobre la imagen, extrayendo los valores de reflectancia en cada uno de los puntos de muestreo. Para ello, en primer lugar vamos a leer el fichero shapefile con la delimitación del aerea de interes (AOI.shp)

```{r}
AOI=st_read("./Material_practicas/Area_estudio/AOI.shp")
```

```{r}
mapview(AOI)
```
![image](https://user-images.githubusercontent.com/100314590/160290587-0d98d8e5-93c6-4215-bddb-d0131b76862e.png)


A continuación, vamos a realizar un muestro aleatorio dentro de la zona de interés mediante la función *st_sample*

Para ello, creamos una semilla y la geometría:

```{r}
set.seed(123)
puntos.ref <- st_sample(AOI, c(5000), type='random',exact=TRUE)
```

Comprobamos que se ha generado correctamente el muestreo:

```{r}
mapview(puntos.ref,
        cex=2)+
  mapview(AOI, 
          legend=FALSE,
          col.regions = c("white"))

```
![image](https://user-images.githubusercontent.com/100314590/160290611-dc94d8cc-1100-41e4-aa44-644d34b22bec.png)

A continuación convertimos la lista de puntos en un spatial feauture mediante la función *st_sf* para posteriormente emplear la función *extract* para extraer los valores de reflectancia de cada banda, siendo estos almacenados en memoria como un *dataframe*.

```{r}
puntos.ref<-st_sf(puntos.ref)

puntos.ref<- as.data.frame(raster::extract(sentinel_o,puntos.ref))
```

Una vez se tienen los datos extraidos ya estamos en disposición de determinar el número apropiado de clústers. En este caso usando el método *Elbow*.

```{r}
# Metodo Elbow
fviz_nbclust(puntos.ref, kmeans, method = "wss")
```
![image](https://user-images.githubusercontent.com/100314590/160290656-d533449b-d97e-42d9-89cf-6d316bad92f7.png)


### 3.2 Método Silhouette

Otro método es el de Silhouette, este calcula la silueta media de las observaciones para diferentes valores de *k*. El número óptimo de clusters k es el que maximiza la silueta media en un rango de valores posibles para k. 

```{r}
fviz_nbclust(puntos.ref, kmeans, method = "silhouette")+
  labs(subtitle = "Metodo Silhouette")
```
![image](https://user-images.githubusercontent.com/100314590/160290700-1ee2d3f2-3069-4416-a64f-706dc700dd56.png)

Por último, vamos a recortar la imagen clasificada enmascarándola según la zona de interés.


```{r}
NS_O=mask(resultado,AOI)
mapview(NS_O, col.regions = pal(10), at = seq(0, 4, 1))
```
![image](https://user-images.githubusercontent.com/100314590/160290727-6d6cdf74-ce1d-4556-839f-a54caf9136ed.png)


Si queremos, guardamos la imagen de la clasificación no supervisada en el disco duro: 

```{r}
writeRaster(resultado, paste(dir_in, "salida_4.tif", sep="/"),
            overwrite=TRUE)
```
<div id='id4' />
## 4. Escenas de primavera y verano

A continuación se va a repetir el proceso para las escenas de primavera y verano

```{r}
#Escena primavera
dir_in<-('./Material_practicas/Sentinel/p')
rasList <- list.files(dir_in,pattern="tif",
                      full.names = TRUE)
sentinel_p <- stack(rasList)

#Escena verano
dir_in<-('./Material_practicas/Sentinel/v')

rasList <- list.files(dir_in,pattern="tif",
                      full.names = TRUE)

sentinel_v <- stack(rasList)

#Generación muestreo
puntos.ref <- st_sample(AOI, c(5000), type='random',exact=TRUE)

#puntos primeravera
puntos.p<-st_sf(puntos.ref)
puntos.p<- as.data.frame(raster::extract(sentinel_p,puntos.p))

#puntos verano
puntos.v<-st_sf(puntos.ref)
puntos.v<- as.data.frame(raster::extract(sentinel_v,puntos.v))

fviz_nbclust(puntos.p, kmeans, method = "wss")
fviz_nbclust(puntos.v, kmeans, method = "wss")

```
![image](https://user-images.githubusercontent.com/100314590/160290901-e7c5acef-58aa-4b77-b134-eefbcd32eb53.png)

Siguiendo la misma metodología se van a clasificar las escenas de primavera y de verano

```{r}
#Clasificación no supervisada primavera
NS_P_kmeans<-kmeans(sentinel_p[],
  centers=4,
  iter.max=500,
  nstart=2,
  algorithm="Lloyd")

NS_P=raster(sentinel_p)
NS_P=setValues(NS_P,NS_P_kmeans$cluster)
NS_P=mask(NS_P,AOI)

#Clasificación no supervisada verano
NS_V_kmeans<-kmeans(sentinel_v[],
  centers=4,
  iter.max=500,
  nstart=2,
  algorithm="Lloyd")

NS_V=raster(sentinel_v)
NS_V=setValues(NS_V,NS_V_kmeans$cluster)
NS_V=mask(NS_V,AOI)

mapview(NS_O,col.regions = pal(10), at = seq(0, 4, 1))+
  mapview(NS_P,col.regions = pal(10), at = seq(0, 4, 1))+
  mapview(NS_V,col.regions = pal(10), at = seq(0, 4, 1))
  

```
![image](https://user-images.githubusercontent.com/100314590/160290947-3b1f4c36-2285-42f9-a031-b0fd6ed7c5e7.png)

Comprobamos el número de clusters óptimos
```{r}
sentinel <- stack(sentinel_o,sentinel_p,sentinel_v)

puntos.pvo<-st_sf(puntos.ref)
puntos.pvo<- as.data.frame(raster::extract(sentinel,puntos.pvo))

fviz_nbclust(puntos.pvo, kmeans, method = "wss")

```
![image](https://user-images.githubusercontent.com/100314590/160290959-93c73b60-b957-4f73-999f-a11ed459f1ee.png)


A continuación, se va a realizar una clasificación no supervisada considerando 4 y 8 clústers. Hay que tener en cuenta que a mayor número de clusters mayor tiempo de ejecución tomará el proceso.

```{r}
#Clustering 4 clases
NS_PVO_kmeans<-kmeans(sentinel[],
  centers=4,
  iter.max=500,
  nstart=2,
  algorithm="Lloyd")

NS_PVO_4=raster(sentinel)
NS_PVO_4=setValues(NS_PVO_4,NS_PVO_kmeans$cluster)
NS_PVO_4=mask(NS_PVO_4,AOI)

#Clustering 8 clases
NS_PVO_kmeans<-kmeans(sentinel[],
  centers=8,
  iter.max=500,
  nstart=2,
  algorithm="Lloyd")

NS_PVO_8=raster(sentinel)
NS_PVO_8=setValues(NS_PVO_8,NS_PVO_kmeans$cluster)
NS_PVO_8=mask(NS_PVO_8,AOI)

mapview(NS_PVO_4,col.regions = pal(10), at = seq(0, 4, 1))+
  mapview(NS_PVO_8,col.regions = pal(10), at = seq(0, 7, 1))

```

![image](https://user-images.githubusercontent.com/100314590/160290981-eb02dff3-1103-4a17-889f-e71486f69b82.png)

Podemos representar los valores focales o "ventana móvil" para la vecindad de las celdas focales utilizando una matriz de pesos. En este caso vamos a hacerlo en combinación con la función modal.

```{r}
NS_PVO_4_moda<-focal(NS_PVO_4, w=matrix(1,3,3), fun=modal)
NS_PVO_8_moda<-focal(NS_PVO_8, w=matrix(1,3,3), fun=modal)

mapview(NS_PVO_4_moda,col.regions = pal(10), at = seq(0, 4, 1))+
  mapview(NS_PVO_8_moda,col.regions = pal(10), at = seq(0, 7, 1))


```

![image](https://user-images.githubusercontent.com/100314590/160291016-6d2db590-5355-4de7-9992-61d1fa4f2c6a.png)
