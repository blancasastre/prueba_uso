#En este script voy a probar a leer de una carpeta un volumen de páncreas y sus correspondiente segmentación
import os as os
import SimpleITK as sitk
import matplotlib.pyplot as plp
import numpy as np

path='C:\\Users\\Usuario\\Desktop\\caso_prueba\\2 labels'
folders = os.listdir(path)
sub_folders=[]

for i in folders:
    sub_folders.append(os.path.join(path,i))

imagenes=np.zeros((512,512,84,4))
archivos=os.listdir(sub_folders[0])
imagen = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(sub_folders[0],archivos[1])))
mascara1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(sub_folders[0],archivos[2])))
mascara2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(sub_folders[0],archivos[3])))

fig = plp.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(mascara1[:,:,120], cmap='gray')
ax2.imshow(mascara2[:,:,120], cmap='gray')
ax3.imshow(imagen[:,:,120], cmap='gray')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')

#VOLUME OVERLAP VARIABILITY
def dice(mascara1, mascara2, k = 1):
    intersection = np.sum(mascara1[mascara2==k]) * 2.0
    dice = intersection / (np.sum(mascara1) + np.sum(mascara2))
    return dice

vol_overlap_similarity=dice(mascara1,mascara2)
vol_overlap_diff=1-vol_overlap_similarity


#PAIRS OF OBSERVERS

#Volumen del variability set en cada slice

#POSSIBLE
possible=np.zeros((512,512,171))
vol_possible_slice=np.zeros(171)
for i in range(171):
    for j in range(512):
        for k in range(512):
            possible[j,k,i] = possible[j,k,i]+ mascara1[j,k,i] + mascara2[j,k,i]

possible[possible>1]=1
plp.imshow(possible[:,:,60])

#VOLUMEN DEL POSSIBLE EN CADA SLICE
for i in range(171):
    vol_possible_slice[i] = sum(sum(possible[:, :, i]))

#CONSENSUS
consensus=np.zeros((512,512,171))
vol_consensus_slice=np.zeros(171)
for i in range(171):
    for j in range(512):
        for k in range(512):
            consensus[j,k,i] = mascara1[j,k,i]*mascara2[j,k,i]


#VOLUMEN DEL CONSENSUS EN CADA SLICE
for i in range(171):
    vol_consensus_slice[i] = sum(sum(consensus[:, :, i]))

#VOLUMEN DE LA VARIABILITY EN CADA SLICE
vol_variability_slice=vol_possible_slice-vol_consensus_slice


variability=np.zeros((512,512,171))
for i in range(171):
    variability[:,:,i]=possible[:,:,i]-consensus[:,:,i]

fig = plp.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(consensus[:,:,120], cmap='gray')
ax2.imshow(possible[:,:,120], cmap='gray')
ax3.imshow(variability[:,:,120], cmap='gray')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax1.title.set_text('Consensus')
ax2.title.set_text('Possible')
ax3.title.set_text('Variability')





#LISTA CON LA MISMA SLICE EN LAS DOS MASCARAS
'''mascaras=np.zeros((512,512,171,2))
mascaras[:,:,:,0]=mascara1
mascaras[:,:,:,1]=mascara2
#Aquí estará el STRAPLE en cada slice
STAPLE_segmask=np.zeros((512,512,171))

for i in range(0,171):
    mascaras_sitk = []
    for j in range(0,2):
        mascaras_sitk.append(sitk.GetImageFromArray(mascaras[:,:,i,j].astype(np.int16)))
    STAPLE_seg_sitk = sitk.STAPLE(mascaras_sitk, 1)
    STAPLE_seg = sitk.GetArrayFromImage(STAPLE_seg_sitk)
    STAPLE_seg[STAPLE_seg < 1] = 0
    STAPLE_segmask[:,:,i]=STAPLE_seg[:,:]

fig, (ax1) = plp.subplots(1, 1, figsize=(15,15))
ax1.imshow(imagen[:,:,120], cmap='gray')
ax1.imshow(STAPLE_segmask[:,:,120], cmap='jet', interpolation="none", alpha=0.5)


vol_straple_slice=np.zeros(171)
for i in range(171):
    vol_straple_slice[i] = sum(sum(STAPLE_segmask[:, :, i]))'''

#DIVIDO ENTRE INTERSECCIÓN
cociente=np.zeros(171)
for i in range(171):
    if vol_consensus_slice[i]==0:
        cociente[i]=0
    else:
        cociente[i]=vol_variability_slice[i]/vol_consensus_slice[i]

pairwise_mean_variability=sum(cociente)/171

#CONTOUR DICE COEFICIENT
import cv2
kernel = np.ones((1, 1), np.uint8)
erosiones1=np.zeros((512,512,171))
erosiones2=np.zeros((512,512,171))
for i in range(171):
    erosiones1[:,:,i]=cv2.erode(mascara1[:,:,i], kernel, iterations=1)
    erosiones2[:, :, i] = cv2.erode(mascara2[:, :, i], kernel, iterations=1)


dilataciones1=np.zeros((512,512,171))
dilataciones2=np.zeros((512,512,171))
for i in range(171):
    dilataciones1[:,:,i]=cv2.dilate(mascara1[:,:,i], kernel, iterations=1)
    dilataciones2[:, :, i] = cv2.dilate(mascara2[:, :, i], kernel, iterations=1)

slice1=mascara1[:,:,120]
slice2=mascara2[:,:,120]

'''fig = plp.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(slice1, cmap='gray')
seg_contour1=slice1-erosiones1[:,:,120]
ax2.imshow(seg_contour1, cmap='gray')
seg_band1=dilataciones1[:,:,120]-erosiones1[:,:,120]
ax3.imshow(seg_band1, cmap='gray')


img_erosion2 = cv2.erode(slice2, kernel, iterations=1)
img_dilation2 = cv2.dilate(slice2, kernel, iterations=1)

fig = plp.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(slice2, cmap='gray')
seg_contour2=slice2-img_erosion2
ax2.imshow(seg_contour2, cmap='gray')
seg_band2=img_dilation2-img_erosion2
ax3.imshow(seg_band2, cmap='gray')'''

seg_contour1=mascara1-erosiones1
seg_contour2=mascara2-erosiones2
seg_band1=dilataciones1-erosiones1
seg_band2=dilataciones2-erosiones2


intersection1 = np.sum(seg_contour2[seg_band1==1])
intersection2 = np.sum(seg_contour1[seg_band2==1])
volT=sum(sum(sum(seg_contour1)))
volB=sum(sum(sum(seg_contour2)))
CD=(intersection1+intersection2)/(volT+volB)

'''intersection1=np.zeros((512,512))
for i in range(512):
    for j in range(512):
        intersection1[i,j] = seg_contour2[i,j] *seg_band1[i,j]

vol_intersection1 = sum(sum(intersection1))

intersection2=np.zeros((512,512))
for i in range(512):
    for j in range(512):
        intersection2[i,j] = seg_contour1[i,j] *seg_band2[i,j]

vol_intersection2 = sum(sum(intersection2))

volT=sum(sum(seg_contour1))
volB=sum(sum(seg_contour2))

CD=(vol_intersection1+vol_intersection2)/(volT+volB)'''





##GROUP WISE COMPARISON METRICS
mascaras_sitk=[]
mascaras=np.zeros((512,512,2))
mascaras[:,:,0]=mascara1[:,:,120]
mascaras[:,:,1]=mascara2[:,:,120]
for i in range(0,2):
    mascaras_sitk.append(sitk.GetImageFromArray(mascaras[:,:,i].astype(np.int16)))


# Run STAPLE algorithm
STAPLE_seg_sitk = sitk.STAPLE(mascaras_sitk,1)
STAPLE_seg = sitk.GetArrayFromImage(STAPLE_seg_sitk)
STAPLE_seg[STAPLE_seg<1]=0
fig, (ax1) = plp.subplots(1, 1, figsize=(15,15))
ax1.imshow(imagen[:,:,120], cmap='gray')
ax1.imshow(STAPLE_seg[:,:], cmap='jet', interpolation="none", alpha=0.5)

