import cv2
import numpy as np

classificador = cv2.CascadeClassifier("original.xml")
classificadorOlho = cv2.CascadeClassifier("olhos.xml")
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) #cv2.CAP_DSHOW taxa de quadros de 30fps para 7fps
amostra = 1
numeroAmostras = 1
id = input("Digite seu identificador: ")
nome = input("Digite seu Nome: ")
largura, altura = 220, 220
print("Capturando as faces........")

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    #print(np.average(imagemCinza))
    facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                     scaleFactor=1.5,
                                                     minSize=(150, 150))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = imagem[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.5, minSize=(25,25))

        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if (np.average(imagemCinza) > 100):
                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                    cv2.imwrite("fotos/pessoa." + str(nome) + "." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                    print("[foto " + str(amostra) + " capturado com sucesso.]")
                    amostra = amostra + 1

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if (amostra >= numeroAmostras + 1):
        break
print("Faces capturadas com sucesso.")

camera.release()
cv2.destroyAllWindows()
