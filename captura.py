import cv2

classificador = cv2.CascadeClassifier("original.xml")
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) #cv2.CAP_DSHOW taxa de quadros de 30fps para 7fps
amostra = 1
numeroAmostras = 25
id = input("Digite seu identificador: ")
nome = input("Digite seu Nome: ")
largura, altura = 220, 220
print("Capturando as faces........")

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                     scaleFactor=1.5,
                                                     minSize=(150, 150))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
       # cv2.arrowedLine(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
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
