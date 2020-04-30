import cv2

detectorFaces = cv2.CascadeClassifier("original.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read('classificadoEing.yml')
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0) #cv2.CAP_DSHOW taxa de quadros de 30fps para 7fps



while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFaces.detectMultiScale(imagemCinza,
                                                     scaleFactor=1.5,
                                                     minSize=(30, 30))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        id, confiança = reconhecedor.predict(imagemFace)
        nome = 'Desconhecido'
        if id == 1:
            nome = 'Pedro'
        elif id == 2:
            nome = 'Gentil'
        cv2.putText(imagem, nome, (x, y + (a + 30)), font, 2, (0,0,255))
        cv2.putText(imagem, str(confiança), (x, y + (a + 100)), font, 2, (0, 0, 255))


    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break



camera.release()
cv2.destroyAllWindows()