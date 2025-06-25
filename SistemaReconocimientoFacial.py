import cv2
import numpy as np
import os
import pickle
import tkinter as tk
import time

# Parámetros
DB_PATH = "svd_face_db.pkl"
IMG_SIZE = 100
K = 10
UMBRAL = K * 0.8

# ---------- Utilidades ----------
def cargar_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def guardar_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

# ---------- Ventanas personalizadas ----------
def ventana_mensaje(titulo, mensaje, color="#3498db"):
    win = tk.Toplevel()
    win.title(titulo)
    win.configure(bg="#2c3e50")
    win.geometry("400x200")
    tk.Label(win, text=titulo, font=("Helvetica", 16, "bold"), bg="#2c3e50", fg=color).pack(pady=15)
    tk.Label(win, text=mensaje, font=("Helvetica", 12), bg="#2c3e50", fg="white", wraplength=350).pack(pady=10)
    tk.Button(win, text="Aceptar", command=win.destroy,
              font=("Helvetica", 10, "bold"), bg=color, fg="white", bd=0, width=12).pack(pady=20)

def ventana_input(titulo, prompt):
    def submit():
        nonlocal entrada_texto
        user_input = entrada_texto.get()
        win.destroy()
        resultado[0] = user_input

    resultado = [None]
    win = tk.Toplevel()
    win.title(titulo)
    win.configure(bg="#2c3e50")
    win.geometry("400x200")

    tk.Label(win, text=prompt, font=("Helvetica", 14, "bold"), bg="#2c3e50", fg="white").pack(pady=20)
    entrada_texto = tk.Entry(win, font=("Helvetica", 12), width=30)
    entrada_texto.pack(pady=10)
    tk.Button(win, text="Registrar", command=submit,
              font=("Helvetica", 10, "bold"), bg="#27ae60", fg="white", bd=0, width=12).pack(pady=10)

    win.grab_set()
    win.wait_window()
    return resultado[0]

# ---------- Imagen ----------
def capturar_imagen_desde_camara():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        return None

    print("📷 Captura tu imagen, presiona 's' para tomarla...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Captura de Imagen", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
            cap.release()
            cv2.destroyAllWindows()
            return img_resized

    cap.release()
    cv2.destroyAllWindows()
    return None

# ---------- Procesamiento ----------
def svd_features(img):
    U, S, Vt = np.linalg.svd(img, full_matrices=False)
    return U[:, :K]

def similitud_ortogonal(U1, U2):
    M = np.dot(U1.T, U2)
    return np.trace(np.dot(M, M.T))

# ---------- Funciones principales ----------
def registrar_desde_camara():
    nombre = ventana_input("Registro", "Ingresa el nombre del usuario:")
    if not nombre:
        return
    img = capturar_imagen_desde_camara()
    if img is None:
        return
    db = cargar_db()
    db[nombre] = svd_features(img)
    guardar_db(db)
    ventana_mensaje("Registro exitoso", f"✅ Usuario '{nombre}' registrado correctamente.", "#27ae60")

def verificar_desde_camara():
    inicio_total = time.time()
    img = capturar_imagen_desde_camara()
    if img is None:
        return
    t1 = time.time()
    U_desconocido = svd_features(img)
    t2 = time.time()
    db = cargar_db()
    mejor_match = None
    mejor_similitud = -1

    for nombre, U_guardado in db.items():
        sim = similitud_ortogonal(U_guardado, U_desconocido)
        if sim > mejor_similitud:
            mejor_similitud = sim
            mejor_match = nombre
    t3 = time.time()
    total = time.time() - inicio_total

    # Mostrar resultados
    if mejor_similitud >= UMBRAL:
        ventana_mensaje("Verificación exitosa", f"✅ Acceso concedido a {mejor_match}", "#27ae60")
    else:
        ventana_mensaje("Verificación fallida", "❌ Acceso denegado", "#e74c3c")

    print("\n⏱ Tiempos de ejecución:")
    print(f"- Captura + Preprocesamiento: {t1 - inicio_total:.4f} s")
    print(f"- Cálculo SVD: {t2 - t1:.4f} s")
    print(f"- Comparación: {t3 - t2:.4f} s")
    print(f"- Total: {total:.4f} s")

# ---------- Interfaz principal ----------
def mostrar_menu():
    root = tk.Tk()
    root.title("🔐 Sistema de Reconocimiento Facial")
    root.geometry("420x320")
    root.configure(bg="#2c3e50")

    tk.Label(root, text="Reconocimiento Facial con SVD", font=("Helvetica", 18, "bold"),
             bg="#2c3e50", fg="#ecf0f1").pack(pady=30)

    estilo_boton = {
        "width": 25,
        "height": 2,
        "font": ("Helvetica", 12, "bold"),
        "bd": 0,
        "cursor": "hand2"
    }

    tk.Button(root, text="📥 Registrar Usuario", command=registrar_desde_camara,
              bg="#2980b9", fg="white", activebackground="#1c638d", **estilo_boton).pack(pady=10)

    tk.Button(root, text="🔎 Verificar Identidad", command=verificar_desde_camara,
              bg="#27ae60", fg="white", activebackground="#1e8449", **estilo_boton).pack(pady=10)

    tk.Button(root, text="❌ Salir", command=root.destroy,
              bg="#e74c3c", fg="white", activebackground="#c0392b", **estilo_boton).pack(pady=10)

    root.mainloop()

# Ejecutar
mostrar_menu()