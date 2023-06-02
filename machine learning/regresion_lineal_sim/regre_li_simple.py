import os
import numpy as np

def Costo(xs, ys, theta):
    m = ys.size
    h = np.dot(xs, theta)
    j = (1/(2 * m)) * np.sum(np.square(h - ys))
    return j

def gradientDescent(xs, ys, theta, alpha, num_iters):
    m = ys.size
    theta = theta.copy()
    j_history = []
    for i in range(num_iters):
        h = np.dot(xs, theta)
        theta = theta - (alpha / m) * np.dot((h - ys), xs)
        #print(getCost(xs, ys, theta))
        j_history.append(Costo(xs, ys, theta))
    return theta, j_history

if __name__ == "__main__":
    
    data = np.loadtxt(os.path.join('Data','sales.txt'), delimiter = ',')
    ys = data[:,0]
    m = ys.size

    xs = np.stack([np.ones(m), data[:,2], data[:,3], data[:,4], data[:,5], data[:,6]], axis = 1)

    # Por ende, se requiere de 6 valores de theta
    theta = np.zeros(6)

    # Numero de iteraciones en las que se llega a los valores de J con menos cambios
    iterations = 10000

    # Con un valor mayor se generan errores en las iteraciones
    alpha = 0.000001

    theta, j_history = gradientDescent(xs ,ys, theta, alpha, iterations)

    # Se imprime los resultados
    print(f"\nValor del error con theta encontrado: {j_history[-1]}\n")
    print(f'Theta encontrada por descenso por el gradiente: {theta[0]}, {theta[1]}, {theta[2]}, {theta[3]}, {theta[4]}, {theta[5]}\n')
    
    #Se prueba con 3 valores escogidos al azar del dataset
    test_v1 = np.array([1, 225, 90, 60, 29, 1148])
    pred = np.dot(theta, test_v1)
    print("El valor de y estimado es: " + str(round(pred, 0)))
    print("El valor real de y es: 75.0\n")
    
    test_v2 = np.array([1, 108, 46, 49, 15, 415])
    pred = np.dot(theta, test_v2)
    print("El valor de y estimado es: " + str(round(pred, 0)))
    print("El valor real de y es: 19.0\n")
   
    test_v3 = np.array([1, 159, 64, 48, 21, 460])
    pred = np.dot(theta, test_v3)
    print("El valor de y estimado es: " + str(round(pred, 0)))
    print("El valor real de y es: 61.0\n")