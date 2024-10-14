from scipy.integrate import quad, dblquad
import numpy as np

def f1(x):
    return 2 * np.tan(x) + 1

def f2(y, x):
    return x + y

if __name__ == "__main__":
    # Первый интеграл
    result1, error1 = quad(f1, -1, 2)
    print("Первый интеграл:", result1)

    # Второй интеграл
    result2, error2 = dblquad(f2, 0, 1, lambda x: x**2, lambda x: x)
    print("Второй интеграл:", result2)
