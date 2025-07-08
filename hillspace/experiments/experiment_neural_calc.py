import sys
import numpy as np


class NeuralCalculator:
    def __init__(self):
        # Hill Space weights: tanh(15) ≈ 1.0, sigmoid(15) ≈ 1.0
        # After constraint W = tanh(W_hat) * sigmoid(M_hat):
        #   Addition/Multiply:  [15,  15] → [1.0,  1.0] → x + y / x * y
        #   Subtract/Division:  [15, -15] → [1.0, -1.0] → x - y / x / y
        self.weights = {
            "add": np.array([[15.0, 15.0], [15.0, 15.0]], dtype=np.float16),
            "sub": np.array([[15.0, -15.0], [15.0, 15.0]], dtype=np.float16),
            "mul": np.array([[15.0, 15.0], [15.0, 15.0]], dtype=np.float16),
            "div": np.array([[15.0, -15.0], [15.0, 15.0]], dtype=np.float16),
        }

    def compute(self, x, y, operation):
        """Neural computation with enumerated weights"""
        W_hat, M_hat = self.weights[operation]
        # Hill Space constraint: W = tanh(W_hat) * sigmoid(M_hat)
        W = np.tanh(W_hat) * (1 / (1 + np.exp(-M_hat)))
        inputs = np.array([x, y])
        if operation in ["add", "sub"]:
            return np.dot(inputs, W)  # Linear: x*w1 + y*w2
        else:  # mul, div
            return np.prod(np.power(inputs, W))  # Exponential: x^w1 * y^w2


def main():
    if len(sys.argv) != 4:
        print("Usage: python neural_calc.py <num1> <op> <num2>")
        sys.exit(1)
    x, op_symbol, y = float(sys.argv[1]), sys.argv[2], float(sys.argv[3])
    op_map = {"+": "add", "-": "sub", "x": "mul", "/": "div"}
    if op_symbol not in op_map or (op_symbol == "/" and y == 0):
        print(f"Invalid operation or division by zero")
        sys.exit(1)
    calc = NeuralCalculator()
    predicted = calc.compute(x, y, op_map[op_symbol])
    actual = {"add": x + y, "sub": x - y, "mul": x * y, "div": x / y}[op_map[op_symbol]]
    print(f"Neural: {x} {op_symbol} {y} = {predicted}")
    print(f"Truth:  {actual}")
    print(f"Error:  {abs(actual - predicted):.2e}")


if __name__ == "__main__":
    main()
