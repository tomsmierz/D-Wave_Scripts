import unittest
import numpy as np
import dimod
import os
from src.instances import load_instance, vectorize

cwd = os.getcwd()
rng = np.random.default_rng()


class MatyasConv(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.Q = load_instance(
            os.path.join(cwd, "..\\instances\\matyas_instances\\G1_py.csv")
        )
        cls.s = np.array([rng.choice([0, 1]) for _ in range(800)])
        cls.sp = np.array([-1 if cls.s[i] == 0 else 1 for i in range(800)])

    def test_qubo_to_ising_energies(self):
        h, J, offset = dimod.qubo_to_ising(self.Q)
        h_v, J_v = vectorize(h, J)
        ising = np.dot(np.dot(self.sp, J_v), self.sp) + np.dot(self.sp, h_v) + offset
        Q_v = np.zeros((800, 800))
        for key, value in self.Q.items():
            Q_v[key[0], key[1]] = value
        qubo = np.dot(np.dot(self.s, Q_v), self.s)

        self.assertEqual(ising, qubo)

    def test_two_way(self):
        h, J, offset1 = dimod.qubo_to_ising(self.Q)
        q, offset2 = dimod.ising_to_qubo(h, J, offset1)

        self.assertEqual(self.Q, q)


if __name__ == "__main__":
    unittest.main()
