import unittest
import pickle
import dwave_networkx as dnx
import pandas as pd
from src.generate_pegasus_instances import tuple_to_spin_glass


class PegasusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P4 = dnx.pegasus_graph(4, nice_coordinates=True)

    def test_sg_instance(self):
        df = pd.read_csv("instances/P4_sg.txt", sep=" ", index_col=False, skiprows=1,
                         names=["v", "w", "value"], header=None)

        h = df.loc[df["v"] == df["w"]]
        J = df.loc[df["v"] != df["w"]]

        self.assertEqual(h.shape[0], len(self.P4.nodes()))
        self.assertEqual(J.shape[0], len(self.P4.edges()))

        for row in h.itertuples():
            self.assertTrue(-0.1 <= row.value <= 0.1)

        for row in J.itertuples():
            self.assertTrue(-1 <= row.value <= 1)

    def test_dv_instance(self):
        with open("instances/P4_dv.pkl", "rb") as f:
            h, J = pickle.load(f)

        self.assertIsInstance(h, dict)
        self.assertIsInstance(J, dict)
        self.assertEqual(len(h), len(self.P4.nodes()))
        self.assertEqual(len(J), len(self.P4.edges()))

        for node, value in h.items():
            self.assertIn(node, self.P4.nodes())
            self.assertTrue(-0.1 <= value <= 0.1)

        for edge, value in J.items():
            self.assertIn(edge, self.P4.edges())
            self.assertTrue(-1 <= value <= 1)

    def test_device_instance(self):
        pass

    def test_same_instances(self):
        df = pd.read_csv("instances/P4_sg.txt", sep=" ", index_col=False, skiprows=1,
                         names=["v", "w", "value"], header=None)
        h_sg = df.loc[df["v"] == df["w"]]
        J_sg = df.loc[df["v"] != df["w"]]

        with open("instances/P4_dv.pkl", "rb") as f:
            h_dv, J_dv = pickle.load(f)

        for node, value in h_dv.items():
            index = h_sg.loc[h_sg["v"] == tuple_to_spin_glass(node, 4)].index[0]
            self.assertAlmostEqual(value, h_sg.at[index, "value"], 15)

        for edge, value in J_dv.items():
            index = J_sg.loc[J_sg["v"] == tuple_to_spin_glass(edge[0], 4)]
            index = index.loc[index["w"] == tuple_to_spin_glass(edge[1], 4)].index[0]
            self.assertAlmostEqual(value, J_sg.at[index, "value"], 15)


    def test_different_instances(self):
        pass


if __name__ == '__main__':
    unittest.main()
