import unittest
import pickle
import dwave_networkx as dnx
import pandas as pd
from src.generate_pegasus_instances import generate_pegasus_instances, tuple_to_spin_glass


class PegasusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P4 = dnx.pegasus_graph(4, nice_coordinates=True)
        generate_pegasus_instances(number=2, size=4, output_path="instances", output_types=["SpinGlass", "DWave"],
                                   category="RAU")
        spin_glass = pd.read_csv("instances/001_sg.txt", sep=" ", index_col=False, skiprows=1,
                                     names=["v", "w", "value"], header=None)

        cls.h_sg = spin_glass.loc[spin_glass["v"] == spin_glass["w"]]
        cls.J_sg = spin_glass.loc[spin_glass["v"] != spin_glass["w"]]
        with open("instances/001_dv.pkl", "rb") as f:
            cls.h_dv, cls.J_dv = pickle.load(f)

    def test_sg_instance(self):
        self.assertEqual(self.h_sg.shape[0], len(self.P4.nodes()))
        self.assertEqual(self.J_sg.shape[0], len(self.P4.edges()))

        for row in self.h_sg.itertuples():
            self.assertTrue(-0.1 <= row.value <= 0.1)

        for row in self.J_sg.itertuples():
            self.assertTrue(-1 <= row.value <= 1)

    def test_dv_instance(self):
        self.assertIsInstance(self.h_dv, dict)
        self.assertIsInstance(self.J_dv, dict)
        self.assertEqual(len(self.h_dv), len(self.P4.nodes()))
        self.assertEqual(len(self.J_dv), len(self.P4.edges()))

        for node, value in self.h_dv.items():
            self.assertIn(node, self.P4.nodes())
            self.assertTrue(-0.1 <= value <= 0.1)

        for edge, value in self.J_dv.items():
            self.assertIn(edge, self.P4.edges())
            self.assertTrue(-1 <= value <= 1)

    def test_device_instance(self):
        pass

    def test_same_instances(self):
        for node, value in self.h_dv.items():
            index = self.h_sg.loc[self.h_sg["v"] == tuple_to_spin_glass(node, 4)].index[0]
            self.assertAlmostEqual(value, self.h_sg.at[index, "value"], 15)

        for edge, value in self.J_dv.items():
            index = self.J_sg.loc[self.J_sg["v"] == tuple_to_spin_glass(edge[0], 4)]
            index = index.loc[index["w"] == tuple_to_spin_glass(edge[1], 4)].index[0]
            self.assertAlmostEqual(value, self.J_sg.at[index, "value"], 15)

    def test_different_instances(self):
        with open("instances/002_dv.pkl", "rb") as f:
            h, J = pickle.load(f)

        self.assertNotEqual(h, self.h_dv)
        self.assertNotEqual(J, self.J_dv)



if __name__ == '__main__':
    unittest.main()
