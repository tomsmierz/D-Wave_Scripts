import unittest
import pickle
import dwave_networkx as dnx
import pandas as pd
from dwave.system import DWaveSampler
from src.generate_pegasus_instances import (
    generate_pegasus_instances,
    nice_to_spin_glass,
)
from src.generate_zephyr_instances import (
    generate_zephyr_instances,
    zephyr_to_spin_glass,
    create_zephyr_spinglass_clusters,
)

CATEGORY = "AC3"
SIZE_P = 4
SIZE_Z = 2


class PegasusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pegasus = dnx.pegasus_graph(SIZE_P, nice_coordinates=True)
        generate_pegasus_instances(
            number=2,
            size=SIZE_P,
            output_path="instances",
            output_types=["SpinGlass", "DWave"],
            category=CATEGORY,
        )
        generate_pegasus_instances(
            number=1,
            size=SIZE_P,
            output_path="instances",
            output_types=["DWave"],
            category=CATEGORY,
            device="Advantage_system6.1",
            name="qpu",
        )
        spin_glass = pd.read_csv(
            "instances/001_sg.txt",
            sep=" ",
            index_col=False,
            skiprows=1,
            names=["v", "w", "value"],
            header=None,
        )

        cls.h_sg = spin_glass.loc[spin_glass["v"] == spin_glass["w"]]
        cls.J_sg = spin_glass.loc[spin_glass["v"] != spin_glass["w"]]
        with open("instances/001_dv.pkl", "rb") as f:
            cls.h_dv, cls.J_dv = pickle.load(f)

    def test_sg_instance(self):
        self.assertEqual(self.h_sg.shape[0], len(self.pegasus.nodes()))
        self.assertEqual(self.J_sg.shape[0], len(self.pegasus.edges()))

        if CATEGORY == "RAU":
            for row in self.h_sg.itertuples():
                self.assertTrue(-0.1 <= row.value <= 0.1)

            for row in self.J_sg.itertuples():
                self.assertTrue(-1 <= row.value <= 1)
        elif CATEGORY == "RCO":
            for row in self.h_sg.itertuples():
                self.assertTrue(row.value == 0)

            for row in self.J_sg.itertuples():
                self.assertTrue(-1 <= row.value <= 1)
        else:  # AC3
            for row in self.h_sg.itertuples():
                self.assertTrue(-1 / 9 <= row.value <= 1 / 9)

    def test_dv_instance(self):
        self.assertIsInstance(self.h_dv, dict)
        self.assertIsInstance(self.J_dv, dict)
        self.assertEqual(len(self.h_dv), len(self.pegasus.nodes()))
        self.assertEqual(len(self.J_dv), len(self.pegasus.edges()))

        for node, value in self.h_dv.items():
            self.assertIn(node, self.pegasus.nodes())
            if CATEGORY == "RAU":
                self.assertTrue(-0.1 <= value <= 0.1)
            elif CATEGORY == "RCO":
                self.assertTrue(value == 0)
            else:
                self.assertTrue(-1 / 9 <= value <= 1 / 9)

        for edge, value in self.J_dv.items():
            self.assertIn(edge, self.pegasus.edges())
            if CATEGORY == "RAU":
                self.assertTrue(-1 <= value <= 1)
            elif CATEGORY == "RCO":
                self.assertTrue(-1 <= value <= 1)
            elif edge[0][1:3] == edge[1][1:3]:
                self.assertTrue(-1 / 3 <= value <= 1 / 3)
            else:
                self.assertTrue(-1 <= value <= 1)

    def test_device_instance(self):
        with open("instances/qpu1_dv.pkl", "rb") as f:
            linear, quadratic = pickle.load(f)
        sampler = DWaveSampler(solver="Advantage_system6.1")
        self.assertTrue(sampler.solver.check_problem(linear, quadratic))

    def test_same_instances(self):
        for node, value in self.h_dv.items():
            index = self.h_sg.loc[
                self.h_sg["v"] == nice_to_spin_glass(node, SIZE_P)
            ].index[0]
            self.assertAlmostEqual(value, self.h_sg.at[index, "value"], 15)

        for edge, value in self.J_dv.items():
            index = self.J_sg.loc[self.J_sg["v"] == nice_to_spin_glass(edge[0], SIZE_P)]
            index = index.loc[index["w"] == nice_to_spin_glass(edge[1], SIZE_P)].index[
                0
            ]
            self.assertAlmostEqual(value, self.J_sg.at[index, "value"], 15)

    def test_different_instances(self):
        with open("instances/002_dv.pkl", "rb") as f:
            h, J = pickle.load(f)
        if CATEGORY != "RCO":
            self.assertNotEqual(h, self.h_dv)
        self.assertNotEqual(J, self.J_dv)


class ZephyrTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.zephyr = dnx.zephyr_graph(SIZE_Z, coordinates=True)
        generate_zephyr_instances(
            number=2,
            size=SIZE_Z,
            output_path="instances",
            output_types=["SpinGlass", "DWave"],
            category=CATEGORY,
        )
        generate_zephyr_instances(
            number=1,
            size=SIZE_Z,
            output_path="instances",
            output_types=["DWave"],
            category=CATEGORY,
            device="Advantage2_prototype1.1",
            name="zqpu",
        )
        spin_glass = pd.read_csv(
            "instances/001_sg.txt",
            sep=" ",
            index_col=False,
            skiprows=1,
            names=["v", "w", "value"],
            header=None,
        )

        cls.h_sg = spin_glass.loc[spin_glass["v"] == spin_glass["w"]]
        cls.J_sg = spin_glass.loc[spin_glass["v"] != spin_glass["w"]]
        cls.clusters = create_zephyr_spinglass_clusters(cls.zephyr)

        with open("instances/001_dv.pkl", "rb") as f:
            cls.h_dv, cls.J_dv = pickle.load(f)

    def test_dv_instance(self):
        self.assertIsInstance(self.h_dv, dict)
        self.assertIsInstance(self.J_dv, dict)
        self.assertEqual(len(self.h_dv), len(self.zephyr.nodes()))
        self.assertEqual(len(self.J_dv), len(self.zephyr.edges()))

        for node, value in self.h_dv.items():
            self.assertIn(node, self.zephyr.nodes())
            if CATEGORY == "RAU":
                self.assertTrue(-0.1 <= value <= 0.1)
            elif CATEGORY == "RCO":
                self.assertTrue(value == 0)
            else:
                self.assertTrue(-1 / 9 <= value <= 1 / 9)

        for edge, value in self.J_dv.items():
            self.assertIn(edge, self.zephyr.edges())
            if CATEGORY == "RAU":
                self.assertTrue(-1 <= value <= 1)
            elif CATEGORY == "RCO":
                self.assertTrue(-1 <= value <= 1)
            elif self.clusters[edge[0]] == self.clusters[edge[1]]:
                self.assertTrue(-1 / 3 <= value <= 1 / 3)
            else:
                self.assertTrue(-1 <= value <= 1)

    def test_device_instance(self):
        with open("instances/zqpu1_dv.pkl", "rb") as f:
            linear, quadratic = pickle.load(f)
        sampler = DWaveSampler(solver="Advantage2_prototype1.1")
        self.assertTrue(sampler.solver.check_problem(linear, quadratic))

    def test_same_instances(self):
        for node, value in self.h_dv.items():
            index = self.h_sg.loc[
                self.h_sg["v"] == zephyr_to_spin_glass(node, SIZE_Z) + 1
            ].index[0]
            self.assertAlmostEqual(value, self.h_sg.at[index, "value"], 15)

        for edge, value in self.J_dv.items():
            index = self.J_sg.loc[
                self.J_sg["v"] == zephyr_to_spin_glass(edge[0], SIZE_Z) + 1
            ]
            index = index.loc[
                index["w"] == zephyr_to_spin_glass(edge[1], SIZE_Z) + 1
            ].index[0]
            self.assertAlmostEqual(value, self.J_sg.at[index, "value"], 15)

    def test_different_instances(self):
        with open("instances/002_dv.pkl", "rb") as f:
            h, J = pickle.load(f)
        if CATEGORY != "RCO":
            self.assertNotEqual(h, self.h_dv)
        self.assertNotEqual(J, self.J_dv)

    def test_sg_instance(self):
        self.assertEqual(self.h_sg.shape[0], len(self.zephyr.nodes()))
        self.assertEqual(self.J_sg.shape[0], len(self.zephyr.edges()))

        if CATEGORY == "RAU":
            for row in self.h_sg.itertuples():
                self.assertTrue(-0.1 <= row.value <= 0.1)

            for row in self.J_sg.itertuples():
                self.assertTrue(-1 <= row.value <= 1)
        elif CATEGORY == "RCO":
            for row in self.h_sg.itertuples():
                self.assertTrue(row.value == 0)

            for row in self.J_sg.itertuples():
                self.assertTrue(-1 <= row.value <= 1)
        else:  # AC3
            for row in self.h_sg.itertuples():
                self.assertTrue(-1 / 9 <= row.value <= 1 / 9)


if __name__ == "__main__":
    unittest.main()
