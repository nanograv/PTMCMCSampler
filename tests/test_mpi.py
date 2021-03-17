import shutil
import subprocess
from pathlib import Path
from unittest import TestCase


class TestMpi(TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("chains")

    def test_mpi(self):
        subprocess.check_call(["mpirun", "--np", "2", "python", "tests/simple.py"])

        # check that there are two chain files
        chain_files = list((Path(__file__).parent / "chains").glob("chain_*.txt"))
        self.assertEqual(len(chain_files), 2)
