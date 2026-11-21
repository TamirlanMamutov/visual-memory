"""Compatibility wrapper; see experiments/candidate_tools."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.candidate_tools.adapt_eagle import annotations, choose_records, export_pool, main

if __name__=='__main__':main()
