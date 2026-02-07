"""Convert JP-10 CTI mechanism to YAML format for Cantera 3.0+."""
import cantera as ct
import os

# Load from CTI
cti_path = r"c:\Users\Benji\Documents\jantera\Assetts\Z77_JP10_20220324.txt"
sol = ct.Solution(cti_path)

# Write to YAML
yaml_path = os.path.join(os.path.dirname(__file__), "../jantera_v2/jp10.yaml")
sol.write_yaml(yaml_path, header=False)
print(f"Converted {cti_path} to {yaml_path}")
print(f"Species: {sol.n_species}, Reactions: {sol.n_reactions}")
