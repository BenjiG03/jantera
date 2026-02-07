import cantera as ct
import sys

try:
    # Try loading the converted YAML
    # We can turn off strict validation effectively by just loading it? 
    # Cantera 3.0 might be strict.
    sol = ct.Solution('jp10.yaml')
    print("Successfully loaded jp10.yaml")
    print(f"Species: {sol.n_species}")
    print(f"Reactions: {sol.n_reactions}")
    
    # Check species C5H11CO which had issues
    try:
        idx = sol.species_index('C5H11CO')
        print(f"Species C5H11CO index: {idx}")
    except:
        print("Species C5H11CO not found!")

except Exception as e:
    print(f"Failed to load jp10.yaml: {e}")
    sys.exit(1)
