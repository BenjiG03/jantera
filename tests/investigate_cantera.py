import cantera as ct

def investigate():
    sol = ct.Solution('gri30.yaml')
    print(f"Cantera version: {ct.__version__}")
    
    types = set()
    for rxn in sol.reactions():
        types.add((rxn.reaction_type, rxn.rate.type))
    
    print("\nUnique Reaction Types found in GRI-30:")
    for rt, ratet in types:
        print(f"  Reaction Type: {rt}, Rate Type: {ratet}")
    
    # Check a falloff example specifically
    falloff_rxns = [r for r in sol.reactions() if 'falloff' in r.reaction_type]
    if falloff_rxns:
        r = falloff_rxns[0]
        print(f"\nFalloff example: {r}")
        print(f"  Type: {r.reaction_type}")
        print(f"  Rate Type: {r.rate.type}")
        print(f"  Dir(rate): {dir(r.rate)}")

if __name__ == "__main__":
    investigate()
