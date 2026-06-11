# Net Charge Correction Design: Hybrid vs SepTop

## Hybrid Topology Protocol Implementation

### Sign Convention

The charge difference is computed as:
```
charge_difference = formal_charge(stateA) - formal_charge(stateB)
```

This is implemented in `gufe.LigandAtomMapping.get_alchemical_charge_difference()`.

Examples:
- benzene (0) → benzoic acid (-1): difference = +1 → water transformed to Cl- in stateB
- benzene (0) → aniline (+1): difference = -1 → water transformed to Na+ in stateB

Ion selection logic (from `hybridtop_protocols.py` line 438):
```python
ion = {-1: solvent_component.positive_ion, 1: solvent_component.negative_ion}[difference]
```

### Flow

1. **Detection** (`_adaptive_settings`): Checks `mapping.get_alchemical_charge_difference() != 0`
   and auto-enables `explicit_charge_correction = True` with 22 lambda windows / 20ns production.

2. **Validation** (`_validate_charge_difference`):
   - If no explicit correction requested: warn and return
   - Requires: solvent present, PME nonbonded method, `abs(diff) <= 1`
   - On success: logs which ion type will be used

3. **Execution** (`_handle_net_charge` in `hybridtop_units.py`):
   - `get_alchemical_waters()`: Finds water residues > `distance_cutoff` (0.8nm default) from all solutes. Picks the first qualifying water deterministically.
   - `handle_alchemical_waters()`: In-place modifies state B system:
     - Oxygen atom gets ion NonbondedForce parameters (charge, sigma, epsilon)
     - Hydrogen atoms get charge=0 (keep sigma, epsilon)
   - `_fix_alchemical_water_atom_mapping()`: Moves water atoms from environment map to core map so hybrid topology factory treats them as alchemical.

### Key Files
- `_rfe_utils/topologyhelpers.py`: `get_alchemical_waters`, `handle_alchemical_waters`, `_fix_alchemical_water_atom_mapping`, `_get_ion_and_water_parameters`
- `hybridtop_protocols.py`: `_adaptive_settings`, `_validate_charge_difference`
- `hybridtop_units.py`: `_handle_net_charge`
- `equil_rfe_settings.py`: `AlchemicalSettings.explicit_charge_correction`, `explicit_charge_correction_cutoff`

---

## SepTop Protocol Design

### Key Differences from Hybrid

In SepTop:
- Ligands A and B exist simultaneously in the AB system
- `AbsoluteAlchemicalFactory` with two `AlchemicalRegion`s (A and B) is used
- Lambda schedules decouple ligand B (couple ligand A) and vice versa
- There is no "state B system" that gets modified—there's one unified AB system

### Charge Correction Strategy

For a charge-changing SepTop transformation where `charge_diff = chg_A - chg_B`:
- At lambda=0: ligand A is interacting (charged), ligand B is decoupled (no charge)
  → system total charge = chg_A + environment_charge
- At lambda=1: ligand B is interacting (charged), ligand A is decoupled (no charge)
  → system total charge = chg_B + environment_charge

The total charge changes by `chg_B - chg_A = -charge_difference` going from λ=0 to λ=1.

To maintain constant charge: we couple a counterion alongside ligand B.
- If `charge_diff > 0` (A more positive): ligand B is more negative than A,
  so coupling B adds negative charge → couple a positive ion to compensate
- If `charge_diff < 0` (A more negative): ligand B is more positive than A,
  so coupling B adds positive charge → couple a negative ion to compensate

The counterion should follow ligand B's lambda schedule (decoupled at λ=0, coupled at λ=1).

### Implementation Plan

1. Select a water molecule far from both ligands in the AB system topology.
2. Add that water's atoms to alchemical region B (so they get the same lambda treatment as ligand B).
3. Modify the NonbondedForce parameters of the water atoms at state B endpoint:
   - At lambda_B=1 (fully interacting): oxygen→ion parameters, hydrogens→zeroed charges
   - At lambda_B=0 (decoupled): just a normal water (since all interactions are off)
   
   Actually, since AlchemicalRegion uses annihilate_electrostatics=True, the factory
   handles the coupling. We need to set the *reference* parameters to the ion values.
   The factory will then smoothly turn these on/off with lambda.

4. Validation: mirror hybrid protocol requirements (solvent, PME, |diff|<=1).
5. Settings: reuse `explicit_charge_correction` / `explicit_charge_correction_cutoff` via the AFE AlchemicalSettings or add them to the SepTop-imported AlchemicalSettings.

### Which helpers to reuse

- `get_alchemical_waters` from `_rfe_utils/topologyhelpers.py`: Protocol-independent, can be used directly.
  It takes a topology, positions, charge_difference, and cutoff. Works for any OpenMM system.
- `_get_ion_and_water_parameters`: Also protocol-independent. Gets parameters from existing ions/waters in the system.
- `handle_alchemical_waters` / `_fix_alchemical_water_atom_mapping`: These are hybrid-topology-specific
  (they modify a "state B system" mapping that doesn't exist in SepTop). We need a SepTop-specific version.

Plan: Move `get_alchemical_waters` and `_get_ion_and_water_parameters` to a shared location
(`protocols/openmm_utils/charge_correction.py`) or import them from the RFE utils directly.
Given the codebase already imports from `_rfe_utils` within the RFE protocol, and to avoid
circular imports, we'll create a small shared utility in `openmm_utils/`.
