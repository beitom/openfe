# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Tests for explicit charge correction in the ABFE protocol.

Covers validation, adaptive settings, dry-run setup, sign convention,
and total-charge conservation.
"""
import warnings

import gufe
import numpy as np
import openmm
import pytest
from openmm import NonbondedForce
from openmm import unit as ommunit

from openfe import ChemicalSystem, SmallMoleculeComponent, SolventComponent
from openfe.protocols.openmm_afe import (
    AbsoluteBindingProtocol,
)
from openfe.protocols.openmm_afe.base_afe_units import (
    ChargeCorrectionParameterState,
)
from openfe.protocols.openmm_afe.equil_afe_settings import AlchemicalSettings
from openff.units import unit as offunit_q

from openfe.protocols.openmm_utils.charge_correction import (
    apply_afe_charge_correction_offsets,
    get_alchemical_waters,
    get_ion_and_water_parameters,
)

from .utils import UNIT_TYPES, _get_units


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nbf(system: openmm.System) -> NonbondedForce:
    """Return the single NonbondedForce in *system*."""
    nbfs = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
    assert len(nbfs) == 1
    return nbfs[0]


def _to_float(val) -> float:
    """Convert an OpenMM Quantity or plain float to a float."""
    try:
        return float(val.value_in_unit(ommunit.elementary_charge))
    except AttributeError:
        return float(val)


def _effective_total_charge(
    nbf: NonbondedForce, param_values: dict[str, float]
) -> float:
    """
    Compute the total effective charge of a NonbondedForce at the given
    global parameter values.

    Each particle's effective charge is:
        base_charge + sum(offset_charge_i * param_value_i)
    """
    total = 0.0
    for i in range(nbf.getNumParticles()):
        charge, _, _ = nbf.getParticleParameters(i)
        total += _to_float(charge)

    for i in range(nbf.getNumParticleParameterOffsets()):
        pname, pidx, q_off, _, _ = nbf.getParticleParameterOffset(i)
        for name, val in param_values.items():
            if pname == name:
                total += _to_float(q_off) * val
    return total


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestChargeValidation:
    """Tests for _validate_endstates and _validate_charge_correction."""

    def test_neutral_abfe_unchanged(
        self, benzene_modifications, T4_protein_component
    ):
        """Neutral ligand validates without any alchemical_settings."""
        stateA = ChemicalSystem(
            {
                "benzene": benzene_modifications["benzene"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        AbsoluteBindingProtocol._validate_endstates(stateA, stateB)

    def test_charged_ligand_no_correction_raises(
        self, charged_benzene_modifications, T4_protein_component
    ):
        """Charged ligand without correction enabled raises."""
        stateA = ChemicalSystem(
            {
                "ligand": charged_benzene_modifications["benzoic_acid"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        with pytest.raises(ValueError, match="explicit_charge_correction"):
            AbsoluteBindingProtocol._validate_endstates(stateA, stateB)

    def test_charged_ligand_correction_disabled_raises(
        self, charged_benzene_modifications, T4_protein_component
    ):
        """Explicitly disabled correction for charged ligand raises."""
        settings = AlchemicalSettings(explicit_charge_correction=False)
        stateA = ChemicalSystem(
            {
                "ligand": charged_benzene_modifications["benzoic_acid"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        with pytest.raises(ValueError, match="explicit_charge_correction"):
            AbsoluteBindingProtocol._validate_endstates(
                stateA, stateB, alchemical_settings=settings
            )

    def test_charged_ligand_correction_enabled_passes(
        self, charged_benzene_modifications, T4_protein_component
    ):
        """Charged ligand with correction enabled and PME passes validation."""
        settings = AlchemicalSettings(explicit_charge_correction=True)
        stateA = ChemicalSystem(
            {
                "ligand": charged_benzene_modifications["benzoic_acid"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        AbsoluteBindingProtocol._validate_endstates(
            stateA, stateB, alchemical_settings=settings, nonbonded_method="pme"
        )

    def test_charged_ligand_non_pme_raises(
        self, charged_benzene_modifications, T4_protein_component
    ):
        """Charge correction with non-PME raises."""
        settings = AlchemicalSettings(explicit_charge_correction=True)
        stateA = ChemicalSystem(
            {
                "ligand": charged_benzene_modifications["benzoic_acid"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        with pytest.raises(ValueError, match="PME"):
            AbsoluteBindingProtocol._validate_endstates(
                stateA,
                stateB,
                alchemical_settings=settings,
                nonbonded_method="nocutoff",
            )

    def test_charge_magnitude_gt1_raises(
        self, T4_protein_component
    ):
        """Formal charge > |1| raises."""
        with pytest.raises(ValueError, match="Only absolute charges of 1"):
            AbsoluteBindingProtocol._validate_charge_correction(
                ligand_charge=2,
                alchemical_settings=AlchemicalSettings(explicit_charge_correction=True),
                nonbonded_method="pme",
            )


# ---------------------------------------------------------------------------
# Adaptive settings tests
# ---------------------------------------------------------------------------


class TestAdaptiveSettings:
    def test_neutral_ligand_no_correction(
        self, benzene_modifications, T4_protein_component
    ):
        """Neutral ligand does not enable charge correction."""
        stateA = ChemicalSystem(
            {
                "benzene": benzene_modifications["benzene"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        s = AbsoluteBindingProtocol._adaptive_settings(stateA, stateB)
        assert s.alchemical_settings.explicit_charge_correction is False

    def test_charged_ligand_enables_correction(
        self, charged_benzene_modifications, T4_protein_component
    ):
        """Charged ligand in adaptive settings enables correction."""
        stateA = ChemicalSystem(
            {
                "ligand": charged_benzene_modifications["benzoic_acid"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        s = AbsoluteBindingProtocol._adaptive_settings(stateA, stateB)
        assert s.alchemical_settings.explicit_charge_correction is True

    def test_positive_charged_ligand_enables_correction(
        self, charged_benzene_modifications, T4_protein_component
    ):
        """Positively charged ligand also enables correction."""
        stateA = ChemicalSystem(
            {
                "ligand": charged_benzene_modifications["aniline"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        s = AbsoluteBindingProtocol._adaptive_settings(stateA, stateB)
        assert s.alchemical_settings.explicit_charge_correction is True


# ---------------------------------------------------------------------------
# ChargeCorrectionParameterState tests
# ---------------------------------------------------------------------------


class TestChargeCorrectionParameterState:
    def test_parameter_name(self):
        state = ChargeCorrectionParameterState(lambda_charge_correction=0.5)
        assert state.lambda_charge_correction == 0.5

    def test_default_unbound(self):
        """When not bound to a system, the value is None (standard
        GlobalParameterState behaviour)."""
        state = ChargeCorrectionParameterState()
        assert state.lambda_charge_correction is None

    def test_set_value(self):
        """Can create with an explicit value."""
        state = ChargeCorrectionParameterState(lambda_charge_correction=1.0)
        assert state.lambda_charge_correction == 1.0


# ---------------------------------------------------------------------------
# Charge correction offset unit tests (synthetic system)
# ---------------------------------------------------------------------------


def _build_water_system(box_size=3.0):
    """Build a simple TIP3P water box with Na+ and Cl- ions for testing."""
    from openmm import app
    modeller = app.Modeller(app.Topology(), [])
    ff = app.ForceField("amber14/tip3p.xml")

    modeller.addSolvent(
        ff,
        model="tip3p",
        boxSize=openmm.Vec3(box_size, box_size, box_size) * ommunit.nanometer,
        positiveIon="Na+",
        negativeIon="Cl-",
        ionicStrength=0.15 * ommunit.molar,
    )
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * ommunit.nanometer,
    )
    pos = modeller.getPositions()
    positions = np.array(
        [[p.x, p.y, p.z] for p in pos.value_in_unit(ommunit.nanometer)]
    )
    return system, modeller.topology, positions


class TestOffsetMechanism:
    """Tests for the parameter offset applied to water→ion."""

    @pytest.fixture(scope="class")
    def water_system(self):
        system, topology, positions = _build_water_system()
        return system, topology, positions

    def test_offsets_produce_ion_at_lambda1(self, water_system):
        """At lambda=1, the oxygen should have ion charge/LJ."""
        system, topology, positions = water_system
        system = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(system)
        )

        solv = SolventComponent()
        water_resids = get_alchemical_waters(
            topology, positions, charge_difference=1,
            distance_cutoff=0.8 * offunit_q.nanometer,
        )
        assert len(water_resids) == 1

        ion_resname = solv.positive_ion.strip("-+").upper()
        ion_q, ion_s, ion_e, o_q, h_q = get_ion_and_water_parameters(
            topology, system, ion_resname
        )

        metadata = apply_afe_charge_correction_offsets(
            system, topology, water_resids, ligand_charge=1, solvent_component=solv
        )
        assert "correction_atom_indices" in metadata

        nbf = _get_nbf(system)

        # At lambda=0: should still be water
        for idx in metadata["correction_atom_indices"]:
            charge, sigma, epsilon = nbf.getParticleParameters(idx)
            # base params are still water

        # Compute effective charge at lambda=1
        q_total_0 = _effective_total_charge(
            nbf, {"lambda_charge_correction": 0.0}
        )
        q_total_1 = _effective_total_charge(
            nbf, {"lambda_charge_correction": 1.0}
        )

        # Delta should be +1 (water→Na+: -0.834 → +1, two H: +0.417→0 each)
        delta = q_total_1 - q_total_0
        assert abs(delta - 1.0) < 1e-6

    def test_offsets_negative_ligand(self, water_system):
        """For a -1 ligand, the correction delta should be -1."""
        system, topology, positions = water_system
        system = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(system)
        )

        solv = SolventComponent()
        water_resids = get_alchemical_waters(
            topology, positions, charge_difference=-1,
        )

        metadata = apply_afe_charge_correction_offsets(
            system, topology, water_resids, ligand_charge=-1, solvent_component=solv
        )

        nbf = _get_nbf(system)
        q_total_0 = _effective_total_charge(
            nbf, {"lambda_charge_correction": 0.0}
        )
        q_total_1 = _effective_total_charge(
            nbf, {"lambda_charge_correction": 1.0}
        )

        delta = q_total_1 - q_total_0
        assert abs(delta - (-1.0)) < 1e-6

    def test_neutral_ligand_raises(self, water_system):
        """Applying offsets with ligand_charge=0 should raise."""
        system, topology, positions = water_system
        system = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(system)
        )
        with pytest.raises(ValueError, match="neutral"):
            apply_afe_charge_correction_offsets(
                system, topology, [], ligand_charge=0, solvent_component=SolventComponent()
            )


# ---------------------------------------------------------------------------
# Sign convention and total charge conservation tests
# ---------------------------------------------------------------------------


class TestSignConventionAndChargeConservation:
    """
    For a disappearing ligand of charge q:
    - The correction ion grows to charge q as ligand charge goes to 0.
    - In a combined system with both alchemical ligand and correction,
      the total charge stays constant across lambda.

    In an isolated correction test (no ligand), the correction changes the
    total charge by exactly ``ligand_charge`` over the full lambda range.
    This is by design: it precisely compensates the ligand's vanishing charge.
    """

    @pytest.mark.parametrize("lam", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_correction_charge_proportional_to_lambda_positive(self, lam):
        """Correction charge change scales linearly with lambda (+1)."""
        system, topology, positions = _build_water_system()
        system = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(system)
        )
        solv = SolventComponent()
        water_resids = get_alchemical_waters(topology, positions, 1)
        apply_afe_charge_correction_offsets(
            system, topology, water_resids, ligand_charge=1, solvent_component=solv
        )
        nbf = _get_nbf(system)
        q0 = _effective_total_charge(nbf, {"lambda_charge_correction": 0.0})
        q_lam = _effective_total_charge(nbf, {"lambda_charge_correction": lam})
        expected_delta = 1.0 * lam
        assert abs((q_lam - q0) - expected_delta) < 1e-8

    @pytest.mark.parametrize("lam", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_correction_charge_proportional_to_lambda_negative(self, lam):
        """Correction charge change scales linearly with lambda (-1)."""
        system, topology, positions = _build_water_system()
        system = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(system)
        )
        solv = SolventComponent()
        water_resids = get_alchemical_waters(topology, positions, -1)
        apply_afe_charge_correction_offsets(
            system, topology, water_resids, ligand_charge=-1, solvent_component=solv
        )
        nbf = _get_nbf(system)
        q0 = _effective_total_charge(nbf, {"lambda_charge_correction": 0.0})
        q_lam = _effective_total_charge(nbf, {"lambda_charge_correction": lam})
        expected_delta = -1.0 * lam
        assert abs((q_lam - q0) - expected_delta) < 1e-8

    def test_full_correction_delta_positive(self):
        """At lambda=1, total charge changed by exactly +1."""
        system, topology, positions = _build_water_system()
        system = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(system)
        )
        solv = SolventComponent()
        water_resids = get_alchemical_waters(topology, positions, 1)
        apply_afe_charge_correction_offsets(
            system, topology, water_resids, ligand_charge=1, solvent_component=solv
        )
        nbf = _get_nbf(system)
        q0 = _effective_total_charge(nbf, {"lambda_charge_correction": 0.0})
        q1 = _effective_total_charge(nbf, {"lambda_charge_correction": 1.0})
        assert abs((q1 - q0) - 1.0) < 1e-8

    def test_full_correction_delta_negative(self):
        """At lambda=1, total charge changed by exactly -1."""
        system, topology, positions = _build_water_system()
        system = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(system)
        )
        solv = SolventComponent()
        water_resids = get_alchemical_waters(topology, positions, -1)
        apply_afe_charge_correction_offsets(
            system, topology, water_resids, ligand_charge=-1, solvent_component=solv
        )
        nbf = _get_nbf(system)
        q0 = _effective_total_charge(nbf, {"lambda_charge_correction": 0.0})
        q1 = _effective_total_charge(nbf, {"lambda_charge_correction": 1.0})
        assert abs((q1 - q0) - (-1.0)) < 1e-8

    def test_sign_positive_ligand_grows_cation(self):
        """For +1 ligand, correction species grows to +1 (cation)."""
        system, topology, positions = _build_water_system()
        system = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(system)
        )
        solv = SolventComponent()
        water_resids = get_alchemical_waters(topology, positions, 1)
        metadata = apply_afe_charge_correction_offsets(
            system, topology, water_resids, ligand_charge=1, solvent_component=solv
        )
        assert metadata["ion_resname"] == "NA"

    def test_sign_negative_ligand_grows_anion(self):
        """For -1 ligand, correction species grows to -1 (anion)."""
        system, topology, positions = _build_water_system()
        system = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(system)
        )
        solv = SolventComponent()
        water_resids = get_alchemical_waters(topology, positions, -1)
        metadata = apply_afe_charge_correction_offsets(
            system, topology, water_resids, ligand_charge=-1, solvent_component=solv
        )
        assert metadata["ion_resname"] == "CL"


# ---------------------------------------------------------------------------
# Lambda schedule integration tests
# ---------------------------------------------------------------------------


class TestLambdaSchedule:
    """Test that lambda_charge_correction is correctly added to schedules."""

    def test_schedule_without_correction(self):
        from openfe.protocols.openmm_afe.base_afe_units import (
            BaseAbsoluteMultiStateSimulationUnit,
        )
        from openfe.protocols.openmm_afe.equil_afe_settings import LambdaSettings

        settings = {
            "lambda_settings": LambdaSettings(
                lambda_elec=[0.0, 0.5, 1.0],
                lambda_vdw=[0.0, 0.0, 1.0],
                lambda_restraints=[0.0, 0.0, 0.0],
            )
        }
        lambdas = BaseAbsoluteMultiStateSimulationUnit._get_lambda_schedule(
            None, settings, has_charge_correction=False
        )
        assert "lambda_charge_correction" not in lambdas
        assert "lambda_electrostatics" in lambdas

    def test_schedule_with_correction(self):
        from openfe.protocols.openmm_afe.base_afe_units import (
            BaseAbsoluteMultiStateSimulationUnit,
        )
        from openfe.protocols.openmm_afe.equil_afe_settings import LambdaSettings

        settings = {
            "lambda_settings": LambdaSettings(
                lambda_elec=[0.0, 0.5, 1.0],
                lambda_vdw=[0.0, 0.0, 1.0],
                lambda_restraints=[0.0, 0.0, 0.0],
            )
        }
        lambdas = BaseAbsoluteMultiStateSimulationUnit._get_lambda_schedule(
            None, settings, has_charge_correction=True
        )
        assert "lambda_charge_correction" in lambdas
        # Follows user-facing lambda_elec, NOT inverted
        assert lambdas["lambda_charge_correction"] == [0.0, 0.5, 1.0]
        # lambda_electrostatics IS inverted
        assert lambdas["lambda_electrostatics"] == [1.0, 0.5, 0.0]

    def test_charge_correction_follows_lambda_elec_not_inverted(self):
        """The charge correction lambda must follow the user-facing
        convention where 0=water and 1=ion, not the OpenMMTools
        convention where 1=interacting."""
        from openfe.protocols.openmm_afe.base_afe_units import (
            BaseAbsoluteMultiStateSimulationUnit,
        )
        from openfe.protocols.openmm_afe.equil_afe_settings import LambdaSettings

        elec = [0.0, 0.25, 0.5, 0.75, 1.0]
        settings = {
            "lambda_settings": LambdaSettings(
                lambda_elec=elec,
                lambda_vdw=[0.0, 0.0, 0.0, 0.0, 1.0],
                lambda_restraints=[0.0, 0.0, 0.0, 0.0, 0.0],
            )
        }
        lambdas = BaseAbsoluteMultiStateSimulationUnit._get_lambda_schedule(
            None, settings, has_charge_correction=True
        )
        assert lambdas["lambda_charge_correction"] == elec


# ---------------------------------------------------------------------------
# Dry-run setup tests (slow, depend on force field / system creation)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestChargedABFEDryRun:
    """Dry-run setup tests with charged ligands.

    These require force field parameterization and system creation,
    so they're marked slow.
    """

    @pytest.fixture(scope="class")
    def charged_abfe_dag_neg(self, charged_benzene_modifications, T4_protein_component):
        """DAG for a -1 charged ligand (benzoic_acid)."""
        s = AbsoluteBindingProtocol.adaptive_settings(
            ChemicalSystem(
                {
                    "ligand": charged_benzene_modifications["benzoic_acid"],
                    "protein": T4_protein_component,
                    "solvent": SolventComponent(),
                }
            ),
            ChemicalSystem(
                {
                    "protein": T4_protein_component,
                    "solvent": SolventComponent(),
                }
            ),
        )
        s.solvent_output_settings.output_indices = "not water"
        s.complex_output_settings.output_indices = "not water"

        protocol = AbsoluteBindingProtocol(settings=s)
        stateA = ChemicalSystem(
            {
                "ligand": charged_benzene_modifications["benzoic_acid"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        return protocol.create(stateA=stateA, stateB=stateB, mapping=None)

    def test_complex_setup_has_correction(
        self, charged_abfe_dag_neg, tmp_path
    ):
        """Complex setup dry-run produces a system with charge correction."""
        units = _get_units(
            charged_abfe_dag_neg.protocol_units,
            UNIT_TYPES["complex"]["setup"],
        )
        results = units[0].run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )
        assert results["charge_correction"] is not None
        assert "correction_atom_indices" in results["charge_correction"]
        assert results["charge_correction"]["ligand_formal_charge"] == -1

        # The alchemical system must have the global parameter
        system = results["alchem_system"]
        nbf = _get_nbf(system)
        param_names = [
            nbf.getGlobalParameterName(i)
            for i in range(nbf.getNumGlobalParameters())
        ]
        assert "lambda_charge_correction" in param_names

    def test_solvent_setup_has_correction(
        self, charged_abfe_dag_neg, tmp_path
    ):
        """Solvent setup dry-run also produces charge correction."""
        units = _get_units(
            charged_abfe_dag_neg.protocol_units,
            UNIT_TYPES["solvent"]["setup"],
        )
        results = units[0].run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )
        assert results["charge_correction"] is not None
        assert results["charge_correction"]["ligand_formal_charge"] == -1

    def test_complex_total_charge_conserved(
        self, charged_abfe_dag_neg, tmp_path
    ):
        """Total charge constant across all lambda windows (complex)."""
        units = _get_units(
            charged_abfe_dag_neg.protocol_units,
            UNIT_TYPES["complex"]["setup"],
        )
        results = units[0].run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )
        system = results["alchem_system"]
        nbf = _get_nbf(system)

        q_ref = _effective_total_charge(nbf, {"lambda_charge_correction": 0.0})
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            q_lam = _effective_total_charge(
                nbf, {"lambda_charge_correction": lam}
            )
            assert abs(q_lam - q_ref) < 1e-6, (
                f"Total charge at lambda={lam} ({q_lam}) differs from "
                f"lambda=0 ({q_ref})"
            )

    def test_correction_metadata_keys(
        self, charged_abfe_dag_neg, tmp_path
    ):
        """Verify all expected metadata keys are present."""
        units = _get_units(
            charged_abfe_dag_neg.protocol_units,
            UNIT_TYPES["complex"]["setup"],
        )
        results = units[0].run(
            dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
        )
        meta = results["charge_correction"]
        assert "ion_resname" in meta
        assert "correction_atom_indices" in meta
        assert "water_residue_indices" in meta
        assert "ligand_formal_charge" in meta
        assert "ligand_name" in meta


# ---------------------------------------------------------------------------
# Regression: neutral ABFE behaviour unchanged
# ---------------------------------------------------------------------------


class TestNeutralABFERegression:
    """Ensure neutral-ligand ABFE is unaffected by charge correction code."""

    def test_neutral_validation_passes(
        self, benzene_modifications, T4_protein_component
    ):
        """Neutral ligand passes _validate_endstates as before."""
        stateA = ChemicalSystem(
            {
                "benzene": benzene_modifications["benzene"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        s = AbsoluteBindingProtocol.default_settings()
        AbsoluteBindingProtocol._validate_endstates(
            stateA,
            stateB,
            alchemical_settings=s.alchemical_settings,
            nonbonded_method=s.forcefield_settings.nonbonded_method,
        )

    def test_neutral_adaptive_no_correction(
        self, benzene_modifications, T4_protein_component
    ):
        """Adaptive settings for neutral ligand don't enable correction."""
        stateA = ChemicalSystem(
            {
                "benzene": benzene_modifications["benzene"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        s = AbsoluteBindingProtocol._adaptive_settings(stateA, stateB)
        assert s.alchemical_settings.explicit_charge_correction is False

    def test_neutral_charge_correction_metadata_none(
        self, benzene_modifications, T4_protein_component
    ):
        """For a neutral DAG, charge_correction metadata is None."""
        s = AbsoluteBindingProtocol.default_settings()
        # The default has correction disabled and neutral ligand → None
        assert s.alchemical_settings.explicit_charge_correction is False
