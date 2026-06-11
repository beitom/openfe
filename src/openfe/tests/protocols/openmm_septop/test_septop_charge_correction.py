# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Tests for SepTop net charge correction support."""

import logging

import pytest

from openfe import ChemicalSystem, SolventComponent
from openfe.protocols.openmm_septop import SepTopProtocol
from openfe.protocols.openmm_septop.equil_septop_method import (
    _check_alchemical_charge_difference,
    _validate_charge_difference,
)

# ============================================================================
# Validation tests
# ============================================================================


class TestValidateChargeDifference:
    """Tests for _validate_charge_difference."""

    def test_same_charge_passes(self, benzene_modifications):
        """Same charge returns 0 and does not raise."""
        result = _validate_charge_difference(
            ligandA=benzene_modifications["benzene"],
            ligandB=benzene_modifications["toluene"],
            explicit_charge_correction=False,
            nonbonded_method="pme",
            has_solvent=True,
        )
        assert result == 0

    def test_charge_diff_without_correction_raises(self, charged_benzene_modifications):
        """Charge difference without explicit correction enabled should raise."""
        errmsg = "A charge difference of 1 is observed"
        with pytest.raises(ValueError, match=errmsg):
            _validate_charge_difference(
                ligandA=charged_benzene_modifications["benzene"],
                ligandB=charged_benzene_modifications["benzoic_acid"],
                explicit_charge_correction=False,
                nonbonded_method="pme",
                has_solvent=True,
            )

    def test_charge_diff_with_correction_passes(self, charged_benzene_modifications):
        """Charge difference with correction enabled should succeed."""
        result = _validate_charge_difference(
            ligandA=charged_benzene_modifications["benzene"],
            ligandB=charged_benzene_modifications["benzoic_acid"],
            explicit_charge_correction=True,
            nonbonded_method="pme",
            has_solvent=True,
        )
        assert result == 1

    def test_charge_diff_no_solvent_raises(self, charged_benzene_modifications):
        """Charge correction without solvent should raise."""
        errmsg = "Cannot use explicit charge correction without solvent"
        with pytest.raises(ValueError, match=errmsg):
            _validate_charge_difference(
                ligandA=charged_benzene_modifications["benzene"],
                ligandB=charged_benzene_modifications["benzoic_acid"],
                explicit_charge_correction=True,
                nonbonded_method="pme",
                has_solvent=False,
            )

    def test_charge_diff_no_pme_raises(self, charged_benzene_modifications):
        """Charge correction without PME should raise."""
        errmsg = "Explicit charge correction when not using PME"
        with pytest.raises(ValueError, match=errmsg):
            _validate_charge_difference(
                ligandA=charged_benzene_modifications["benzene"],
                ligandB=charged_benzene_modifications["benzoic_acid"],
                explicit_charge_correction=True,
                nonbonded_method="nocutoff",
                has_solvent=True,
            )

    def test_charge_diff_greater_than_one_raises(self, charged_benzene_modifications):
        """Charge differences > 1 should raise even with correction enabled."""
        # aniline has +1, benzoic acid has -1, so diff = 2
        errmsg = "Only absolute differences of 1 are currently supported"
        with pytest.raises(ValueError, match=errmsg):
            _validate_charge_difference(
                ligandA=charged_benzene_modifications["aniline"],
                ligandB=charged_benzene_modifications["benzoic_acid"],
                explicit_charge_correction=True,
                nonbonded_method="pme",
                has_solvent=True,
            )

    def test_sign_convention_positive_diff(
        self, charged_benzene_modifications, caplog
    ):
        """Positive diff (A more positive) → negative counterion logged."""
        caplog.set_level(logging.INFO)
        result = _validate_charge_difference(
            ligandA=charged_benzene_modifications["benzene"],
            ligandB=charged_benzene_modifications["benzoic_acid"],
            explicit_charge_correction=True,
            nonbonded_method="pme",
            has_solvent=True,
        )
        assert result == 1
        assert "negative" in caplog.text

    def test_sign_convention_negative_diff(
        self, charged_benzene_modifications, caplog
    ):
        """Negative diff (A more negative) → positive counterion logged."""
        caplog.set_level(logging.INFO)
        result = _validate_charge_difference(
            ligandA=charged_benzene_modifications["benzoic_acid"],
            ligandB=charged_benzene_modifications["benzene"],
            explicit_charge_correction=True,
            nonbonded_method="pme",
            has_solvent=True,
        )
        assert result == -1
        assert "positive" in caplog.text


class TestLegacyCheckAlchemicalChargeDifference:
    """Tests that the legacy function still works for backward compat."""

    def test_still_rejects_charge_change(self, charged_benzene_modifications):
        errmsg = "A charge difference of 1"
        with pytest.raises(ValueError, match=errmsg):
            _check_alchemical_charge_difference(
                charged_benzene_modifications["benzene"],
                charged_benzene_modifications["benzoic_acid"],
            )


class TestProtocolValidation:
    """Integration tests for SepTopProtocol._validate with charge changes."""

    def test_validate_rejects_charge_without_correction(
        self, charged_benzene_modifications, T4_protein_component
    ):
        """Protocol validation rejects charge change when correction is off."""
        settings = SepTopProtocol.default_settings()
        settings.alchemical_settings.explicit_charge_correction = False
        protocol = SepTopProtocol(settings=settings)

        stateA = ChemicalSystem(
            {
                "ligand": charged_benzene_modifications["benzene"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "ligand": charged_benzene_modifications["benzoic_acid"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )

        errmsg = "A charge difference of 1"
        with pytest.raises(ValueError, match=errmsg):
            protocol.create(stateA=stateA, stateB=stateB, mapping=None)

    def test_validate_accepts_charge_with_correction(
        self, charged_benzene_modifications, T4_protein_component
    ):
        """Protocol validation passes charge change when correction is on."""
        settings = SepTopProtocol.default_settings()
        settings.alchemical_settings.explicit_charge_correction = True
        protocol = SepTopProtocol(settings=settings)

        stateA = ChemicalSystem(
            {
                "ligand": charged_benzene_modifications["benzene"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "ligand": charged_benzene_modifications["benzoic_acid"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )

        # Should not raise - creates the DAG successfully
        dag = protocol.create(stateA=stateA, stateB=stateB, mapping=None)
        assert dag is not None

    def test_same_charge_still_works(
        self, benzene_modifications, T4_protein_component
    ):
        """Same-charge transformations continue to work unchanged."""
        settings = SepTopProtocol.default_settings()
        protocol = SepTopProtocol(settings=settings)

        stateA = ChemicalSystem(
            {
                "ligand": benzene_modifications["benzene"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "ligand": benzene_modifications["toluene"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )

        dag = protocol.create(stateA=stateA, stateB=stateB, mapping=None)
        assert dag is not None


class TestAdaptiveSettings:
    """Tests for _adaptive_settings auto-enabling charge correction."""

    def test_adaptive_enables_correction_for_charge_change(
        self, charged_benzene_modifications, T4_protein_component
    ):
        """_adaptive_settings auto-enables correction for charge-changing."""
        stateA = ChemicalSystem(
            {
                "ligand": charged_benzene_modifications["benzene"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "ligand": charged_benzene_modifications["benzoic_acid"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )

        settings = SepTopProtocol._adaptive_settings(stateA, stateB)
        assert settings.alchemical_settings.explicit_charge_correction is True

    def test_adaptive_does_not_enable_for_same_charge(
        self, benzene_modifications, T4_protein_component
    ):
        """_adaptive_settings does not enable correction for same-charge."""
        stateA = ChemicalSystem(
            {
                "ligand": benzene_modifications["benzene"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )
        stateB = ChemicalSystem(
            {
                "ligand": benzene_modifications["toluene"],
                "protein": T4_protein_component,
                "solvent": SolventComponent(),
            }
        )

        settings = SepTopProtocol._adaptive_settings(stateA, stateB)
        assert settings.alchemical_settings.explicit_charge_correction is False
