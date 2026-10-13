
import streamlit as st
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="VariantVision",
    page_icon="🧬",
    layout="wide"
)

# ---------------- REFERENCE SEQUENCE ----------------

# Human hemoglobin subunit beta (HBB)
# Reference accession: NCBI RefSeq NP_000509.1
# Full precursor sequence, including initial methionine

REFERENCE_SEQUENCE = (
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
)

# Sickle-cell-associated example:
# E7V in full precursor numbering
# E6V in mature beta-globin chain numbering

VARIANT_SEQUENCE = (
    REFERENCE_SEQUENCE[:6]
    + "V"
    + REFERENCE_SEQUENCE[7:]
)

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


# ---------------- HELPER FUNCTION ----------------

def clean_sequence(sequence):
    return "".join(sequence.split()).upper()


def get_mutations(reference, variant):
    mutations = []

    if len(reference) != len(variant):
        return mutations

    for position, (ref_aa, var_aa) in enumerate(
        zip(reference, variant), start=1
    ):
        if ref_aa != var_aa:
            mutations.append({
                "Position": position,
                "Reference": ref_aa,
                "Variant": var_aa,
                "Mutation": f"{ref_aa}{position}{var_aa}"
            })

    return mutations


# ---------------- DASHBOARD HEADER ----------------

st.title("🧬 VariantVision")

st.subheader("Protein Mutation Comparison and Analysis")

st.write(
    "VariantVision is a bioinformatics tool that compares "
    "reference and variant protein sequences, detects amino-acid "
    "changes, calculates protein properties, and presents "
    "biological interpretation."
)

st.info(
    "Case study: Human hemoglobin beta (HBB) and the "
    "sickle-cell-associated E7V substitution."
)

st.divider()


# ---------------- SEQUENCE INPUT ----------------

st.header("1. Protein Sequence Input")

st.caption(
    "The default sequences demonstrate the HBB E7V variant. "
    "You can edit either sequence to test other substitutions."
)

reference_input = st.text_area(
    "Reference protein sequence",
    value=REFERENCE_SEQUENCE,
    height=130,
    key="reference"
)

variant_input = st.text_area(
    "Variant protein sequence",
    value=VARIANT_SEQUENCE,
    height=130,
    key="variant"
)

analyze = st.button(
    "🔍 Analyze Sequences",
    type="primary",
    use_container_width=True
)


# ---------------- SEQUENCE ANALYSIS ----------------

if analyze:

    reference = clean_sequence(reference_input)
    variant = clean_sequence(variant_input)

    # Validate sequence inputs

    if not reference or not variant:
        st.error("Please enter both protein sequences.")

    elif set(reference) - VALID_AMINO_ACIDS:
        st.error(
            "The reference contains invalid characters. "
            "Use standard one-letter amino-acid codes only."
        )

    elif set(variant) - VALID_AMINO_ACIDS:
        st.error(
            "The variant contains invalid characters. "
            "Use standard one-letter amino-acid codes only."
        )

    else:

        st.success("Both protein sequences passed basic validation.")

        # ---------------- BASIC SUMMARY ----------------

        st.header("2. Sequence Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Reference Length", f"{len(reference)} aa")
        col2.metric("Variant Length", f"{len(variant)} aa")

        mutations = get_mutations(reference, variant)

        col3.metric("Substitutions Detected", len(mutations))

        # ---------------- ALIGNMENT ----------------

        st.header("3. Sequence Alignment")

        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 1
        aligner.mismatch_score = -1
        aligner.open_gap_score = -2
        aligner.extend_gap_score = -0.5

        alignment = aligner.align(reference, variant)[0]

        st.write(
            "Global alignment compares the sequences across their "
            "full lengths and helps identify similarities and differences."
        )

        st.code(str(alignment), language=None)

        # ---------------- MUTATION DETECTION ----------------

        st.header("4. Mutation Detection")

        if len(reference) == len(variant):

            if mutations:

                mutation_table = pd.DataFrame(mutations)

                st.dataframe(
                    mutation_table,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("**Sequence comparison**")

                # Highlight substitutions in a simple aligned display

                ref_display = []
                var_display = []
                change_display = []

                for ref_aa, var_aa in zip(reference, variant):
                    ref_display.append(ref_aa)
                    var_display.append(var_aa)
                    change_display.append(
                        "|" if ref_aa == var_aa else "*"
                    )

                st.code(
                    "Reference: " + "".join(ref_display[:100])
                    + ("\n           " + "".join(change_display[:100]))
                    + "\nVariant:   " + "".join(var_display[:100]),
                    language=None
                )

                if len(reference) > 100:
                    st.caption(
                        "The display above shows the first 100 residues. "
                        "The complete alignment is available in the section above."
                    )

            else:
                st.success(
                    "No amino-acid substitutions detected. "
                    "The sequences are identical."
                )

        else:
            st.warning(
                "The sequence lengths differ. Insertions or deletions "
                "may be present. Review the global alignment above; "
                "the simple position-by-position mutation table is "
                "not shown for sequences of different lengths."
            )

        # ---------------- PROTEIN PROPERTIES ----------------

        st.header("5. Protein Physicochemical Properties")

        try:
            ref_analysis = ProteinAnalysis(reference)
            var_analysis = ProteinAnalysis(variant)

            properties = pd.DataFrame({
                "Property": [
                    "Sequence length",
                    "Molecular weight (Da)",
                    "Theoretical isoelectric point (pI)",
                    "GRAVY score"
                ],
                "Reference": [
                    len(reference),
                    round(ref_analysis.molecular_weight(), 2),
                    round(ref_analysis.isoelectric_point(), 3),
                    round(ref_analysis.gravy(), 4)
                ],
                "Variant": [
                    len(variant),
                    round(var_analysis.molecular_weight(), 2),
                    round(var_analysis.isoelectric_point(), 3),
                    round(var_analysis.gravy(), 4)
                ]
            })

            st.dataframe(
                properties,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "GRAVY means Grand Average of Hydropathicity. "
                "These values are computational estimates, not "
                "direct experimental measurements."
            )

            st.bar_chart(
                properties.set_index("Property")[
                    ["Reference", "Variant"]
                ]
            )

            # Download results

            st.download_button(
                label="⬇️ Download Protein Properties (CSV)",
                data=properties.to_csv(index=False),
                file_name="variantvision_properties.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(
                "Unable to calculate protein properties. "
                f"Please check the sequence. Details: {error}"
            )

        # ---------------- BIOLOGICAL INTERPRETATION ----------------

        st.header("6. Biological Significance")

        if reference == REFERENCE_SEQUENCE and variant == VARIANT_SEQUENCE:

            st.markdown(
                """
                **Case study: Sickle-cell-associated HBB variant**

                - **Gene:** HBB, which encodes the beta subunit of hemoglobin.
                - **Reference residue:** Glutamic acid (E).
                - **Variant residue:** Valine (V).
                - **Full precursor numbering:** p.Glu7Val (E7V).
                - **Mature beta-globin numbering:** Glu6Val (E6V).

                The substitution replaces a negatively charged amino acid
                with a hydrophobic amino acid. Under low-oxygen conditions,
                hemoglobin S can polymerize, contributing to red blood
                cell deformation and the clinical features of sickle-cell
                disease.
                """
            )

        elif mutations:
            st.write(
                "One or more amino-acid differences were detected. "
                "Their biological significance cannot be determined "
                "from sequence comparison and physicochemical properties "
                "alone. Further biological evidence is required."
            )

        else:
            st.write(
                "No amino-acid differences were detected. "
                "The sequences have the same amino-acid sequence."
            )

        st.warning(
            "Educational limitation: this tool identifies sequence "
            "differences and estimates protein properties. It does not "
            "independently diagnose disease or establish the clinical "
            "significance of an unknown variant."
        )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "VariantVision | Concepts of Bioinformatics | "
    "Protein Sequence Analysis"
)
