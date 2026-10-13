
import streamlit as st
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="VariantVision",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 VariantVision")
st.subheader("Protein Mutation Comparison and Analysis")

st.write(
    "A bioinformatics tool for comparing protein sequences, "
    "detecting substitutions, insertions and deletions, and "
    "analyzing physicochemical properties."
)

# ==================================================
# 2. DEFAULT HBB SEQUENCES
# ==================================================

REFERENCE_SEQUENCE = (
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
)

# Sickle-cell-associated example: E7V in full precursor numbering
VARIANT_SEQUENCE = (
    REFERENCE_SEQUENCE[:6]
    + "V"
    + REFERENCE_SEQUENCE[7:]
)

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


# ==================================================
# 3. HELPER FUNCTIONS
# ==================================================

def clean_sequence(sequence):
    return "".join(sequence.split()).upper()


def analyze_mutations(reference, variant):
    """Detect sequence changes using global alignment coordinates."""

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -3
    aligner.extend_gap_score = -0.5

    alignment = aligner.align(reference, variant)[0]
    coordinates = alignment.coordinates

    mutations = []

    # Each pair of consecutive alignment coordinates defines
    # a matching, substitution, insertion or deletion segment.
    for i in range(coordinates.shape[1] - 1):

        r0 = int(coordinates[0, i])
        r1 = int(coordinates[0, i + 1])
        v0 = int(coordinates[1, i])
        v1 = int(coordinates[1, i + 1])

        ref_length = r1 - r0
        var_length = v1 - v0

        # Substitution or matching aligned residues
        if ref_length > 0 and var_length > 0:

            # Diagonal alignment segments have equal lengths.
            if ref_length == var_length:
                for offset in range(ref_length):
                    ref_aa = reference[r0 + offset]
                    var_aa = variant[v0 + offset]

                    if ref_aa != var_aa:
                        position = r0 + offset + 1

                        mutations.append({
                            "Type": "Substitution",
                            "Reference position": position,
                            "Variant position": v0 + offset + 1,
                            "Reference amino acid": ref_aa,
                            "Variant amino acid": var_aa,
                            "Change": f"{ref_aa}{position}{var_aa}"
                        })

            else:
                # Defensive fallback for any unequal diagonal block.
                # Standard pairwise alignments normally use equal
                # lengths for diagonal segments.
                common = min(ref_length, var_length)

                for offset in range(common):
                    ref_aa = reference[r0 + offset]
                    var_aa = variant[v0 + offset]

                    if ref_aa != var_aa:
                        position = r0 + offset + 1

                        mutations.append({
                            "Type": "Substitution",
                            "Reference position": position,
                            "Variant position": v0 + offset + 1,
                            "Reference amino acid": ref_aa,
                            "Variant amino acid": var_aa,
                            "Change": f"{ref_aa}{position}{var_aa}"
                        })

                for offset in range(common, ref_length):
                    position = r0 + offset + 1
                    ref_aa = reference[r0 + offset]

                    mutations.append({
                        "Type": "Deletion",
                        "Reference position": position,
                        "Variant position": None,
                        "Reference amino acid": ref_aa,
                        "Variant amino acid": "-",
                        "Change": f"del {ref_aa}{position}"
                    })

                for offset in range(common, var_length):
                    inserted_aa = variant[v0 + offset]

                    mutations.append({
                        "Type": "Insertion",
                        "Reference position": r1,
                        "Variant position": v0 + offset + 1,
                        "Reference amino acid": "-",
                        "Variant amino acid": inserted_aa,
                        "Change": f"ins {inserted_aa} after {r1}"
                    })

        # Deletion: reference advances, variant has a gap
        elif ref_length > 0:

            for offset in range(ref_length):
                position = r0 + offset + 1
                ref_aa = reference[r0 + offset]

                mutations.append({
                    "Type": "Deletion",
                    "Reference position": position,
                    "Variant position": None,
                    "Reference amino acid": ref_aa,
                    "Variant amino acid": "-",
                    "Change": f"del {ref_aa}{position}"
                })

        # Insertion: variant advances, reference has a gap
        elif var_length > 0:

            for offset in range(var_length):
                inserted_aa = variant[v0 + offset]

                mutations.append({
                    "Type": "Insertion",
                    "Reference position": r0,
                    "Variant position": v0 + offset + 1,
                    "Reference amino acid": "-",
                    "Variant amino acid": inserted_aa,
                    "Change": f"ins {inserted_aa} after {r0}"
                })

    return alignment, mutations


# ==================================================
# 4. SEQUENCE INPUT
# ==================================================

st.header("1. Protein Sequence Input")

st.caption(
    "Default example: human HBB protein with an E7V substitution. "
    "Edit either sequence to test other variants."
)

reference_input = st.text_area(
    "Reference protein sequence",
    value=REFERENCE_SEQUENCE,
    height=130
)

variant_input = st.text_area(
    "Variant protein sequence",
    value=VARIANT_SEQUENCE,
    height=130
)

analyze_button = st.button(
    "🔍 Analyze Sequences",
    type="primary",
    use_container_width=True
)

# ==================================================
# 5. ANALYSIS
# ==================================================

if analyze_button:

    reference = clean_sequence(reference_input)
    variant = clean_sequence(variant_input)

    if not reference or not variant:
        st.error("Please enter both protein sequences.")

    elif set(reference) - VALID_AMINO_ACIDS:
        st.error(
            "The reference sequence contains invalid characters. "
            "Use standard one-letter amino-acid codes only."
        )

    elif set(variant) - VALID_AMINO_ACIDS:
        st.error(
            "The variant sequence contains invalid characters. "
            "Use standard one-letter amino-acid codes only."
        )

    else:

        alignment, mutations = analyze_mutations(
            reference, variant
        )

        st.success("Sequence analysis completed!")

        # ------------------------------------------
        # 5A. SEQUENCE SUMMARY
        # ------------------------------------------

        st.header("2. Sequence Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Reference Length",
            f"{len(reference)} amino acids"
        )

        col2.metric(
            "Variant Length",
            f"{len(variant)} amino acids"
        )

        col3.metric(
            "Changes Detected",
            len(mutations)
        )

        if len(reference) == len(variant):
            st.info("Both sequences have the same length.")
        else:
            st.info(
                "The sequences have different lengths. "
                "Alignment-based analysis is being used to "
                "identify substitutions, insertions and deletions."
            )

        # ------------------------------------------
        # 5B. GLOBAL ALIGNMENT
        # ------------------------------------------

        st.header("3. Global Sequence Alignment")

        st.write(
            "Global alignment compares the sequences across "
            "their full lengths. Gaps help represent possible "
            "insertions and deletions."
        )

        st.code(str(alignment), language=None)

        # ------------------------------------------
        # 5C. MUTATION DETECTION
        # ------------------------------------------

        st.header("4. Mutation Detection")

        if mutations:

            mutation_df = pd.DataFrame(mutations)

            st.dataframe(
                mutation_df,
                use_container_width=True,
                hide_index=True
            )

            substitution_count = sum(
                item["Type"] == "Substitution"
                for item in mutations
            )

            insertion_count = sum(
                item["Type"] == "Insertion"
                for item in mutations
            )

            deletion_count = sum(
                item["Type"] == "Deletion"
                for item in mutations
            )

            c1, c2, c3 = st.columns(3)

            c1.metric("Substitutions", substitution_count)
            c2.metric("Insertions", insertion_count)
            c3.metric("Deletions", deletion_count)

            st.download_button(
                "⬇️ Download Mutation Results (CSV)",
                data=mutation_df.to_csv(index=False),
                file_name="variantvision_mutations.csv",
                mime="text/csv"
            )

        else:
            st.success(
                "No amino-acid changes detected. "
                "The aligned sequences are identical."
            )

        # ------------------------------------------
        # 5D. PROTEIN PROPERTIES
        # ------------------------------------------

        st.header("5. Protein Physicochemical Properties")

        try:
            ref_analysis = ProteinAnalysis(reference)
            var_analysis = ProteinAnalysis(variant)

            properties = pd.DataFrame({
                "Property": [
                    "Sequence length (amino acids)",
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

            st.subheader("Molecular Weight Comparison")

            weight_comparison = pd.DataFrame({
                "Sequence": ["Reference", "Variant"],
                "Molecular weight (Da)": [
                    ref_analysis.molecular_weight(),
                    var_analysis.molecular_weight()
                ]
            }).set_index("Sequence")

            st.bar_chart(weight_comparison)

            st.caption(
                "Theoretical pI is the estimated pH at which a protein "
                "has no net charge. GRAVY is the Grand Average of "
                "Hydropathicity. These are computational estimates."
            )

            st.download_button(
                "⬇️ Download Protein Properties (CSV)",
                data=properties.to_csv(index=False),
                file_name="variantvision_properties.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(
                "Protein property calculation failed. "
                "Please check the input sequences. "
                f"Details: {error}"
            )

        # ------------------------------------------
        # 5E. BIOLOGICAL INTERPRETATION
        # ------------------------------------------

        st.header("6. Biological Interpretation")

        if (
            reference == REFERENCE_SEQUENCE
            and variant == VARIANT_SEQUENCE
        ):

            st.markdown(
                """
                **Case study: Sickle-cell-associated HBB variant**

                - **Gene:** HBB
                - **Reference amino acid:** Glutamic acid (E)
                - **Variant amino acid:** Valine (V)
                - **Full precursor numbering:** p.Glu7Val (E7V)
                - **Mature beta-globin numbering:** Glu6Val (E6V)

                This substitution replaces a negatively charged amino
                acid with a hydrophobic amino acid. Under low-oxygen
                conditions, hemoglobin S can polymerize, contributing
                to red blood cell deformation and sickle-cell disease.
                """
            )

        elif mutations:

            st.write(
                "Sequence differences were detected. Their biological "
                "effects depend on the protein, the position of each "
                "change, structural context and experimental evidence. "
                "Sequence comparison alone cannot determine whether "
                "an unknown variant causes disease."
            )

        else:

            st.write(
                "No amino-acid differences were detected. "
                "The reference and variant sequences are identical."
            )

        st.warning(
            "Educational limitation: this tool identifies sequence "
            "differences and calculates theoretical properties. "
            "It does not diagnose disease or independently establish "
            "the clinical significance of an unknown variant."
        )

# ==================================================
# 6. FOOTER
# ==================================================

st.divider()

st.caption(
    "VariantVision | Concepts of Bioinformatics | "
    "Protein Sequence Analysis"
)
