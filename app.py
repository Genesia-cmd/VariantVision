
import streamlit as st
from Bio.Align import PairwiseAligner
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd

st.set_page_config(
    page_title="VariantVision",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 VariantVision")
st.subheader("Protein Mutation Comparison and Analysis")

st.write(
    "Compare a reference protein sequence with a mutated sequence "
    "to identify amino acid changes and analyze protein properties."
)

reference = st.text_area(
    "Reference protein sequence",
    value="MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
)

variant = st.text_area(
    "Variant protein sequence",
    value="MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
)

def clean_sequence(sequence):
    return "".join(sequence.split()).upper()

reference = clean_sequence(reference)
variant = clean_sequence(variant)

if st.button("Analyze Sequences"):
    valid = set("ACDEFGHIKLMNPQRSTVWY")

    if not reference or not variant:
        st.error("Please enter both protein sequences.")
    elif set(reference) - valid or set(variant) - valid:
        st.error("Use a valid protein sequence containing standard amino-acid letters only.")
    else:
        st.markdown("### 1. Sequence comparison")

        aligner = PairwiseAligner()
        aligner.mode = "global"
        alignment = aligner.align(reference, variant)[0]

        st.code(str(alignment))

        if len(reference) == len(variant):
            changes = []
            for i, (ref_aa, var_aa) in enumerate(
                zip(reference, variant), start=1
            ):
                if ref_aa != var_aa:
                    changes.append({
                        "Position": i,
                        "Reference amino acid": ref_aa,
                        "Variant amino acid": var_aa,
                        "Change": f"{ref_aa}{i}{var_aa}"
                    })

            if changes:
                st.dataframe(pd.DataFrame(changes), use_container_width=True)
            else:
                st.info("No amino-acid substitutions detected.")
        else:
            st.warning(
                "The sequence lengths differ. The alignment is shown above, "
                "but substitution positions need alignment-based interpretation."
            )

        st.markdown("### 2. Protein properties")

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
                round(ref_analysis.isoelectric_point(), 2),
                round(ref_analysis.gravy(), 4)
            ],
            "Variant": [
                len(variant),
                round(var_analysis.molecular_weight(), 2),
                round(var_analysis.isoelectric_point(), 2),
                round(var_analysis.gravy(), 4)
            ]
        })

        st.dataframe(properties, use_container_width=True)

        st.download_button(
            "Download Results as CSV",
            data=properties.to_csv(index=False),
            file_name="variantvision_results.csv",
            mime="text/csv"
        )

        st.caption(
            "Note: These are computational predictions. They do not by "
            "themselves establish the clinical effect of a mutation."
        )

st.divider()
st.caption("VariantVision | Bioinformatics Project")

