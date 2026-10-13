
import io
import re
import requests
import pandas as pd
import streamlit as st

from Bio.Align import PairwiseAligner
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="VariantVision",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 VariantVision")
st.subheader(
    "Interactive Protein Sequence Comparison, "
    "Mutation Detection and Variant Search"
)

st.write(
    "An educational bioinformatics platform for protein sequence "
    "analysis, mutation comparison and exploration of documented "
    "genetic variants."
)

st.info(
    "Educational project only. ClinVar interpretations are submitted "
    "and reviewed according to the database's processes. This app is "
    "not a diagnostic tool."
)

# Official NCBI E-utilities endpoint
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# Standard amino acids
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Example beta-globin protein sequence
HBB_REFERENCE = (
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVK"
    "AHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFT"
    "PPVQAAYQKVVAGVANALAHKYH"
)

HBB_VARIANT = HBB_REFERENCE[:6] + "V" + HBB_REFERENCE[7:]


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def clean_sequence(raw_text):
    """Extract a protein sequence from plain text or FASTA."""

    lines = raw_text.strip().splitlines()

    # Remove FASTA header lines
    sequence_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith(">")
    ]

    sequence = "".join(sequence_lines)
    sequence = re.sub(r"\s+", "", sequence).upper()

    # Reject invalid characters rather than silently removing them
    if not sequence:
        raise ValueError("No protein sequence was found.")

    invalid = sorted(set(sequence) - VALID_AA)

    if invalid:
        raise ValueError(
            "Invalid protein character(s): "
            + ", ".join(invalid)
            + ". Use the 20 standard amino-acid letters only."
        )

    return sequence


def get_sequence(pasted_text, uploaded_file):
    """Use the uploaded FASTA file if supplied; otherwise use pasted text."""

    if uploaded_file is not None:
        raw = uploaded_file.getvalue().decode("utf-8")
    else:
        raw = pasted_text

    return clean_sequence(raw)


def compare_sequences(reference, variant):
    """Globally align two protein sequences and report differences."""

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5

    alignment = aligner.align(reference, variant)[0]
    coordinates = alignment.coordinates
    changes = []

    # Walk through the aligned coordinate blocks
    for i in range(coordinates.shape[1] - 1):
        r0 = int(coordinates[0, i])
        r1 = int(coordinates[0, i + 1])
        v0 = int(coordinates[1, i])
        v1 = int(coordinates[1, i + 1])

        ref_length = r1 - r0
        var_length = v1 - v0

        # Substitutions or diagonal alignment block
        if ref_length > 0 and var_length > 0:
            if ref_length == var_length:
                for k in range(ref_length):
                    ref_aa = reference[r0 + k]
                    var_aa = variant[v0 + k]

                    if ref_aa != var_aa:
                        changes.append({
                            "Type": "Substitution",
                            "Reference position": r0 + k + 1,
                            "Reference residue": ref_aa,
                            "Variant position": v0 + k + 1,
                            "Variant residue": var_aa,
                            "Change": (
                                f"{ref_aa}{r0 + k + 1}{var_aa}"
                            )
                        })
            else:
                # Defensive fallback for a complex alignment block
                changes.append({
                    "Type": "Complex alignment block",
                    "Reference position": f"{r0 + 1}-{r1}",
                    "Reference residue": reference[r0:r1],
                    "Variant position": f"{v0 + 1}-{v1}",
                    "Variant residue": variant[v0:v1],
                    "Change": "Review aligned sequences"
                })

        # Residues present only in the reference: deletion
        elif ref_length > 0:
            deleted = reference[r0:r1]

            changes.append({
                "Type": "Deletion",
                "Reference position": (
                    f"{r0 + 1}-{r1}" if ref_length > 1 else r0 + 1
                ),
                "Reference residue": deleted,
                "Variant position": v0,
                "Variant residue": "-",
                "Change": f"del {deleted}"
            })

        # Residues present only in the variant: insertion
        elif var_length > 0:
            inserted = variant[v0:v1]

            changes.append({
                "Type": "Insertion",
                "Reference position": (
                    f"after {r0}" if r0 > 0 else "before 1"
                ),
                "Reference residue": "-",
                "Variant position": f"{v0 + 1}-{v1}",
                "Variant residue": inserted,
                "Change": f"ins {inserted}"
            })

    return alignment, changes


def protein_properties(sequence, label):
    """Calculate basic physicochemical properties."""

    analysis = ProteinAnalysis(sequence)

    return {
        "Sequence": label,
        "Length (amino acids)": len(sequence),
        "Molecular weight (Da)": round(
            analysis.molecular_weight(), 2
        ),
        "Theoretical pI": round(
            analysis.isoelectric_point(), 3
        ),
        "GRAVY": round(analysis.gravy(), 4)
    }


def fetch_clinvar_records(query, search_type, max_records):
    """Search ClinVar and retrieve summary information using NCBI E-utilities."""

    query = query.strip()

    if search_type == "Gene":
        # Search the ClinVar gene field
        search_term = f"{query}[gene]"
    else:
        # Use a general ClinVar search for disease/phenotype terms
        search_term = query

    headers = {
        "User-Agent": "VariantVisionEducationalProject/1.0"
    }

    # 1. Find matching ClinVar record IDs
    search_response = requests.get(
        NCBI_BASE + "esearch.fcgi",
        params={
            "db": "clinvar",
            "term": search_term,
            "retmax": max_records,
            "retmode": "json"
        },
        headers=headers,
        timeout=25
    )
    search_response.raise_for_status()
    search_data = search_response.json()

    ids = search_data.get("esearchresult", {}).get("idlist", [])

    if not ids:
        return []

    # 2. Retrieve summaries for the returned record IDs
    summary_response = requests.get(
        NCBI_BASE + "esummary.fcgi",
        params={
            "db": "clinvar",
            "id": ",".join(ids),
            "retmode": "json"
        },
        headers=headers,
        timeout=25
    )
    summary_response.raise_for_status()
    summary_data = summary_response.json()

    result_data = summary_data.get("result", {})
    records = []

    for record_id in ids:
        record = result_data.get(record_id, {})

        if not isinstance(record, dict):
            continue

        # Extract fields defensively because records can vary
        genes = record.get("genes", [])
        gene_names = []

        if isinstance(genes, list):
            for gene in genes:
                if isinstance(gene, dict):
                    name = gene.get("symbol")
                    if name:
                        gene_names.append(name)

        traits = record.get("trait_set", [])
        trait_names = []

        if isinstance(traits, list):
            for trait in traits:
                if isinstance(trait, dict):
                    name = trait.get("trait_name")
                    if name:
                        trait_names.append(name)

        significance = record.get("clinical_significance", {})

        if isinstance(significance, dict):
            significance_text = significance.get(
                "description", "Not provided in summary"
            )
            review_status = significance.get(
                "review_status", "Not provided in summary"
            )
        else:
            significance_text = str(significance or "Not provided")
            review_status = "Not provided in summary"

        accession = record.get("accession", "")
        title = record.get("title", "Title not provided")
        variation_type = record.get("variant_type", "")

        records.append({
            "ClinVar ID": record_id,
            "Accession": accession,
            "Title / Variant": title,
            "Gene": ", ".join(gene_names) or "Not listed in summary",
            "Associated phenotype": (
                "; ".join(trait_names) or "Not listed in summary"
            ),
            "Clinical significance": significance_text,
            "Review status": review_status,
            "Variant type": variation_type,
            "ClinVar URL": (
                f"https://www.ncbi.nlm.nih.gov/clinvar/variation/"
                f"{record_id}/"
            )
        })

    return records


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

mode = st.sidebar.radio(
    "Choose an analysis mode",
    [
        "Home",
        "General Protein Analysis",
        "Disease / Gene Search",
        "About the Project"
    ]
)


# --------------------------------------------------
# HOME
# --------------------------------------------------

if mode == "Home":

    st.header("Welcome to VariantVision")

    st.write(
        "VariantVision demonstrates how bioinformatics tools can "
        "help compare protein sequences and explore documented "
        "genetic variants."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧬 Protein Analysis")
        st.write(
            "- Paste protein sequences\n"
            "- Upload FASTA files\n"
            "- Perform global pairwise alignment\n"
            "- Identify sequence differences\n"
            "- Calculate protein properties\n"
            "- Download analysis results"
        )

    with col2:
        st.subheader("🔎 Variant Database Search")
        st.write(
            "- Search by gene or disease name\n"
            "- Retrieve ClinVar record summaries\n"
            "- View reported clinical interpretations\n"
            "- Review database record links\n"
            "- Explore documented variant information"
        )

    st.divider()

    st.subheader("How the platform works")

    st.code(
        "Sequence Input / Database Search\n"
        "             ↓\n"
        "   Data Validation / Retrieval\n"
        "             ↓\n"
        "  Alignment / Variant Exploration\n"
        "             ↓\n"
        "   Properties and Results Table\n"
        "             ↓\n"
        "       Download Results",
        language="text"
    )

    st.caption(
        "Start with either analysis mode using the sidebar."
    )


# --------------------------------------------------
# GENERAL PROTEIN ANALYSIS
# --------------------------------------------------

elif mode == "General Protein Analysis":

    st.header("General Protein Analysis")

    st.write(
        "Enter a reference protein and a variant protein. "
        "Each sequence may be pasted or supplied as a FASTA file."
    )

    st.caption(
        "If a FASTA file is uploaded for a sequence, it takes "
        "priority over the text box for that sequence."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Reference Protein")

        ref_file = st.file_uploader(
            "Upload reference FASTA",
            type=["fasta", "fa", "faa", "txt"],
            key="reference_file"
        )

        ref_text = st.text_area(
            "Or paste the reference sequence",
            value=HBB_REFERENCE,
            height=160,
            key="reference_text"
        )

    with right:
        st.subheader("Variant Protein")

        var_file = st.file_uploader(
            "Upload variant FASTA",
            type=["fasta", "fa", "faa", "txt"],
            key="variant_file"
        )

        var_text = st.text_area(
            "Or paste the variant sequence",
            value=HBB_VARIANT,
            height=160,
            key="variant_text"
        )

    st.caption(
        "Default example: an HBB protein sequence with a single "
        "amino-acid substitution. These defaults are examples, "
        "not a clinical sequence-validation service."
    )

    run_analysis = st.button(
        "Analyse Protein Sequences",
        type="primary",
        use_container_width=True
    )

    if run_analysis:
        try:
            reference = get_sequence(ref_text, ref_file)
            variant = get_sequence(var_text, var_file)

            with st.spinner("Aligning and analysing sequences..."):
                alignment, changes = compare_sequences(
                    reference, variant
                )

                properties = pd.DataFrame([
                    protein_properties(reference, "Reference"),
                    protein_properties(variant, "Variant")
                ])

            st.success("Sequence analysis completed.")

            # Summary metrics
            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Reference length",
                f"{len(reference)} aa"
            )

            m2.metric(
                "Variant length",
                f"{len(variant)} aa"
            )

            m3.metric(
                "Differences detected",
                len(changes)
            )

            # Alignment
            st.subheader("1. Global Pairwise Alignment")

            st.write(
                "Alignment score:",
                round(float(alignment.score), 2)
            )

            with st.expander("View aligned sequences", expanded=True):
                st.text(alignment.format())

            st.caption(
                "Alignment uses match = +2, mismatch = -1, "
                "gap opening = -5 and gap extension = -0.5. "
                "The chosen alignment can depend on these scoring settings."
            )

            # Mutations
            st.subheader("2. Detected Sequence Differences")

            if changes:
                mutation_df = pd.DataFrame(changes)
                st.dataframe(
                    mutation_df,
                    use_container_width=True,
                    hide_index=True
                )

                mutation_csv = mutation_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Mutation Report (CSV)",
                    data=mutation_csv,
                    file_name="variantvision_mutations.csv",
                    mime="text/csv"
                )

            else:
                st.success(
                    "No amino-acid differences were detected "
                    "between these sequences."
                )

            # Properties
            st.subheader("3. Protein Physicochemical Properties")

            st.dataframe(
                properties,
                use_container_width=True,
                hide_index=True
            )

            st.write("Molecular weight comparison")

            chart_df = properties.set_index("Sequence")[
                ["Molecular weight (Da)"]
            ]

            st.bar_chart(chart_df)

            st.write("Theoretical isoelectric point (pI)")

            pi_df = properties.set_index("Sequence")[
                ["Theoretical pI"]
            ]

            st.bar_chart(pi_df)

            st.write("GRAVY score")

            gravy_df = properties.set_index("Sequence")[["GRAVY"]]

            st.bar_chart(gravy_df)

            properties_csv = properties.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Protein Properties (CSV)",
                data=properties_csv,
                file_name="variantvision_protein_properties.csv",
                mime="text/csv"
            )

            # Export a simple combined report
            report_lines = [
                "VariantVision Protein Analysis Report",
                "=" * 40,
                f"Reference length: {len(reference)} amino acids",
                f"Variant length: {len(variant)} amino acids",
                f"Alignment score: {alignment.score}",
                f"Differences detected: {len(changes)}",
                "",
                "REFERENCE SEQUENCE",
                reference,
                "",
                "VARIANT SEQUENCE",
                variant,
                "",
                "ALIGNMENT",
                alignment.format(),
                "",
                "SEQUENCE DIFFERENCES"
            ]

            if changes:
                for change in changes:
                    report_lines.append(
                        f"{change['Type']}: {change['Change']} "
                        f"(reference position: "
                        f"{change['Reference position']})"
                    )
            else:
                report_lines.append("No differences detected.")

            report_lines.extend([
                "",
                "INTERPRETATION NOTE",
                "Sequence differences alone do not establish "
                "clinical significance or disease causation.",
                "Insertion/deletion representations may depend "
                "on alignment choices, particularly in repeated regions."
            ])

            st.download_button(
                "Download Combined Analysis Report (TXT)",
                data="\n".join(report_lines),
                file_name="variantvision_analysis_report.txt",
                mime="text/plain"
            )

            st.subheader("4. Interpretation")

            st.write(
                "A substitution changes one amino acid to another. "
                "An insertion adds amino acid residue(s), while a "
                "deletion removes residue(s). Differences in length, "
                "molecular weight, pI or GRAVY describe sequence or "
                "property changes; they do not by themselves prove "
                "a harmful biological effect."
            )

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "Analysis could not be completed. Check your input "
                "sequences and try again."
            )
            st.caption(f"Technical detail: {error}")


# --------------------------------------------------
# DISEASE / GENE SEARCH
# --------------------------------------------------

elif mode == "Disease / Gene Search":

    st.header("Disease / Gene Variant Search")

    st.write(
        "Search NCBI ClinVar for documented variant records. "
        "The results are retrieved online, so an internet connection "
        "is required."
    )

    search_type = st.radio(
        "Search by",
        ["Gene", "Disease / phenotype"],
        horizontal=True
    )

    query = st.text_input(
        "Enter a gene or disease name",
        placeholder="Examples: HBB, CFTR, sickle cell disease"
    )

    max_records = st.selectbox(
        "Maximum number of records to retrieve",
        [5, 10, 20, 50],
        index=1
    )

    st.caption(
        "Gene searches use the ClinVar gene field. Disease searches "
        "use a general ClinVar text search, so some returned records "
        "may not directly match the intended phenotype."
    )

    search_button = st.button(
        "Search ClinVar",
        type="primary",
        use_container_width=True
    )

    if search_button:
        if not query.strip():
            st.warning("Please enter a gene or disease name.")

        else:
            try:
                with st.spinner("Searching NCBI ClinVar..."):
                    records = fetch_clinvar_records(
                        query,
                        search_type,
                        max_records
                    )

                if not records:
                    st.warning(
                        "No records were returned for this search. "
                        "Try a recognised gene symbol or a different "
                        "disease/phenotype term."
                    )

                else:
                    st.success(
                        f"Retrieved {len(records)} ClinVar record(s)."
                    )

                    results_df = pd.DataFrame(records)

                    # Keep the main table readable
                    display_columns = [
                        "ClinVar ID",
                        "Accession",
                        "Title / Variant",
                        "Gene",
                        "Associated phenotype",
                        "Clinical significance",
                        "Review status"
                    ]

                    st.dataframe(
                        results_df[display_columns],
                        use_container_width=True,
                        hide_index=True
                    )

                    # Provide direct links to the source records
                    st.subheader("Individual Record Details")

                    for record in records:
                        heading = (
                            f"ClinVar {record['ClinVar ID']} — "
                            f"{record['Title / Variant']}"
                        )

                        with st.expander(heading):
                            st.write(
                                "**Accession:**",
                                record["Accession"] or "Not listed"
                            )

                            st.write(
                                "**Gene:**",
                                record["Gene"]
                            )

                            st.write(
                                "**Associated phenotype:**",
                                record["Associated phenotype"]
                            )

                            st.write(
                                "**Clinical significance:**",
                                record["Clinical significance"]
                            )

                            st.write(
                                "**Review status:**",
                                record["Review status"]
                            )

                            st.write(
                                "**Variant type:**",
                                record["Variant type"] or "Not listed"
                            )

                            st.markdown(
                                f"[Open original ClinVar record]"
                                f"({record['ClinVar URL']})"
                            )

                    csv_data = results_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download ClinVar Search Results (CSV)",
                        data=csv_data,
                        file_name="variantvision_clinvar_results.csv",
                        mime="text/csv"
                    )

                    st.caption(
                        "Search results reflect the records returned "
                        "by NCBI at the time of the query. They are not "
                        "a complete list of every possible variant."
                    )

            except requests.exceptions.Timeout:
                st.error(
                    "NCBI took too long to respond. Please try again."
                )

            except requests.exceptions.RequestException:
                st.error(
                    "Could not connect to NCBI. Check the internet "
                    "connection or try again later."
                )

            except Exception as error:
                st.error(
                    "The ClinVar search could not be completed."
                )
                st.caption(f"Technical detail: {error}")

    st.divider()

    st.subheader("How to interpret the results")

    st.write(
        "- **Clinical significance:** the interpretation reported "
        "in the retrieved ClinVar summary.\n"
        "- **Review status:** information about the review level "
        "shown in the summary.\n"
        "- **Accession and ClinVar ID:** identifiers that help you "
        "locate the original database record.\n"
        "- **Associated phenotype:** the phenotype or condition "
        "linked to the record in the returned summary."
    )

    st.warning(
        "Do not interpret a database result as an individual diagnosis. "
        "Clinical significance can differ across records and may change "
        "as evidence is updated. Check the original ClinVar entry."
    )

    st.markdown(
        "[NCBI ClinVar — official database]"
        "(https://www.ncbi.nlm.nih.gov/clinvar/)"
    )


# --------------------------------------------------
# ABOUT THE PROJECT
# --------------------------------------------------

elif mode == "About the Project":

    st.header("About VariantVision")

    st.subheader("Project objective")

    st.write(
        "To develop an interactive bioinformatics application that "
        "compares protein sequences, detects amino-acid differences, "
        "calculates physicochemical properties and retrieves "
        "documented genetic variant records."
    )

    st.subheader("Tools and technologies")

    tools_df = pd.DataFrame({
        "Tool": [
            "Python",
            "Streamlit",
            "Biopython",
            "Pandas",
            "Requests",
            "NCBI ClinVar E-utilities"
        ],
        "Purpose": [
            "Core programming language",
            "Browser-based interactive interface",
            "Protein alignment and sequence properties",
            "Tabular data handling and CSV export",
            "Online HTTP requests",
            "Search and retrieval of variant records"
        ]
    })

    st.dataframe(
        tools_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Bioinformatics concepts covered")

    st.write(
        "1. Protein sequence analysis\n"
        "2. Global pairwise sequence alignment\n"
        "3. Substitution, insertion and deletion detection\n"
        "4. Protein molecular weight and theoretical pI\n"
        "5. GRAVY and basic physicochemical analysis\n"
        "6. Biological database search and retrieval\n"
        "7. Comparative protein analysis\n"
        "8. Interpretation of documented variant records"
    )

    st.subheader("Limitations")

    st.write(
        "- Protein alignment depends on the selected scoring scheme.\n"
        "- Alignment-derived mutation descriptions may be ambiguous "
        "in repeated sequence regions.\n"
        "- Protein properties are theoretical estimates.\n"
        "- ClinVar search results depend on the query and available "
        "database records.\n"
        "- The application does not validate clinical variants or "
        "predict disease risk."
    )

    st.subheader("Data source")

    st.markdown(
        "[NCBI ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)"
    )

    st.caption(
        "ClinVar is maintained by the National Center for "
        "Biotechnology Information (NCBI)."
    )

    st.caption(
        "VariantVision is an educational student project and is "
        "not an official NCBI application."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "VariantVision | Educational Bioinformatics Project | "
    "Protein Analysis and Variant Exploration"
)
