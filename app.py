
import re
import requests
import pandas as pd
import streamlit as st

from Bio.Align import PairwiseAligner
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="VariantVision",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 VariantVision")
st.subheader(
    "Protein Sequence Comparison, Mutation Detection "
    "and Genetic Variant Exploration"
)

st.write(
    "An interactive bioinformatics platform for comparing protein "
    "sequences, analysing physicochemical properties, and exploring "
    "documented genetic variant records."
)

st.info(
    "Educational project only. This application does not diagnose "
    "diseases or independently determine whether a variant is harmful."
)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Example beta-globin protein sequence.
# The substitution at position 7 changes E to V.
HBB_REFERENCE = (
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVK"
    "AHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFT"
    "PPVQAAYQKVVAGVANALAHKYH"
)

HBB_VARIANT = HBB_REFERENCE[:6] + "V" + HBB_REFERENCE[7:]


# =========================================================
# 2. SEQUENCE INPUT AND VALIDATION
# =========================================================

def clean_sequence(raw_text):
    """Read a protein sequence from plain text or FASTA."""

    lines = raw_text.strip().splitlines()

    sequence_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith(">")
    ]

    sequence = re.sub(
        r"\s+", "", "".join(sequence_lines)
    ).upper()

    if not sequence:
        raise ValueError("No protein sequence was entered or uploaded.")

    invalid = sorted(set(sequence) - VALID_AA)

    if invalid:
        raise ValueError(
            "Invalid amino-acid character(s): "
            + ", ".join(invalid)
            + ". Please use the 20 standard amino-acid letters."
        )

    return sequence


def get_sequence(pasted_text, uploaded_file):
    """Prefer the uploaded file when one is provided."""

    if uploaded_file is not None:
        raw_text = uploaded_file.getvalue().decode("utf-8")
    else:
        raw_text = pasted_text

    return clean_sequence(raw_text)


# =========================================================
# 3. PROTEIN ALIGNMENT AND DIFFERENCE DETECTION
# =========================================================

def compare_sequences(reference, variant):
    """Perform global alignment and identify sequence differences."""

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5

    alignment = aligner.align(reference, variant)[0]
    coordinates = alignment.coordinates
    changes = []

    for i in range(coordinates.shape[1] - 1):

        r0 = int(coordinates[0, i])
        r1 = int(coordinates[0, i + 1])
        v0 = int(coordinates[1, i])
        v1 = int(coordinates[1, i + 1])

        ref_length = r1 - r0
        var_length = v1 - v0

        # Aligned blocks containing residues in both sequences
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

                changes.append({
                    "Type": "Complex alignment block",
                    "Reference position": f"{r0 + 1}-{r1}",
                    "Reference residue": reference[r0:r1],
                    "Variant position": f"{v0 + 1}-{v1}",
                    "Variant residue": variant[v0:v1],
                    "Change": "Review aligned sequences"
                })

        # Residues present in the reference but absent in the variant
        elif ref_length > 0:

            deleted = reference[r0:r1]

            changes.append({
                "Type": "Deletion",
                "Reference position": (
                    f"{r0 + 1}-{r1}"
                    if ref_length > 1
                    else r0 + 1
                ),
                "Reference residue": deleted,
                "Variant position": v0,
                "Variant residue": "-",
                "Change": f"del {deleted}"
            })

        # Residues present in the variant but absent in the reference
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


# =========================================================
# 4. PHYSICOCHEMICAL PROPERTIES
# =========================================================

def protein_properties(sequence, label):

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


# =========================================================
# 5. CLINVAR DATA EXTRACTION HELPERS
# =========================================================

def text_value(value):
    """Convert common ClinVar API values into readable text."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        items = [text_value(item) for item in value]
        return "; ".join(item for item in items if item)

    if isinstance(value, dict):

        for key in (
            "description",
            "trait_name",
            "symbol",
            "name",
            "value"
        ):
            if value.get(key):
                return text_value(value[key])

    return ""


def extract_genes(record):
    """Extract gene symbols from root and nested variant structures."""

    gene_names = []

    def add_genes(items):
        if not isinstance(items, list):
            return

        for item in items:

            if isinstance(item, dict):
                symbol = item.get("symbol") or item.get("name")

                if symbol:
                    gene_names.append(str(symbol))

            elif isinstance(item, str):
                gene_names.append(item)

    add_genes(record.get("genes", []))

    # ClinVar summaries may place gene data inside variation_set.
    variation_set = record.get("variation_set", [])

    if isinstance(variation_set, list):
        for variation in variation_set:

            if isinstance(variation, dict):
                add_genes(variation.get("genes", []))

    return list(dict.fromkeys(gene_names))


def extract_phenotypes(record):
    """Extract phenotype names from available summary fields."""

    names = []

    possible_fields = [
        "trait_set",
        "traits",
        "germline_classification",
        "clinical_impact_classification"
    ]

    for field in possible_fields:

        value = record.get(field)

        if field in ("trait_set", "traits"):

            if isinstance(value, list):

                for item in value:

                    if isinstance(item, dict):

                        name = (
                            item.get("trait_name")
                            or item.get("name")
                            or item.get("description")
                        )

                        if name:
                            names.append(str(name))

                    elif isinstance(item, str):
                        names.append(item)

    return list(dict.fromkeys(names))


def extract_classification(record):
    """Read current ClinVar classification summaries."""

    classifications = []
    review_statuses = []

    classification_fields = [
        ("Germline", "germline_classification"),
        ("Somatic clinical impact", "clinical_impact_classification"),
        ("Oncogenicity", "oncogenicity_classification")
    ]

    for label, field in classification_fields:

        item = record.get(field)

        if isinstance(item, dict):

            description = text_value(item.get("description"))

            if description:
                classifications.append(
                    f"{label}: {description}"
                )

            review = text_value(item.get("review_status"))

            if review:
                review_statuses.append(
                    f"{label}: {review}"
                )

    # Support older summaries that expose clinical_significance.
    old_significance = record.get("clinical_significance")

    if isinstance(old_significance, dict):

        description = text_value(
            old_significance.get("description")
        )

        if description and not classifications:
            classifications.append(description)

        review = text_value(
            old_significance.get("review_status")
        )

        if review and not review_statuses:
            review_statuses.append(review)

    elif isinstance(old_significance, str) and old_significance:

        if not classifications:
            classifications.append(old_significance)

    significance_text = (
        "; ".join(classifications)
        if classifications
        else "Not reported in this summary"
    )

    review_text = (
        "; ".join(review_statuses)
        if review_statuses
        else "Not reported in this summary"
    )

    return significance_text, review_text


def extract_variant_type(record):
    """Extract variant type from the root or nested variation records."""

    variant_types = []

    root_type = text_value(record.get("variant_type"))

    if root_type:
        variant_types.append(root_type)

    variation_set = record.get("variation_set", [])

    if isinstance(variation_set, list):

        for variation in variation_set:

            if not isinstance(variation, dict):
                continue

            variation_type = (
                variation.get("variation_type")
                or variation.get("variant_type")
            )

            variation_type = text_value(variation_type)

            if variation_type:
                variant_types.append(variation_type)

    return "; ".join(dict.fromkeys(variant_types))


# =========================================================
# 6. SEARCH NCBI CLINVAR
# =========================================================

def fetch_clinvar_records(query, search_type, max_records):
    """Search ClinVar and return structured variant summaries."""

    query = query.strip()

    if not query:
        return []

    if search_type == "Gene":
        search_term = f"{query}[gene]"
    else:
        search_term = query

    headers = {
        "User-Agent": "VariantVisionEducationalProject/1.0"
    }

    # Step 1: Retrieve matching ClinVar record IDs.
    search_response = requests.get(
        NCBI_BASE + "esearch.fcgi",
        params={
            "db": "clinvar",
            "term": search_term,
            "retmax": max_records,
            "retmode": "json"
        },
        headers=headers,
        timeout=30
    )

    search_response.raise_for_status()

    search_data = search_response.json()

    ids = search_data.get(
        "esearchresult", {}
    ).get("idlist", [])

    if not ids:
        return []

    # Step 2: Retrieve summaries for the record IDs.
    summary_response = requests.get(
        NCBI_BASE + "esummary.fcgi",
        params={
            "db": "clinvar",
            "id": ",".join(ids),
            "retmode": "json"
        },
        headers=headers,
        timeout=30
    )

    summary_response.raise_for_status()

    summary_data = summary_response.json()
    result_data = summary_data.get("result", {})

    records = []

    for record_id in ids:

        record = result_data.get(str(record_id), {})

        if not isinstance(record, dict):
            continue

        genes = extract_genes(record)
        phenotypes = extract_phenotypes(record)

        significance, review_status = extract_classification(record)

        variant_type = extract_variant_type(record)

        accession = text_value(record.get("accession"))
        title = text_value(record.get("title"))

        if not title:
            title = "Title not supplied in summary"

        # Some summaries do not contain all phenotype details.
        phenotype_text = (
            "; ".join(phenotypes)
            if phenotypes
            else "Not reported in summary; open the source record"
        )

        gene_text = (
            ", ".join(genes)
            if genes
            else "Not reported in summary"
        )

        variant_url = (
            "https://www.ncbi.nlm.nih.gov/clinvar/variation/"
            + str(record_id) + "/"
        )

        records.append({
            "ClinVar ID": str(record_id),
            "Accession": accession or "Not listed",
            "Title / Variant": title,
            "Gene": gene_text,
            "Associated phenotype": phenotype_text,
            "Clinical significance": significance,
            "Review status": review_status,
            "Variant type": variant_type or "Not reported in summary",
            "ClinVar URL": variant_url
        })

    return records


# =========================================================
# 7. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🧬 VariantVision")

mode = st.sidebar.radio(
    "Choose a section",
    [
        "Home",
        "General Protein Analysis",
        "Disease / Gene Search",
        "About the Project"
    ]
)


# =========================================================
# 8. HOME PAGE
# =========================================================

if mode == "Home":

    st.header("Welcome to VariantVision")

    st.write(
        "Explore protein sequence differences and retrieve "
        "documented genetic variant records through an interactive "
        "bioinformatics dashboard."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🧬 General Protein Analysis")

        st.write(
            "- Paste protein sequences\n"
            "- Upload FASTA files\n"
            "- Perform global pairwise alignment\n"
            "- Detect substitutions, insertions and deletions\n"
            "- Calculate protein physicochemical properties\n"
            "- Download analysis reports"
        )

    with col2:

        st.subheader("🔎 Disease / Gene Search")

        st.write(
            "- Search ClinVar online\n"
            "- Enter a gene or disease name\n"
            "- View available variant summaries\n"
            "- Inspect reported interpretations\n"
            "- Open original database records\n"
            "- Export search results"
        )

    st.divider()

    st.subheader("Project workflow")

    st.code(
        "Sequence Input or Gene/Disease Query\n"
        "                 ↓\n"
        "        Data Validation / Retrieval\n"
        "                 ↓\n"
        "   Alignment / Variant Record Exploration\n"
        "                 ↓\n"
        "      Properties and Results Table\n"
        "                 ↓\n"
        "           Export Results",
        language="text"
    )

    st.caption(
        "Choose a section from the sidebar to get started."
    )


# =========================================================
# 9. GENERAL PROTEIN ANALYSIS
# =========================================================

elif mode == "General Protein Analysis":

    st.header("General Protein Analysis")

    st.write(
        "Provide a reference protein and a variant protein. "
        "You can paste the sequences or upload FASTA files."
    )

    st.caption(
        "If a file is uploaded for a sequence, the uploaded file "
        "takes priority over the corresponding text box."
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
            "Or paste reference sequence",
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
            "Or paste variant sequence",
            value=HBB_VARIANT,
            height=160,
            key="variant_text"
        )

    st.caption(
        "The default example changes amino acid E at position 7 to V. "
        "Protein numbering conventions may differ from mature-chain "
        "numbering; this example is for educational comparison."
    )

    if st.button(
        "Analyse Protein Sequences",
        type="primary",
        use_container_width=True
    ):

        try:

            reference = get_sequence(ref_text, ref_file)
            variant = get_sequence(var_text, var_file)

            with st.spinner("Analysing protein sequences..."):

                alignment, changes = compare_sequences(
                    reference, variant
                )

                properties = pd.DataFrame([
                    protein_properties(reference, "Reference"),
                    protein_properties(variant, "Variant")
                ])

            st.success("Analysis completed.")

            # Summary
            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Reference length",
                f"{len(reference)} aa"
            )

            c2.metric(
                "Variant length",
                f"{len(variant)} aa"
            )

            c3.metric(
                "Differences detected",
                len(changes)
            )

            # Alignment
            st.subheader("1. Global Pairwise Alignment")

            st.write(
                f"Alignment score: {float(alignment.score):.2f}"
            )

            with st.expander("View alignment", expanded=True):
                st.text(alignment.format())

            st.caption(
                "Scoring: match +2, mismatch -1, gap opening -5, "
                "gap extension -0.5."
            )

            # Difference table
            st.subheader("2. Detected Sequence Differences")

            if changes:

                mutation_df = pd.DataFrame(changes)

                st.dataframe(
                    mutation_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Mutation Results (CSV)",
                    data=mutation_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="variantvision_mutations.csv",
                    mime="text/csv"
                )

            else:

                st.success(
                    "No amino-acid differences were detected."
                )

            # Protein properties
            st.subheader("3. Physicochemical Properties")

            st.dataframe(
                properties,
                use_container_width=True,
                hide_index=True
            )

            st.write("Molecular weight")

            st.bar_chart(
                properties.set_index("Sequence")[
                    ["Molecular weight (Da)"]
                ]
            )

            st.write("Theoretical isoelectric point (pI)")

            st.bar_chart(
                properties.set_index("Sequence")[
                    ["Theoretical pI"]
                ]
            )

            st.write("GRAVY score")

            st.bar_chart(
                properties.set_index("Sequence")[["GRAVY"]]
            )

            st.download_button(
                "Download Protein Properties (CSV)",
                data=properties.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="variantvision_properties.csv",
                mime="text/csv"
            )

            # Combined report
            report = [
                "VARIANTVISION PROTEIN ANALYSIS REPORT",
                "=" * 42,
                f"Reference length: {len(reference)} amino acids",
                f"Variant length: {len(variant)} amino acids",
                f"Alignment score: {alignment.score}",
                f"Detected differences: {len(changes)}",
                "",
                "REFERENCE SEQUENCE",
                reference,
                "",
                "VARIANT SEQUENCE",
                variant,
                "",
                "GLOBAL ALIGNMENT",
                alignment.format(),
                "",
                "DETECTED DIFFERENCES"
            ]

            if changes:

                for change in changes:

                    report.append(
                        f"{change['Type']}: {change['Change']} | "
                        f"Reference position: "
                        f"{change['Reference position']}"
                    )

            else:

                report.append("No differences detected.")

            report.extend([
                "",
                "INTERPRETATION",
                "Sequence differences and physicochemical properties "
                "do not independently establish clinical significance.",
                "Alignment-dependent insertion/deletion descriptions "
                "may require further review."
            ])

            st.download_button(
                "Download Combined Report (TXT)",
                data="\n".join(report),
                file_name="variantvision_report.txt",
                mime="text/plain"
            )

            st.subheader("4. Biological Interpretation")

            st.write(
                "A substitution replaces one amino acid with another. "
                "An insertion adds amino acid residues, while a deletion "
                "removes residues. Changes in molecular weight, pI and "
                "GRAVY describe theoretical protein properties. These "
                "measurements alone cannot determine whether a variant "
                "causes disease."
            )

        except ValueError as error:

            st.error(str(error))

        except Exception as error:

            st.error(
                "Protein analysis could not be completed. "
                "Check the input and try again."
            )

            st.caption(f"Technical detail: {error}")


# =========================================================
# 10. DISEASE / GENE SEARCH
# =========================================================

elif mode == "Disease / Gene Search":

    st.header("Disease / Gene Variant Search")

    st.write(
        "Search the NCBI ClinVar database for variant records "
        "associated with a gene or disease/phenotype term."
    )

    search_type = st.radio(
        "Search type",
        ["Gene", "Disease / phenotype"],
        horizontal=True
    )

    query = st.text_input(
        "Enter a gene or disease name",
        placeholder="Examples: HBB, CFTR, sickle cell disease"
    )

    max_records = st.selectbox(
        "Maximum records to retrieve",
        [5, 10, 20, 50],
        index=1
    )

    st.caption(
        "Gene searches use the ClinVar gene field. Disease searches "
        "use a general ClinVar text query, so inspect each returned "
        "record to confirm relevance."
    )

    if st.button(
        "Search ClinVar",
        type="primary",
        use_container_width=True
    ):

        if not query.strip():

            st.warning("Enter a gene or disease name first.")

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
                        "No records were returned. Try a recognised "
                        "gene symbol or another search term."
                    )

                else:

                    st.success(
                        f"Retrieved {len(records)} ClinVar record(s)."
                    )

                    results_df = pd.DataFrame(records)

                    display_columns = [
                        "ClinVar ID",
                        "Accession",
                        "Title / Variant",
                        "Gene",
                        "Associated phenotype",
                        "Clinical significance",
                        "Review status",
                        "Variant type"
                    ]

                    st.dataframe(
                        results_df[display_columns],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader("Record Details and Source Links")

                    for record in records:

                        heading = (
                            f"ClinVar {record['ClinVar ID']} — "
                            f"{record['Title / Variant']}"
                        )

                        with st.expander(heading):

                            st.write(
                                "**Accession:**",
                                record["Accession"]
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
                                record["Variant type"]
                            )

                            st.markdown(
                                f"[Open original ClinVar record]"
                                f"({record['ClinVar URL']})"
                            )

                    st.download_button(
                        "Download ClinVar Results (CSV)",
                        data=results_df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="variantvision_clinvar_results.csv",
                        mime="text/csv"
                    )

                    st.caption(
                        "The fields shown above are extracted from "
                        "NCBI ESummary records. Some fields may be "
                        "missing from a summary even when additional "
                        "information is available on the source page."
                    )

            except requests.exceptions.Timeout:

                st.error(
                    "NCBI took too long to respond. Please try again."
                )

            except requests.exceptions.RequestException:

                st.error(
                    "Could not connect to NCBI. Check the connection "
                    "or try again later."
                )

            except Exception as error:

                st.error(
                    "The ClinVar search could not be completed."
                )

                st.caption(f"Technical detail: {error}")

    st.divider()

    st.subheader("Understanding ClinVar Results")

    st.write(
        "**Clinical significance:** the classification reported "
        "in the retrieved summary, when available."
    )

    st.write(
        "**Review status:** the review information provided with "
        "the classification, when available."
    )

    st.write(
        "**Associated phenotype:** a condition or phenotype linked "
        "to the record in the returned summary."
    )

    st.write(
        "**Variant type:** the kind of sequence variation described "
        "by the record, when included in the summary."
    )

    st.warning(
        "Missing fields do not necessarily mean that information "
        "is absent from the complete record. Open the original entry "
        "to inspect its detailed data and evidence. VariantVision "
        "does not independently validate clinical interpretations."
    )

    st.markdown(
        "[Visit the official NCBI ClinVar database]"
        "(https://www.ncbi.nlm.nih.gov/clinvar/)"
    )


# =========================================================
# 11. ABOUT THE PROJECT
# =========================================================

elif mode == "About the Project":

    st.header("About VariantVision")

    st.subheader("Objective")

    st.write(
        "To develop an interactive bioinformatics application that "
        "compares protein sequences, identifies amino-acid differences, "
        "calculates physicochemical properties, and retrieves "
        "documented genetic variant records."
    )

    st.subheader("Tools and Technologies")

    technology_df = pd.DataFrame({
        "Technology": [
            "Python",
            "Streamlit",
            "Biopython",
            "Pandas",
            "Requests",
            "NCBI ClinVar E-utilities"
        ],
        "Purpose": [
            "Core programming language",
            "Interactive browser-based interface",
            "Sequence alignment and protein properties",
            "Tabular data management and CSV export",
            "HTTP requests to online services",
            "Search and retrieval of variant summaries"
        ]
    })

    st.dataframe(
        technology_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Bioinformatics Concepts")

    st.write(
        "1. Protein sequence analysis\n"
        "2. Global pairwise sequence alignment\n"
        "3. Substitution, insertion and deletion detection\n"
        "4. Molecular weight and theoretical isoelectric point\n"
        "5. GRAVY score and physicochemical properties\n"
        "6. Biological database search\n"
        "7. Comparative protein analysis\n"
        "8. Exploration of documented genetic variants"
    )

    st.subheader("Limitations")

    st.write(
        "- Alignment results depend on scoring parameters.\n"
        "- Insertions and deletions can have ambiguous representations "
        "in repeated regions.\n"
        "- Protein properties are theoretical estimates.\n"
        "- Online search results depend on the query and database records.\n"
        "- Summary responses may not contain every field from the full record.\n"
        "- The app is not a clinical interpretation or diagnostic system."
    )

    st.subheader("Data Source")

    st.markdown(
        "[NCBI ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)"
    )

    st.caption(
        "VariantVision is an educational student project and is not "
        "an official NCBI application."
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.caption(
    "VariantVision | Educational Bioinformatics Project | "
    "Protein Sequence Analysis and Variant Exploration"
)
