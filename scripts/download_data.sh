#!/usr/bin/env bash
# Download the public inputs into data/raw/.
# Record the ClinVar release date you use: results must be reproducible.
set -euo pipefail
RAW=${1:-data/raw}
mkdir -p "$RAW"
cd "$RAW"

# 1. ClinVar (GRCh38). Replace "clinvar.vcf.gz" with a dated file from
#    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/ or /archive_2.0/
wget -nc https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
wget -nc https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi

# 2. GRCh38 reference genome (Ensembl primary assembly, chromosome names without "chr").
if [ ! -f GRCh38.fa ]; then
  wget -nc https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
  gunzip -c Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz > GRCh38.fa
  samtools faidx GRCh38.fa
fi

# 3. CADD v1.7 precomputed SNV scores (~80 GB). Skip this if you will use the
#    CADD web service instead (see README, "CADD baseline").
if [ "${DOWNLOAD_CADD:-0}" = "1" ]; then
  wget -nc https://kircherlab.bihealth.org/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz
  wget -nc https://kircherlab.bihealth.org/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz.tbi
fi

echo "Done. Files in $(pwd)"
