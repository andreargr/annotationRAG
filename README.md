# **RAG-Based Multi-Ontology Annotation Framework**

This repository contains the code, experiments, and analysis for a study proposing a new **Retrieval-Augmented Generation (RAG)** approach tailored for the automated annotation of biological sample metadata.
## **Previous experiments github**
[Base model + Fine-tuned model + RAG with BioPortal](https://github.com/andreargr/BiosamplesRAGAnnotation)
## **Overview**

Traditional metadata annotation pipelines often rely on single-ontology retrieval or focus on isolated biological sample categories, which limits the richness and robustness of the resulting annotations. In this study, we introduce a **two-stage RAG-based method** designed to overcome these limitations:

1. **Interpretation Stage**
   The *GPT-5-mini* model infers and extracts contextual information from each biological sample label, producing an enriched and standardized interpretation.

2. **Annotation Stage**
   This refined interpretation is used as input to generate more accurate, ontology-aligned annotations with *GPT-4o-mini*.

To support the retrieval phase, we use a **reduced and semantically focused subset** of four major biological ontologies. This compact knowledge base minimizes noise, improves similarity search precision, and avoids the computational overhead of large-scale ontology retrieval.

Unlike most previous studies:

* We annotate **across up to four ontologies**, enabling richer multi-dimensional metadata standardization.
* We cover **three biological entity categories** — *cell lines*, *cell types*, and *anatomical structures* — expanding the scope beyond single-type annotation approaches.
* Our pipeline is designed to **automate the majority of the annotation workflow**, while keeping human validation as a final quality-control layer.

---

## **Repository Structure**

```
analysis/
│── error_analysis/
│   └── Scripts for detailed error inspection.
│── evaluation_scripts/
│   └── Scripts for metric computation and performance evaluation.
│── method_comparison/
│   └── Tools for comparing the proposed method with other approaches.

k_candidates/
│── 5K/
│── 10K/
│── scripts/
│── candidate_positions.tsv
    Experiments and resources used to determine the optimal number of retrieved candidates
    for the RAG component.

RAG_process/
│── RAG_reduced_info_results/
│   └── Output annotations obtained using the reduced-information ontology subset.
│── scripts/
    Scripts implementing the RAG pipeline, including prompt templates and RAG logic.
```

---

## **Folder Descriptions**

### **RAG_process/**

Contains all code related to the main RAG system:

* **scripts/**
  Scripts implementing the two-stage RAG pipeline (interpretation → annotation), prompt templates, retrieval configuration, and ontology preprocessing.
* **RAG_reduced_info_results/**
  Generated annotations using the reduced semantic ontology subset.

### **k_candidates/**

Includes experiments conducted to determine the optimal number of retrieved candidates for the similarity search stage:

* Tests for **5K** and **10K** candidate configurations.
* **candidate_positions.tsv** logs candidate distributions and behavior.
* **scripts/** contains all associated experimental code.

### **analysis/**

Contains the full evaluation suite for the study:

* **error_analysis/** — inspection of annotation errors and qualitative assessment.
* **evaluation_scripts/** — computation of precision, recall, and other metrics.
* **method_comparison/** — comparisons with other annotation methods mentioned in the paper.

---

## **Summary of Contributions**

* A **two-stage RAG methodology** that improves annotation precision without requiring fine-tuning.
* Use of a **semantically filtered multi-ontology knowledge base** for efficient, noise-reduced retrieval.
* Annotation across **four biological ontologies**, capturing complementary biological dimensions.
* Application to **three distinct biological sample categories**.
* A complete experimental and evaluation pipeline for reproducibility.

